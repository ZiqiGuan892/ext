"""Opt-in, sample-keyed text capture for a validation-only Huanyu A/B run.

HUANYU_TEXT_AB_DIR enables this hook. It observes the existing single-turn
AgentLoop result without replacing its inputs, sampling options, or outputs.
The text is the post-continuous-token-merge representation used by validation
dumps, not a claim that the serving HTTP response used identical token IDs.
"""

from collections.abc import Mapping
from functools import wraps
import hashlib
import importlib
import json
from numbers import Integral
import os
from pathlib import Path
import re
import socket
import time


_HOOK = "_huanyu_text_ab_capture"


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_run(loop):
    trainer = _field(loop.config, "trainer")
    rollout = loop.rollout_config
    if (_field(trainer, "val_only") is not True
            or _field(trainer, "val_before_train") is not True
            or _field(trainer, "resume_mode") != "disable"
            or _field(_field(rollout, "val_kwargs"), "n") != 1):
        raise ValueError("Huanyu text A/B capture requires val_only=true, val_before_train=true, "
                         "resume_mode=disable and rollout.val_kwargs.n=1")


def _messages(messages, *, media=False):
    """Canonicalize role/text/audio order without invoking any template."""
    if isinstance(messages, (str, bytes, Mapping)) or messages is None:
        raise ValueError("text A/B prompt must be a message sequence")
    result, audio_paths = [], []
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise ValueError("text A/B prompt messages require a string role")
        content = message.get("content")
        if not isinstance(content, str):
            if not media or isinstance(content, (bytes, Mapping)) or content is None:
                raise ValueError("text A/B canonical prompt content must be a string")
            parts = []
            for part in content:
                if not isinstance(part, Mapping):
                    raise ValueError("invalid text A/B raw prompt content part")
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") == "audio":
                    path = part.get("audio", part.get("audio_url"))
                    if not isinstance(path, str) or not Path(path).is_absolute():
                        raise ValueError("text A/B raw prompt requires absolute WAV paths")
                    audio_paths.append(str(Path(path).resolve(strict=True)))
                    parts.append("<audio>")
                else:
                    raise ValueError("text A/B capture accepts only text and audio input parts")
            content = "".join(parts)
        result.append({"role": message["role"], "content": content})
    if not result:
        raise ValueError("text A/B prompt is empty")
    return result, audio_paths


def _validate_metadata(kwargs):
    metadata = _field(_field(kwargs, "extra_info"), "huanyu_text_ab")
    if not isinstance(metadata, Mapping):
        raise ValueError("missing extra_info.huanyu_text_ab; prepare the shared A/B subset first")
    sample_id, case_id = metadata.get("sample_id"), metadata.get("case_id")
    if not isinstance(sample_id, str) or re.fullmatch(r"row_[0-9]{6,}", sample_id) is None:
        raise ValueError("text A/B sample_id must be row_XXXXXX")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("text A/B metadata requires case_id")
    mode = metadata.get("prompt_mode")
    if mode not in ("system_audio", "original"):
        raise ValueError("text A/B prompt_mode must be system_audio or original")
    prompt, _ = _messages(metadata.get("prompt"))
    if mode == "system_audio":
        for message in prompt:
            if (message["role"] not in ("system", "user")
                    or (message["role"] == "user" and message["content"].replace("<audio>", "").strip())):
                raise ValueError("system_audio accepts original system messages and user audio placeholders only")
    raw_prompt, actual_paths = _messages(kwargs.get("raw_prompt"), media=True)
    if prompt != raw_prompt:
        raise ValueError("text A/B raw_prompt differs from the prepared canonical prompt")
    paths, hashes = metadata.get("wav_paths"), metadata.get("wav_sha256")
    if (not isinstance(paths, (list, tuple)) or not paths
            or not isinstance(hashes, (list, tuple)) or len(hashes) != len(paths)
            or sum(m["content"].count("<audio>") for m in prompt) != len(paths)):
        raise ValueError("text A/B audio paths, hashes and prompt placeholders must have equal nonzero counts")
    resolved = []
    for path, expected_hash in zip(paths, hashes):
        if (not isinstance(path, str) or not Path(path).is_absolute()
                or not isinstance(expected_hash, str) or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None):
            raise ValueError("text A/B metadata requires absolute WAV paths and lowercase SHA256 hashes")
        path = Path(path).resolve(strict=True)
        if path.suffix.lower() != ".wav" or _file_hash(path) != expected_hash:
            raise ValueError("text A/B WAV changed or is not a WAV path: {}".format(path))
        resolved.append(str(path))
    if actual_paths != resolved:
        raise ValueError("text A/B raw_prompt audio paths/order differ from prepared WAV inputs")
    return {"sample_id": sample_id, "case_id": case_id, "prompt_mode": mode,
            "prompt": prompt, "prompt_sha256": _hash(prompt), "wav_paths": resolved,
            "wav_sha256": list(hashes)}


def _ids(value):
    result = list(value)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in result):
        raise ValueError("text A/B output token IDs must be nonnegative integers")
    return [int(v) for v in result]


def _write_json(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def install_text_ab_capture():
    """Install from configure_processor in each worker; disabled means no imports."""
    if not os.environ.get("HUANYU_TEXT_AB_DIR", "").strip():
        return False
    cls = importlib.import_module("verl.experimental.agent_loop.single_turn_agent_loop").SingleTurnAgentLoop
    original = cls.run
    if getattr(original, _HOOK, False):
        return False

    @wraps(original)
    async def run(self, sampling_params, *args, **kwargs):
        directory = os.environ.get("HUANYU_TEXT_AB_DIR", "").strip()
        if not directory or not getattr(self.processor, "_huanyu_training_processor", False):
            return await original(self, sampling_params, *args, **kwargs)
        _validate_run(self)
        if kwargs.get("session_id", 0) != 0:
            raise ValueError("text A/B expects exactly one validation generation per sample")
        metadata = _validate_metadata(kwargs)
        root = Path(directory).expanduser().resolve()
        output_dir = root / "rollout"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir.resolve().relative_to(root)
        output = output_dir / (metadata["sample_id"] + ".json")
        text_output = output.with_suffix(".txt")
        if text_output.exists():
            raise FileExistsError("text A/B output already exists: {}".format(text_output))
        # Validate/snapshot JSON-able controls before inference; never retain a
        # sampling dict the real engine might subsequently mutate in place.
        requested_sampling = json.loads(_json(sampling_params))
        record = dict(metadata, schema_version=1, status="running", engine="verl_rollout",
                      representation="post_continuous_token_merge",
                      requested_sampling_params=requested_sampling,
                      sampling_scope="incoming_agent_loop_not_final_engine",
                      checkpoint=_field(_field(_field(self.config, "actor_rollout_ref"), "model"), "path"),
                      host=socket.gethostname(), pid=os.getpid(), created_at_unix=time.time())
        # Exclusive reservation works across coroutines and Ray processes. A
        # duplicate run/sample fails instead of silently overwriting evidence.
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        started = time.monotonic()
        try:
            result = await original(self, sampling_params, *args, **kwargs)
            prompt_ids, response_ids = _ids(result.prompt_ids), _ids(result.response_ids)
            decode = self.tokenizer.decode
            record.update(
                reply=decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
                reply_with_special_tokens=decode(response_ids, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False),
                rendered_prompt_with_special_tokens=decode(prompt_ids, skip_special_tokens=False,
                                                          clean_up_tokenization_spaces=False),
                prompt_token_count=len(prompt_ids), prompt_token_sha256=_hash(prompt_ids),
                response_token_count=len(response_ids), response_token_sha256=_hash(response_ids),
                token_count_scope="post_continuous_token_merge_response_ids",
                status="complete", elapsed_seconds=time.monotonic() - started,
            )
            if not isinstance(record["reply"], str):
                raise TypeError("text A/B tokenizer.decode must return a string")
            with text_output.open("x", encoding="utf-8", newline="") as handle:
                handle.write(record["reply"])
            _write_json(output, record)
            print("[huanyu-text-ab] sample={} complete response_tokens={} output={}".format(
                metadata["sample_id"], len(response_ids), text_output), flush=True)
            return result
        except BaseException as exc:
            record.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)},
                          elapsed_seconds=time.monotonic() - started)
            _write_json(output, record)
            raise

    setattr(run, _HOOK, True)
    cls.run = run
    return True
