#!/usr/bin/env python3
"""Export AQA Parquet rows to matched prompts, answers and WAVs for serving tests.

Requires pyarrow, pydantic, numpy, numcodecs and soundfile. Keep external_audio.py
beside this file. No training imports, Ray, model loading or HTTP requests.
"""

import argparse
import base64
from collections.abc import Mapping
import io
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from external_audio import ExternalAudioReference, read_external_audio


DEFAULT_PARQUET = "/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet"


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _audio_reference(item):
    if isinstance(item, Mapping):
        item = next((item[key] for key in ("audio", "audio_url", "path", "url")
                     if item.get(key) is not None), None)
    if not isinstance(item, str) or not item.strip():
        raise ValueError("audios items must contain local file or archive::key references")
    archive, separator, key = item.partition("::")
    if separator and not key.strip():
        raise ValueError("archive reference has an empty member key: {!r}".format(item))
    if not Path(archive).is_absolute():
        raise ValueError("audio path must be absolute: {!r}".format(archive))
    return item, archive, key if separator else None


def _wav_payload(payload):
    # external_audio returns original bytes for direct files/TAR members.
    # Decode these too: giving an MP3 a .wav suffix would not convert its format.
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("audio must contain non-empty finite samples")
    output = io.BytesIO()
    sf.write(output, audio, rate, format="WAV", subtype="PCM_16")
    return output.getvalue(), {
        "sample_rate": rate,
        "channels": audio.shape[1],
        "num_frames": len(audio),
        "duration_seconds": len(audio) / rate,
    }


def _validate_row(row):
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty message list")
    for message in prompt:
        if (not isinstance(message, Mapping) or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str) or not message["content"].strip()):
            raise ValueError("each prompt message requires role and non-empty string content")
    if not any(message["role"] == "user" for message in prompt):
        raise ValueError("prompt has no user message")
    reward = row.get("reward_model")
    answer = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("reward_model.ground_truth must be a non-empty string")
    if not isinstance(row.get("audios"), list) or not row["audios"]:
        raise ValueError("audios must be a non-empty list")
    if row.get("images") or row.get("videos"):
        raise ValueError("export expects audio-only AQA rows; images/videos would be lost")
    return prompt, answer


def _request(prompt, payloads, model, max_tokens):
    """Replace each literal <audio> with its ordered input_audio content part."""
    count = sum(message["content"].count("<audio>") for message in prompt)
    if count != len(payloads):
        raise ValueError("request requires <audio> count={} to equal audios count={}".format(count, len(payloads)))
    messages = []
    audio_index = 0
    for message in prompt:
        result = dict(message)
        pieces = message["content"].split("<audio>")
        if len(pieces) > 1:
            if message["role"] != "user":
                raise ValueError("request audio placeholders must belong to user messages")
            parts = []
            for index, text in enumerate(pieces):
                if text:
                    parts.append({"type": "text", "text": text})
                if index < len(pieces) - 1:
                    parts.append({"type": "input_audio", "input_audio": {
                        "data": base64.b64encode(payloads[audio_index]).decode("ascii"), "format": "wav",
                    }})
                    audio_index += 1
            result["content"] = parts
        messages.append(result)
    return {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": False}


def export_validation(parquet_path, output_dir, limit=0, batch_size=32, model=None, max_tokens=2048):
    if limit < 0 or batch_size < 1 or max_tokens < 1:
        raise ValueError("limit must be non-negative; batch_size and max_tokens must be positive")
    if model is not None and not model.strip():
        raise ValueError("model must be a non-empty served model name")
    source = Path(parquet_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    with pq.ParquetFile(source) as reader:
        required = {"prompt", "reward_model", "audios"}
        missing = required - set(reader.schema_arrow.names)
        if missing:
            raise ValueError("missing Parquet columns: {}".format(sorted(missing)))
        total = reader.metadata.num_rows
        target = min(total, limit) if limit else total
        if not target:
            raise ValueError("validation Parquet has no rows")
        # Exclusive directory creation prevents mixing samples from two exports.
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "samples").mkdir()
        print("[aqa-export] source={} rows={} selected={} output={}".format(source, total, target, destination), flush=True)
        exported = 0
        row_index = 0
        audio_index = None
        current_reference = None
        columns = [name for name in reader.schema_arrow.names
                   if name in required | {"images", "videos", "data_source", "extra_info"}]
        try:
            with (destination / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
                for batch in reader.iter_batches(batch_size=batch_size, columns=columns):
                    for row in batch.to_pylist():
                        row_index = exported
                        audio_index = None
                        current_reference = None
                        sample_id = "row_{:06d}".format(row_index)
                        print("[aqa-export] row={}/{} sample={}".format(row_index + 1, target, sample_id), flush=True)
                        prompt, answer = _validate_row(row)
                        sample_dir = destination / "samples" / sample_id
                        sample_dir.mkdir()
                        audios = []
                        request_payloads = []
                        for audio_index, item in enumerate(row["audios"]):
                            current_reference = None
                            current_reference, archive, key = _audio_reference(item)
                            print("[aqa-export] row={} audio={} reading={}".format(row_index, audio_index, current_reference), flush=True)
                            if not Path(archive).is_file():
                                # Export exactly the referenced archive, without an
                                # inherited AL_AUDIO_ROOT silently selecting another.
                                raise FileNotFoundError("audio file/archive does not exist: {}".format(archive))
                            reference = ExternalAudioReference(
                                conversation_index=0, content_index=audio_index,
                                absolute_path=archive, relative_path=Path(archive).name,
                                archive_key=key, source_path=str(source),
                            )
                            payload, info = _wav_payload(read_external_audio(reference, audio_root=None))
                            wav_path = sample_dir / "audio_{:02d}.wav".format(audio_index)
                            with wav_path.open("xb") as wav:
                                wav.write(payload)
                            audios.append(dict(info, audio_index=audio_index, source_reference=current_reference,
                                               wav_path=wav_path.relative_to(destination).as_posix()))
                            if model is not None:
                                request_payloads.append(payload)
                        record = {
                            "sample_id": sample_id, "row_index": row_index,
                            "source_parquet": str(source), "prompt": prompt,
                            "ground_truth": answer, "audios": audios,
                        }
                        extra = row.get("extra_info")
                        if isinstance(extra, Mapping):
                            record["source_index"] = extra.get("index")
                            record["problem_id"] = extra.get("problem_id")
                        if "data_source" in row:
                            record["data_source"] = row["data_source"]
                        if model is not None:
                            _write_json(sample_dir / "request.json", _request(prompt, request_payloads, model, max_tokens))
                            record["request_path"] = (sample_dir / "request.json").relative_to(destination).as_posix()
                        _write_json(sample_dir / "prompt.json", prompt)
                        with (sample_dir / "ground_truth.txt").open("x", encoding="utf-8") as truth:
                            truth.write(answer)
                        _write_json(sample_dir / "sample.json", record)
                        manifest.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        manifest.flush()
                        exported += 1
                        if exported == target:
                            break
                    if exported == target:
                        break
            if exported != target:
                raise RuntimeError("row count changed while exporting")
        except (Exception, KeyboardInterrupt) as exc:
            error = {
                "row_index": row_index, "audio_index": audio_index,
                "source_reference": current_reference, "source_parquet": str(source),
                "error_type": getattr(exc, "error_type", type(exc).__name__), "message": str(exc),
                "details": getattr(exc, "details", {}),
            }
            _write_json(destination / "error.json", error)
            _write_json(destination / "summary.json", {
                "status": "failed", "exported_rows": exported, "selected_rows": target, "error": error,
            })
            raise RuntimeError("AQA export failed at row={} audio={}; see {}".format(
                row_index, audio_index, destination / "error.json")) from exc
        _write_json(destination / "summary.json", {
            "status": "complete", "source_parquet": str(source), "source_rows": total,
            "selected_rows": target, "exported_rows": exported, "model": model,
        })
    print("[aqa-export] COMPLETE rows={} manifest={}".format(exported, destination / "manifest.jsonl"), flush=True)
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", required=True, help="New directory; existing directories are rejected")
    parser.add_argument("--limit", type=int, default=0, help="Export first N rows; 0 exports all rows")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", help="Also create request.json; must match the server's served model name")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Response limit for optional request.json")
    args = parser.parse_args()
    try:
        export_validation(args.parquet, args.output_dir, args.limit, args.batch_size, args.model, args.max_tokens)
    except (OSError, ValueError, RuntimeError) as exc:
        print("[aqa-export] ERROR: {}".format(exc), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
