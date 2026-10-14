"""Opt-in observations of rollout and IPC boundaries, with no tensor reads.

Importing this module does not import an inference framework. ``install`` is
best-effort and repeatable: a partially initialized/incompatible dependency is
reported as unavailable and can be retried after it finishes importing. A
successful load/sync event proves a call returned, not numerical correctness.
"""

import functools
import importlib
import inspect
import time
from collections.abc import Mapping
from contextvars import ContextVar
from itertools import count

from . import diagnostic_io as diag

_request = ContextVar("ext_rollout_diag_request", default=None)
_MARKER = "_ext_rollout_trace_wrapped"
_calls = count(1)


def _record(event, **fields):
    try:
        diag.emit(event, **fields)
    except Exception:
        pass


def _observe(function, *args):
    try:
        return function(*args)
    except Exception as exc:
        return {"observation_error": type(exc).__name__}


def _active(event):
    try:
        return diag.enabled() and diag.limited(event)
    except Exception:
        return False


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _attrs(value, names):
    result = {}
    for name in names:
        item = _get(value, name)
        if item is not None:
            # torch.dtype is metadata; do not stringify tensors or arbitrary models.
            result[name] = str(item) if type(item).__name__ in ("dtype", "device") else diag.summarize(item)
    return result


def _identity(instance):
    model_config = _get(instance, "model_config")
    if model_config is None:
        model_config = _get(_get(instance, "vllm_config"), "model_config")
    config = _get(instance, "config")
    hf_config = _get(model_config, "hf_config")
    text_config = _get(hf_config, "text_config")
    config_keys = ("architectures", "model_type", "dtype", "torch_dtype", "hidden_size", "num_hidden_layers",
                   "num_attention_heads", "num_key_value_heads", "moe_intermediate_size", "num_experts",
                   "vocab_size", "max_position_embeddings")
    return {
        "instance_type": type(instance).__name__,
        "model": _attrs(model_config, ("model", "path", "local_path", "model_stage", "architecture", "dtype")),
        "hf_config": _attrs(hf_config, config_keys),
        "text_config": _attrs(text_config, config_keys),
        "role_clues": _attrs(instance, ("is_reward_model", "is_teacher_model", "replica_rank", "node_rank",
                                       "local_rank", "rank", "device", "global_steps", "global_step", "_ar_mode")),
        "rollout_config": _attrs(config, ("max_model_len", "response_length", "prompt_length", "name",
                                           "tensor_model_parallel_size", "engine_kwargs")),
    }


def _sampling(value):
    # Do not pop, normalize, copy device values, or construct new SamplingParams.
    if isinstance(value, Mapping):
        return diag.summarize(value)
    if isinstance(value, (tuple, list)):
        return [_sampling(item) for item in value[:4]]
    return _attrs(value, ("max_tokens", "min_tokens", "temperature", "top_p", "top_k", "seed", "n",
                          "repetition_penalty", "presence_penalty", "frequency_penalty", "ignore_eos",
                          "stop", "stop_token_ids", "include_stop_str_in_output", "logprobs"))


def _media(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return diag.summarize(value)
    items = []
    for item in value[:4]:
        if type(item).__module__.startswith("PIL."):
            items.append(_attrs(item, ("size", "mode", "format", "filename")))
        else:
            items.append(diag.summarize(item))
    return {"count": len(value), "items": items, "truncated": len(value) > 4,
            "cache_attrs": _attrs(value, ("_precomputed_vision_embeds", "_precomputed_audio_embeds"))}


def _decoded_prompt(instance, ids):
    # Only decode existing CPU lists. No tensor -> host transfer for tracing.
    if not isinstance(ids, (list, tuple)) or len(ids) > 8192 or not all(isinstance(v, int) for v in ids):
        return {"decoded": False, "reason": "not_cpu_token_list_or_over_8192_tokens"}
    tokenizer = _get(_get(instance, "model_config"), "tokenizer")
    if tokenizer is None or not callable(getattr(tokenizer, "decode", None)):
        return {"decoded": False, "reason": "tokenizer_unavailable"}
    return diag.text_info(tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False))


def _inputs(kind, bound):
    obj = bound.get("self")
    fields = _identity(obj)
    if kind == "request":
        ids = bound.get("prompt_ids")
        fields.update(request_id=bound.get("request_id"), prompt_tokens=diag.token_info(ids),
                      prompt_text=_observe(_decoded_prompt, obj, ids),
                      sampling=_sampling(bound.get("sampling_params")),
                      image_data=_media(bound.get("image_data")), audio_data=_media(bound.get("audio_data")),
                      video_data=_media(bound.get("video_data")),
                      mm_processor_kwargs=diag.summarize(bound.get("mm_processor_kwargs")))
    elif kind == "engine_request":
        prompt = bound.get("prompt")
        media = _get(prompt, "multi_modal_data", {}) or {}
        fields.update(request_id=bound.get("request_id"),
                      prompt_tokens=diag.token_info(_get(prompt, "prompt_token_ids")),
                      sampling=_sampling(bound.get("params")),
                      image_data=_media(_get(media, "image")), audio_data=_media(_get(media, "audio")),
                      video_data=_media(_get(media, "video")),
                      mm_processor_kwargs=diag.summarize(_get(prompt, "mm_processor_kwargs")))
    elif kind == "ipc":
        fields.update(arguments={key: diag.summarize(value) for key, value in bound.items() if key != "self"},
                      note="metadata only; no tensor values or checksums read")
    elif kind == "server_init":
        fields["cli"] = _attrs(bound.get("args"), ("model", "model_stage", "deploy_config", "dtype",
                                                    "max_model_len", "tensor_parallel_size", "enforce_eager"))
    elif kind == "model_load":
        model_config = _get(bound.get("vllm_config"), "model_config") or bound.get("model_config")
        fields["load_config"] = _attrs(model_config, ("model", "model_stage", "dtype", "load_format"))
    return fields


def _outputs(kind, value):
    if kind not in ("request", "engine_request"):
        return {"returned_type": type(value).__name__, "numerical_equivalence_checked": False}
    if kind == "request":
        return {"response_tokens": diag.token_info(_get(value, "token_ids")),
                "stop_reason": _get(value, "stop_reason"),
                "extra_fields": diag.summarize(_get(value, "extra_fields"))}
    request_output = _get(value, "request_output")
    completions = _get(request_output, "outputs")
    if not isinstance(completions, (tuple, list)):
        return {"raw_output_available": False, "returned_type": type(value).__name__}
    outputs = []
    for completion in completions[:4]:
        outputs.append({"response_tokens": diag.token_info(_get(completion, "token_ids")),
                        "text": diag.text_info(_get(completion, "text")),
                        "finish_reason": _get(completion, "finish_reason"),
                        "stop_reason": _get(completion, "stop_reason")})
    return {"raw_output_available": True, "request_id": _get(request_output, "request_id"),
            "completions": outputs, "completion_count": len(completions)}


def _wrap(original, kind, target):
    signature = inspect.signature(original)

    def start(args, kwargs):
        context = None
        bound = {}
        call_id = None
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
            if kind == "request":
                context = _request.set(bound.get("request_id"))
            sampled = _active(target)
            if sampled:
                call_id = next(_calls)
                fields = _observe(_inputs, kind, bound)
                fields.setdefault("request_id", _request.get())
                _record(f"{kind}_begin", target=target, call_id=call_id, **fields)
            return sampled, context, time.monotonic(), _request.get(), call_id
        except Exception:
            return False, context, 0.0, None, call_id

    def end(state, result=None, error=None):
        sampled, context, started, request_id, call_id = state
        try:
            if sampled:
                fields = (_observe(_outputs, kind, result) if error is None else
                          {"error_type": type(error).__name__, "error": diag.text_info(str(error))})
                fields.setdefault("request_id", request_id)
                _record(f"{kind}_{'error' if error is not None else 'end'}", target=target, call_id=call_id,
                        elapsed_seconds=time.monotonic() - started, **fields)
        except Exception:
            pass
        finally:
            if context is not None:
                try:
                    _request.reset(context)
                except Exception:
                    pass

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            state = start(args, kwargs)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                end(state, error=exc)
                raise
            end(state, result=result)
            return result
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            state = start(args, kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                end(state, error=exc)
                raise
            end(state, result=result)
            return result
    setattr(wrapped, _MARKER, True)
    return wrapped


def _install_method(module_name, class_name, method_name, required, kind, asynchronous):
    target = f"{module_name}.{class_name}.{method_name}"
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        descriptor = inspect.getattr_static(cls, method_name)
        if isinstance(descriptor, (staticmethod, classmethod)):
            raise TypeError("expected an instance method, found staticmethod/classmethod")
        original = getattr(cls, method_name)
        if getattr(original, _MARKER, False):
            return True
        if inspect.isasyncgenfunction(original) or inspect.isgeneratorfunction(original):
            raise TypeError("generator contract is unsupported; method left unchanged")
        if inspect.iscoroutinefunction(original) != asynchronous:
            raise TypeError("sync/async contract differs from the supported boundary")
        signature = inspect.signature(original)
        missing = set(required) - set(signature.parameters)
        if missing:
            raise TypeError(f"required signature parameters missing: {sorted(missing)}; signature={signature}")
        setattr(cls, method_name, _wrap(original, kind, target))
        _record("hook_installed", target=target, signature=str(signature), module=diag.module_info(module))
        return True
    except Exception as exc:
        _record("hook_unavailable", target=target, error_type=type(exc).__name__, reason=str(exc),
                note="no observation at this boundary; absence of events is not evidence of success")
        return False


def install():
    """Install only under the opt-in switch, in each process loading the plugin."""
    if not diag.enabled():
        return {}
    server = "verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server"
    extension = "verl_omni.workers.rollout.vllm_rollout.utils"
    specs = (
        (server, "vLLMOmniHttpServer", "generate", ("self", "prompt_ids", "sampling_params", "request_id"),
         "request", True),
        (server, "vLLMOmniHttpServer", "_run_generation", ("self", "prompt", "params", "request_id"),
         "engine_request", True),
        (server, "vLLMOmniHttpServer", "run_server", ("self", "args"), "server_init", True),
        (extension, "vLLMOmniColocateWorkerExtension", "update_weights_from_ipc", ("self",), "ipc", False),
        ("vllm.model_executor.model_loader.base_loader", "BaseModelLoader", "load_model", ("self",),
         "model_load", False),
    )
    return {f"{module}.{cls}.{method}": _install_method(module, cls, method, required, kind, asynchronous)
            for module, cls, method, required, kind, asynchronous in specs}
