"""Opt-in observations of rollout and IPC boundaries, with no tensor reads.

Importing this module does not import an inference framework. ``install`` is
best-effort and repeatable: a partially initialized/incompatible dependency is
reported as unavailable and can be retried after it finishes importing. A
successful load/sync event proves a call returned, not numerical correctness.
"""

import functools
import importlib
import inspect
import sys
import time
from collections.abc import Mapping
from contextvars import ContextVar
from itertools import count

from . import diagnostic_io as diag

_request = ContextVar("ext_rollout_diag_request", default=None)
_MARKER = "_ext_rollout_trace_wrapped"
_calls = count(1)
PROBE_REVISION = "2026-10-15-engine-stream-v3"
_engine_attempts = set()


def _request_context():
    """Resolve worker-local state at call time, never capture it in Ray closures.

    Ray/cloudpickle serializes nested method wrappers by value. Directly using
    the module's ContextVar in start/end would include that unpicklable object
    in their globals, even when exporting an unrelated method such as sleep.
    This accessor only references an importable module and its name.
    """
    return importlib.import_module(__name__)._request


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
    if model_config is None:
        model_config = _get(_get(instance, "model_runner"), "model_config")
    config = _get(instance, "config")
    hf_config = _get(model_config, "hf_config")
    text_config = _get(hf_config, "text_config")
    config_keys = ("architectures", "model_type", "dtype", "torch_dtype", "hidden_size", "num_hidden_layers",
                   "num_attention_heads", "num_key_value_heads", "moe_intermediate_size", "num_experts",
                   "vocab_size", "max_position_embeddings")
    return {
        "instance_type": f"{type(instance).__module__}.{type(instance).__qualname__}",
        "model": _attrs(model_config, ("model", "path", "local_path", "model_stage", "architecture", "dtype")),
        "hf_config": _attrs(hf_config, config_keys),
        "text_config": _attrs(text_config, config_keys),
        "role_clues": _attrs(instance, ("is_reward_model", "is_teacher_model", "replica_rank", "node_rank",
                                       "local_rank", "rank", "device", "global_steps", "global_step", "_ar_mode")),
        "rollout_config": _attrs(config, ("max_model_len", "response_length", "prompt_length", "name",
                                           "tensor_model_parallel_size", "engine_kwargs")),
    }


def _gdn_status():
    """Read existing registrations only; never install/reinstall a compute patch."""
    torch = sys.modules.get("torch")
    helper = sys.modules.get("verl_omni_ext.probes.gdn_trace")
    result = {"helper_revision": getattr(helper, "PROBE_REVISION", None),
              "torch_loaded": torch is not None, "registration_only": True, "execution_checked": False,
              "note": "registration snapshot only; require gdn call/numeric events for execution evidence"}
    if torch is None:
        return result
    # torch.ops lazy attributes can register wrappers. Inspect the existing namespace
    # dictionary instead of manufacturing a namespace/operator while observing it.
    ops = vars(torch.ops).get("_C_ascend")
    packet = vars(ops).get("npu_recurrent_gated_delta_rule") if ops is not None else None
    result["operator_cached"] = packet is not None
    for key, value in (("packet", packet), ("default_cached_attribute", vars(packet).get("default") if hasattr(packet, "__dict__") else None)):
        result[key] = {"available": value is not None,
                       "bf16_patch": bool(getattr(value, "_is_bf16_patched", False)),
                       "probe_revision": getattr(value, "_gdn_probe_revision", None)}
    npu = vars(torch).get("npu")
    result["capture_api_available"] = callable(getattr(npu, "is_current_stream_capturing", None))
    return result


def _stream_token_info(value, full):
    if full or not isinstance(value, (list, tuple)):
        return diag.token_info(value)
    return {"count": len(value), "head": diag.summarize(value[:32]), "tail": diag.summarize(value[-32:]),
            "truncated": len(value) > 64, "sha256_computed": False}


def _stream_text_info(value, full):
    if full or not isinstance(value, str):
        return diag.text_info(value)
    limit = diag._limit("MINICPMO_ROLLOUT_DIAG_TEXT_CHARS", 256, 32768)
    info = {"chars": len(value), "truncated": len(value) > limit, "sha256_computed": False}
    if len(value) <= limit:
        info["text"] = value
    else:
        half = limit // 2
        info.update(head=value[:half], tail=value[-half:] if half else "")
    return info


def _model_identity(model):
    cls = type(model)
    return {"class": f"{cls.__module__}.{cls.__qualname__}",
            "class_source": _observe(diag.module_info, sys.modules.get(cls.__module__)),
            "config": _attrs(_get(model, "config"), ("_name_or_path", "model_type", "model_stage", "dtype")),
            "model_stage": _get(model, "model_stage")}


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
        sampling_name = next((name for name in ("sampling_params_list", "sampling_params", "params")
                              if name in bound), None)
        fields.update(request_id=bound.get("request_id"),
                      prompt_tokens=diag.token_info(_get(prompt, "prompt_token_ids")),
                      prompt_text=diag.text_info(prompt if isinstance(prompt, str) else _get(prompt, "prompt")),
                      sampling_argument=sampling_name, sampling=_sampling(bound.get(sampling_name)),
                      image_data=_media(_get(media, "image")), audio_data=_media(_get(media, "audio")),
                      video_data=_media(_get(media, "video")),
                      mm_processor_kwargs=diag.summarize(_get(prompt, "mm_processor_kwargs")))
    elif kind == "ipc":
        fields.update(arguments={key: diag.summarize(value) for key, value in bound.items() if key != "self"},
                      note="metadata only; no tensor values or checksums read",
                      worker_model=_observe(_model_identity, _get(_get(obj, "model_runner"), "model")),
                      gdn_status=_observe(_gdn_status))
    elif kind in ("wake", "sleep", "global_step"):
        fields["arguments"] = {key: diag.summarize(value) for key, value in bound.items() if key != "self"}
    elif kind == "server_init":
        fields["cli"] = _attrs(bound.get("args"), ("model", "model_stage", "deploy_config", "dtype",
                                                    "max_model_len", "tensor_parallel_size", "enforce_eager"))
    elif kind == "model_load":
        model_config = _get(bound.get("vllm_config"), "model_config") or bound.get("model_config")
        fields["load_config"] = _attrs(model_config, ("model", "model_stage", "dtype", "load_format"))
    return fields


def _outputs(kind, value, full=True):
    if kind not in ("request", "engine_request"):
        fields = {"returned_type": type(value).__name__, "numerical_equivalence_checked": False}
        if kind in ("model_load", "ipc"):
            fields["gdn_status"] = _observe(_gdn_status)
        if kind == "model_load":
            fields["loaded_model"] = _observe(_model_identity, value)
        return fields
    if kind == "request":
        return {"response_tokens": diag.token_info(_get(value, "token_ids")),
                "stop_reason": _get(value, "stop_reason"),
                "extra_fields": diag.summarize(_get(value, "extra_fields"))}
    request_output = _get(value, "request_output")
    if request_output is None:
        request_output = value
    completions = _get(request_output, "outputs")
    if not isinstance(completions, (tuple, list)):
        return {"raw_output_available": False, "returned_type": type(value).__name__}
    outputs = []
    for completion in completions[:4]:
        outputs.append({"response_tokens": _stream_token_info(_get(completion, "token_ids"), full),
                        "text": _stream_text_info(_get(completion, "text"), full),
                        "finish_reason": _get(completion, "finish_reason"),
                        "stop_reason": _get(completion, "stop_reason")})
    return {"raw_output_available": True, "request_id": _get(request_output, "request_id"),
            "finished": _get(request_output, "finished"),
            "prompt_tokens": _stream_token_info(_get(request_output, "prompt_token_ids"), full),
            "completions": outputs, "completion_count": len(completions)}


class _ObservedStream:
    """One pull in, one pull out; do not replay requests or retain device outputs."""

    def __init__(self, stream, state, end):
        self._stream, self._iterator = stream, None
        self._state, self._end = state, end
        self._count, self._last, self._closed = 0, {}, False

    def __aiter__(self):
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        return self

    def _finish(self, reason, error=None):
        if not self._closed:
            self._closed = True
            self._end(self._state, error=error, summary={**self._last, "stream_end": reason,
                                                       "yield_count": self._count})

    async def _pull(self, method_name, *args):
        try:
            if self._iterator is None:
                self.__aiter__()
            value = await getattr(self._iterator, method_name)(*args)
        except StopAsyncIteration:
            self._finish("exhausted")
            raise
        except BaseException as exc:
            self._finish("error", error=exc)
            raise
        self._count += 1
        if self._state[0]:
            # Full-list hashing on every cumulative token chunk is quadratic.
            # Keep exact bounded metadata for the latest chunk, with hashes only
            # at the first/final chunk. Never retain a raw engine output/tensor.
            try:
                raw = _get(value, "request_output")
                raw = value if raw is None else raw
                full = self._count == 1 or _get(raw, "finished") is True
                self._last = _observe(_outputs, "engine_request", value, full)
                if self._count == 1:
                    _record("engine_request_first_output", call_id=self._state[4],
                            **{**self._last, "request_id": self._last.get("request_id") or self._state[3]})
            except Exception as exc:
                self._last = {"observation_error": type(exc).__name__}
        return value

    async def __anext__(self):
        return await self._pull("__anext__")

    async def asend(self, value):
        return await self._pull("asend", value)

    async def athrow(self, *args):
        return await self._pull("athrow", *args)

    async def aclose(self):
        try:
            close = getattr(self._iterator if self._iterator is not None else self._stream, "aclose", None)
            if close is not None:
                await close()
        except BaseException as exc:
            self._finish("close_error", error=exc)
            raise
        self._finish("consumer_closed")


def _wrap(original, kind, target):
    signature = inspect.signature(original)

    def start(args, kwargs):
        context = None
        bound = {}
        call_id = None
        try:
            bound = signature.bind_partial(*args, **kwargs).arguments
            if kind == "request":
                context = _request_context().set(bound.get("request_id"))
                _observe(_discover_engine, bound.get("self"))
            sampled = _active(target)
            if sampled:
                call_id = next(_calls)
                fields = _observe(_inputs, kind, bound)
                fields.setdefault("request_id", _request_context().get())
                _record(f"{kind}_begin", target=target, call_id=call_id, **fields)
            return sampled, context, time.monotonic(), bound.get("request_id", _request_context().get()), call_id
        except Exception:
            return False, context, 0.0, None, call_id

    def end(state, result=None, error=None, summary=None):
        sampled, context, started, request_id, call_id = state
        try:
            if sampled:
                fields = (_observe(_outputs, kind, result) if error is None else
                          {"error_type": type(error).__name__, "error": diag.text_info(str(error))})
                if summary is not None:
                    fields.update(summary)
                if fields.get("request_id") is None:
                    fields["request_id"] = request_id
                _record(f"{kind}_{'error' if error is not None else 'end'}", target=target, call_id=call_id,
                        elapsed_seconds=time.monotonic() - started, **fields)
        except Exception:
            pass
        finally:
            if context is not None:
                try:
                    _request_context().reset(context)
                except Exception:
                    pass

    def returned(state, result, args, kwargs):
        if kind == "server_init":
            obj = args[0] if args else kwargs.get("self")
            _observe(_discover_engine, obj)
        if kind == "engine_request" and hasattr(result, "__aiter__"):
            if not state[0]:
                return result
            return _ObservedStream(result, state, end)
        end(state, result=result)
        return result

    if inspect.isasyncgenfunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            state = start(args, kwargs)
            stream = _ObservedStream(original(*args, **kwargs), state, end)
            # Delegate the whole async-generator protocol, including asend/athrow
            # and explicit close. A plain async-for wrapper silently loses these.
            try:
                item = await stream.__anext__()
                while True:
                    try:
                        sent = yield item
                    except GeneratorExit:
                        await stream.aclose()
                        raise
                    except BaseException as exc:
                        item = await stream.athrow(exc)
                    else:
                        item = await stream.__anext__() if sent is None else await stream.asend(sent)
            except StopAsyncIteration:
                return
    elif inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def wrapped(*args, **kwargs):
            state = start(args, kwargs)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                end(state, error=exc)
                raise
            return returned(state, result, args, kwargs)
    else:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            state = start(args, kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                end(state, error=exc)
                raise
            return returned(state, result, args, kwargs)
    setattr(wrapped, _MARKER, True)
    wrapped._rollout_probe_revision = PROBE_REVISION
    return wrapped


def _install_engine_class(cls):
    target = f"{cls.__module__}.{cls.__qualname__}.generate"
    try:
        descriptor = inspect.getattr_static(cls, "generate")
        if isinstance(descriptor, (staticmethod, classmethod)):
            raise TypeError("engine generate is not an instance method")
        original = getattr(cls, "generate")
        if getattr(original, _MARKER, False):
            return True
        signature = inspect.signature(original)
        keys = set(signature.parameters)
        if not {"self", "prompt", "request_id"}.issubset(keys):
            raise TypeError(f"engine prompt/request_id contract unknown: {signature}")
        if not keys.intersection(("sampling_params_list", "sampling_params", "params")):
            raise TypeError(f"engine sampling contract unknown: {signature}")
        if inspect.isgeneratorfunction(original):
            raise TypeError("synchronous generator is not a supported async engine")
        setattr(cls, "generate", _wrap(original, "engine_request", target))
        contract = ("async_generator" if inspect.isasyncgenfunction(original) else
                    "coroutine" if inspect.iscoroutinefunction(original) else "function")
        _record("hook_installed", target=target, signature=str(signature), contract=contract,
                probe_revision=PROBE_REVISION, execution_observed=False,
                module=_observe(diag.module_info, sys.modules.get(cls.__module__)))
        return True
    except Exception as exc:
        # Unknown engine versions keep their original method. Avoid one warning per request.
        if target not in _engine_attempts:
            _engine_attempts.add(target)
            _record("hook_unavailable", target=target, error_type=type(exc).__name__, reason=str(exc),
                    execution_observed=False)
        return False


def _discover_engine(server):
    engine = _get(server, "engine")
    if engine is None:
        return
    class_hooked = _install_engine_class(type(engine))
    actual = _get(engine, "generate")
    hooked = bool(getattr(actual, _MARKER, False)) and getattr(actual, "_rollout_probe_revision", None) == PROBE_REVISION
    if _active(f"engine_boundary:{type(engine).__module__}.{type(engine).__qualname__}"):
        _record("engine_boundary", engine_class=f"{type(engine).__module__}.{type(engine).__qualname__}",
                installed=hooked, class_hook_installed=class_hooked,
                engine_probe_revision=getattr(actual, "_rollout_probe_revision", None),
                execution_observed=False, server=_observe(_identity, server))


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
        _record("hook_installed", target=target, signature=str(signature), module=diag.module_info(module),
                probe_revision=PROBE_REVISION, execution_observed=False)
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
        (server, "vLLMOmniHttpServer", "run_server", ("self", "args"), "server_init", True),
        (server, "vLLMOmniHttpServer", "wake_up", ("self",), "wake", True),
        (server, "vLLMOmniHttpServer", "_sleep_hybrid", ("self",), "sleep", True),
        (server, "vLLMOmniHttpServer", "set_global_steps", ("self",), "global_step", True),
        (extension, "vLLMOmniColocateWorkerExtension", "update_weights_from_ipc", ("self",), "ipc", False),
        ("vllm.model_executor.model_loader.base_loader", "BaseModelLoader", "load_model", ("self",),
         "model_load", False),
    )
    results = {f"{module}.{cls}.{method}": _install_method(module, cls, method, required, kind, asynchronous)
               for module, cls, method, required, kind, asynchronous in specs}
    target = "vllm_omni.entrypoints.async_omni.AsyncOmni"
    try:
        cls = getattr(importlib.import_module("vllm_omni.entrypoints.async_omni"), "AsyncOmni")
        results[target] = _install_engine_class(cls)
    except Exception as exc:
        _record("hook_unavailable", target=target, reason=str(exc),
                note="will retry against actual server.engine at runtime")
        results[target] = False
    return results
