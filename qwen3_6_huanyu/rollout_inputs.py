"""Contract Huanyu rollout media using the original request's media metadata.

Training keeps expanded tokens. vLLM's MiniCPM-o prompt replacement consumes one
literal per original image/audio item; one audio item can contain several 30s
chunks. Token adjacency alone cannot recover that item boundary. This hook is
scoped to Huanyu processors and leaves other models' server calls unchanged.
"""

from collections.abc import Mapping
from functools import wraps
import inspect
import logging
from numbers import Integral


logger = logging.getLogger(__name__)
_IMAGE_PATTERN = "(<image>./</image>)"
_AUDIO_PATTERN = "(<audio>./</audio>)"
_HOOK_FLAG = "_huanyu_metadata_contraction"
_MISSING = object()


def _field(value, key):
    if isinstance(value, Mapping):
        return value.get(key, _MISSING)
    return getattr(value, key, _MISSING)


def _sampling_overrides(server):
    """Bridge controls omitted by Omni's AR SamplingParams construction.

    Use the server's serialized rollout config, not driver environment variables:
    an existing Ray cluster does not inherit later shell exports. Defaults remain
    untouched, and the Huanyu processor guard keeps reward servers out of scope.
    """
    config = _field(server, "config")
    overrides = {}
    ignore_eos = _field(config, "ignore_eos")
    if ignore_eos is not _MISSING:
        if not isinstance(ignore_eos, bool):
            raise ValueError("Huanyu rollout.ignore_eos must be a boolean")
        if ignore_eos:
            overrides["ignore_eos"] = True
    side = _field(_field(config, "engine_kwargs"), "q36_sampling")
    min_tokens = _field(side, "min_tokens")
    if min_tokens is not _MISSING:
        if isinstance(min_tokens, bool) or not isinstance(min_tokens, Integral) or min_tokens < 0:
            raise ValueError("Huanyu engine_kwargs.q36_sampling.min_tokens must be a nonnegative integer")
        overrides["min_tokens"] = int(min_tokens)
    return overrides


def _require_autoregressive_mode(server):
    # _ar_mode is a private implementation detail, absent in some servers.
    # An explicit AR rollout config can establish the mode without this flag;
    # it must never override an existing False flag or a conflicting config.
    flag = getattr(server, "_ar_mode", _MISSING)
    config = _field(server, "config")
    engine_kwargs = _field(config, "engine_kwargs")
    omni_kwargs = _field(engine_kwargs, "vllm_omni")
    mode = _field(omni_kwargs, "output_mode")
    flag_text = "<missing>" if flag is _MISSING else repr(flag)
    mode_text = "<missing>" if mode is _MISSING else repr(mode)
    details = (
        f"server={type(server).__module__}.{type(server).__name__} "
        f"_ar_mode={flag_text} "
        f"config.engine_kwargs.vllm_omni.output_mode={mode_text}"
    )
    reason = None
    if flag is not _MISSING and flag is not True and flag is not False:
        reason = "the server's private mode flag is not a boolean"
    elif flag is False:
        reason = "the server explicitly reports non-AR mode; config cannot override it"
    elif mode is not _MISSING and mode is not None and mode != "ar":
        reason = "the rollout config explicitly selects a non-AR or unknown output mode"
    elif flag is not True and mode != "ar":
        reason = "neither the private mode flag nor the rollout config confirms AR mode"
    if reason:
        raise RuntimeError(
            f"Huanyu rollout requires the autoregressive server mode: {reason}; {details}"
        )
    if not getattr(server, "_huanyu_mode_reported", False):
        logger.info("Huanyu rollout mode verified: %s", details)
        server._huanyu_mode_reported = True


def _ids(value, name):
    if hasattr(value, "tolist"):
        value = value.tolist()
    result = list(value)
    if any(isinstance(item, bool) or not isinstance(item, Integral) or item < 0 for item in result):
        raise TypeError(f"Huanyu {name} must be a flat sequence of non-negative token IDs")
    return [int(item) for item in result]


def _media_items(data, name):
    value = data.get(name)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Huanyu rollout multi_modal_data[{name!r}] must be a list of media items")
    return list(value)


def _audio_chunk_lengths(item, processor):
    sr = int(processor.sampling_rate)
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], Integral):
        item, actual_sr = item
        if int(actual_sr) != sr:
            raise ValueError(f"Huanyu rollout audio sampling rate {actual_sr} != processor rate {sr}")
    if getattr(item, "ndim", None) != 1:
        raise TypeError("Huanyu rollout audio must be a decoded one-dimensional waveform")
    num_samples = len(item)
    if num_samples <= 0:
        raise ValueError("Huanyu rollout audio waveform is empty")
    n_fft, hop = int(processor.n_fft), int(processor.hop_length)
    chunk_samples = int(processor.audio_chunk_seconds) * sr
    encoder_down = int(processor.audio_encoder_downsample)
    projector_down = int(processor.audio_projector_downsample)
    if min(sr, n_fft, hop, chunk_samples, encoder_down, projector_down) <= 0:
        raise ValueError("Huanyu audio frontend rates and downsampling factors must be positive")
    lengths = []
    for offset in range(0, num_samples, chunk_samples):
        size = min(chunk_samples, num_samples - offset)
        frames = (size - n_fft) // hop + 1
        tokens = ((frames + encoder_down - 1) // encoder_down) // projector_down
        if frames <= 0 or tokens <= 0:
            raise ValueError("Huanyu rollout audio contains a chunk too short to produce a token")
        lengths.append(tokens)
    return lengths


def contract_rollout_prompt(processor, prompt_ids, multi_modal_data):
    """Return contracted IDs without modifying the expanded training IDs.

    Audio chunk counts/lengths come from each original waveform. Image spans
    must each contain only Qwen image-pad tokens. Missing, extra, malformed or
    wrongly sized spans raise before generation; no span merging/truncation or
    fallback to the old Whisper prompt scanner is performed.
    """
    if not isinstance(multi_modal_data, Mapping):
        raise TypeError("Huanyu rollout multi_modal_data must be a mapping")
    if _media_items(multi_modal_data, "video"):
        raise ValueError("Huanyu rollout expects dataset video items converted into image frames")
    images = _media_items(multi_modal_data, "image")
    audios = _media_items(multi_modal_data, "audio")
    groups = [_audio_chunk_lengths(audio, processor) for audio in audios]
    ids = _ids(prompt_ids, "prompt_ids")
    marker_names = ("vision_start_token_id", "vision_end_token_id", "image_token_id",
                    "audio_start_token_id", "audio_end_token_id", "audio_placeholder_token_id")
    vs, ve, image_pad, aus, aue, audio_pad = _ids(
        [getattr(processor, name) for name in marker_names], "media markers"
    )
    if len({vs, ve, image_pad, aus, aue, audio_pad}) != 6:
        raise ValueError("Huanyu media markers must be distinct")
    image_pattern = _ids(processor.tokenizer.encode(_IMAGE_PATTERN, add_special_tokens=False), "image pattern")
    audio_pattern = _ids(processor.tokenizer.encode(_AUDIO_PATTERN, add_special_tokens=False), "audio pattern")
    if not image_pattern or not audio_pattern:
        raise ValueError("Huanyu rollout media patterns must tokenize to non-empty sequences")

    output = []
    image_index = audio_index = index = 0
    while index < len(ids):
        token = ids[index]
        if token == vs:
            if image_index >= len(images):
                raise ValueError("Huanyu rollout has more image spans than image items")
            end = index + 1
            while end < len(ids) and ids[end] == image_pad:
                end += 1
            if end == index + 1 or end >= len(ids) or ids[end] != ve:
                raise ValueError(f"Huanyu rollout has a malformed image span at token {index}")
            output.extend(image_pattern)
            image_index += 1
            index = end + 1
        elif token == aus:
            if audio_index >= len(groups):
                raise ValueError("Huanyu rollout has more audio spans than original audio items")
            for chunk_index, length in enumerate(groups[audio_index]):
                end = index + length + 1
                if (index >= len(ids) or ids[index] != aus or end >= len(ids)
                        or ids[end] != aue or any(value != audio_pad for value in ids[index + 1:end])):
                    raise ValueError(
                        f"Huanyu rollout audio item {audio_index} chunk {chunk_index} does not match "
                        f"{length} placeholder tokens at position {index}"
                    )
                index = end + 1
            output.extend(audio_pattern)
            audio_index += 1
        elif token in (ve, aue, image_pad):
            raise ValueError(f"Huanyu rollout has an unmatched media marker at token {index}")
        else:
            if (ids[index:index + len(image_pattern)] == image_pattern
                    or ids[index:index + len(audio_pattern)] == audio_pattern):
                raise ValueError(
                    f"Huanyu rollout contains an unexpanded media literal at token {index}; "
                    "the processor must expand every media item before generation"
                )
            output.append(token)
            index += 1
    if image_index != len(images) or audio_index != len(audios):
        raise ValueError(
            f"Huanyu rollout media/span counts differ: images={image_index}/{len(images)}, "
            f"audios={audio_index}/{len(audios)}"
        )
    return output


def _install_on_server_class(server_class):
    # Some versions export a Ray ActorClass rather than the Python class.
    # Methods must be patched on the class Ray actually instantiates.
    modified = getattr(getattr(server_class, "__ray_metadata__", None), "modified_class", None)
    target = modified if modified is not None else server_class
    if not inspect.isclass(target):
        raise RuntimeError("Huanyu rollout hook requires a Python server class or Ray modified_class")

    method_name = "_preprocess_input"
    original = getattr(target, method_name, None)
    if not callable(original):
        # Some servers keep preprocessing inline in their async generate().
        method_name = "generate"
        original = getattr(target, method_name, None)
    if not callable(original):
        raise RuntimeError(
            f"Unsupported Huanyu rollout server {target.__module__}.{target.__name__}: "
            "neither _preprocess_input nor generate is available"
        )
    if getattr(original, _HOOK_FLAG, False):
        return
    signature = inspect.signature(original)
    required = {"self", "prompt_ids"}
    if method_name == "_preprocess_input":
        required.add("multi_modal_data")
        async_method = False
        supported_kind = not inspect.iscoroutinefunction(original) and not inspect.isasyncgenfunction(original)
    else:
        required.update(("image_data", "audio_data"))
        async_method = True
        supported_kind = inspect.iscoroutinefunction(original) or inspect.iscoroutinefunction(inspect.unwrap(original))
    if not required.issubset(signature.parameters) or not supported_kind:
        raise RuntimeError(f"Unsupported vLLM-Omni server {method_name} signature/kind: {signature}")

    def contract_arguments(self, args, kwargs):
        processor = getattr(getattr(self, "model_config", None), "processor", None)
        if not getattr(processor, "_huanyu_training_processor", False):
            return None
        _require_autoregressive_mode(self)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        sampling_overrides = _sampling_overrides(self)
        if sampling_overrides:
            params = bound.arguments.get("sampling_params")
            if not isinstance(params, Mapping):
                raise RuntimeError("Huanyu EOS/length controls require a sampling_params mapping on the server call")
            # Do not mutate a request dictionary shared by sibling generations.
            params = dict(params)
            for key, value in sampling_overrides.items():
                params.setdefault(key, value)
            bound.arguments["sampling_params"] = params
            if not getattr(self, "_huanyu_sampling_reported", False):
                print(f"[huanyu-sampling] ignore_eos={params.get('ignore_eos', False)} "
                      f"min_tokens={params.get('min_tokens', 0)} "
                      f"response_length={_field(_field(self, 'config'), 'response_length')} "
                      "source=rollout_config/q36_sampling scope=huanyu", flush=True)
                self._huanyu_sampling_reported = True
        if method_name == "_preprocess_input":
            media = bound.arguments["multi_modal_data"]
        else:
            media = {
                kind: bound.arguments[key]
                for kind, key in (("image", "image_data"), ("audio", "audio_data"), ("video", "video_data"))
                if bound.arguments.get(key) is not None
            }
        original_ids = bound.arguments["prompt_ids"]
        contracted = contract_rollout_prompt(processor, original_ids, media)
        bound.arguments["prompt_ids"] = contracted
        if not getattr(self, "_huanyu_contraction_reported", False):
            logger.info("Huanyu metadata-aware rollout prompt contraction: %d -> %d tokens", len(original_ids), len(contracted))
            self._huanyu_contraction_reported = True
        return bound

    if async_method:
        # Keep our local qualname/module: Ray must serialize the wrapper, not
        # resolve the unpatched server method by its original module path.
        @wraps(original, assigned=("__name__", "__doc__", "__annotations__"))
        async def preprocess(self, *args, **kwargs):
            bound = contract_arguments(self, args, kwargs)
            if bound is None:
                return await original(self, *args, **kwargs)
            return await original(*bound.args, **bound.kwargs)
    else:
        @wraps(original, assigned=("__name__", "__doc__", "__annotations__"))
        def preprocess(self, *args, **kwargs):
            bound = contract_arguments(self, args, kwargs)
            if bound is None:
                return original(self, *args, **kwargs)
            return original(*bound.args, **bound.kwargs)

    setattr(preprocess, _HOOK_FLAG, True)
    setattr(target, method_name, preprocess)
    try:
        source = inspect.getsourcefile(target)
    except (TypeError, OSError):
        source = "<dynamic class>"
    logger.info(
        "Huanyu rollout input hook installed: %s.%s method=%s source=%s",
        target.__module__, target.__name__, method_name, source,
    )


def install_huanyu_rollout_inputs():
    """Install a Huanyu-gated in-memory hook; no upstream source files change."""
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

    _install_on_server_class(vLLMOmniHttpServer)
