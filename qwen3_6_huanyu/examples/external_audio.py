"""Lazy, bounded materialization of audio stored outside an SSTable row."""

from __future__ import annotations

import io
import json
import os
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ExternalMediaError(RuntimeError):
    """Recoverable data error with fields suitable for drop-stat reporting."""

    def __init__(self, error_type: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.error_type = str(error_type)
        self.details: Dict[str, Any] = dict(details)
        self.details.setdefault("absolute_path", None)
        self.details.setdefault("relative_path", None)
        self.details.setdefault("media_type", "audio")


class ExternalAudioReference(BaseModel):
    """Location of one ``AudioContent`` placeholder in a serialized train unit."""

    conversation_index: int = Field(ge=0)
    content_index: int = Field(ge=0)
    absolute_path: str
    relative_path: str
    archive_key: Optional[str] = None
    source_path: Optional[str] = None


def _configured_root(audio_root: Optional[str]) -> Optional[Path]:
    value = (audio_root or "").strip() or os.environ.get("AL_AUDIO_ROOT", "").strip()
    return Path(value).expanduser() if value else None


def _reference_details(reference: ExternalAudioReference, **extra: Any) -> Dict[str, Any]:
    details = {
        "media_type": "audio",
        "absolute_path": reference.absolute_path,
        "relative_path": reference.relative_path,
        "archive_key": reference.archive_key,
        "source_path": reference.source_path,
        "conversation_index": reference.conversation_index,
        "content_index": reference.content_index,
    }
    details.update(extra)
    return details


def resolve_audio_path(reference: ExternalAudioReference, audio_root: Optional[str]) -> Path:
    """Prefer the runtime absolute path, then ``audio_root/relative_path``."""

    absolute = Path(reference.absolute_path).expanduser()
    if not absolute.is_absolute():
        raise ExternalMediaError(
            "audio_invalid_path",
            "external audio absolute_path must be absolute",
            **_reference_details(reference, audio_root=audio_root),
        )
    if absolute.is_file():
        return absolute

    root = _configured_root(audio_root)
    relative = Path(reference.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ExternalMediaError(
            "audio_invalid_path",
            "unsafe external audio relative path",
            **_reference_details(reference, audio_root=str(root) if root else None),
        )
    if root is not None:
        candidate = root / relative
        if candidate.is_file():
            return candidate

    raise ExternalMediaError(
        "audio_missing",
        "external audio file or archive does not exist",
        **_reference_details(reference, audio_root=str(root) if root else None),
    )


def _audio_array_to_wav(audio: Any, sample_rate: int) -> bytes:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to decode external archive audio") from exc
    output = io.BytesIO()
    sf.write(output, audio, int(sample_rate), format="WAV", subtype="PCM_16")
    return output.getvalue()


def _read_zarr_zip(path: Path, key: str) -> bytes:
    try:
        import numpy as np
        from numcodecs import get_codec
    except ImportError as exc:
        raise RuntimeError(
            "numpy and numcodecs are required to read .zarr.zip external audio"
        ) from exc

    def read_array(archive: zipfile.ZipFile, array_path: str, allow_scalar: bool = False) -> Any:
        metadata_name = array_path.rstrip("/") + "/.zarray"
        try:
            metadata = json.loads(archive.read(metadata_name))
        except KeyError as exc:
            raise KeyError(
                "Zarr array {!r} is absent from {}".format(array_path, path)
            ) from exc

        shape = tuple(int(item) for item in metadata["shape"])
        chunks = tuple(int(item) for item in metadata["chunks"])
        scalar = allow_scalar and not shape and not chunks
        if not scalar and (len(shape) != 1 or len(chunks) != 1):
            raise ValueError(
                "only one-dimensional AL audio arrays are supported: {}".format(
                    array_path
                )
            )

        dtype = np.dtype(metadata["dtype"])
        compressor = (
            get_codec(metadata["compressor"])
            if metadata.get("compressor") is not None
            else None
        )
        filters = [get_codec(item) for item in (metadata.get("filters") or [])]
        arrays = []
        if not scalar and (shape[0] < 0 or chunks[0] <= 0):
            raise ValueError("invalid Zarr shape/chunks for {}".format(array_path))
        chunk_count = 1 if scalar else (shape[0] + chunks[0] - 1) // chunks[0]
        for chunk_index in range(chunk_count):
            chunk_name = array_path.rstrip("/") + "/{}".format(chunk_index)
            try:
                encoded = archive.read(chunk_name)
            except KeyError:
                count = 1 if scalar else min(chunks[0], shape[0] - chunk_index * chunks[0])
                arrays.append(
                    np.full(count, metadata.get("fill_value", 0), dtype=dtype)
                )
                continue
            decoded: Any = compressor.decode(encoded) if compressor else encoded
            for filter_codec in reversed(filters):
                decoded = filter_codec.decode(decoded)
            if hasattr(decoded, "tobytes"):
                decoded = decoded.tobytes()
            arrays.append(np.frombuffer(decoded, dtype=dtype))

        if not arrays:
            return np.empty(shape, dtype=dtype)
        return np.concatenate(arrays)[: 1 if scalar else shape[0]]

    candidates = [key, key[:-4]] if key.endswith(".wav") else [key + ".wav", key]
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        selected = next(
            (
                candidate
                for candidate in candidates
                if candidate + "/.zgroup" in names
            ),
            None,
        )
        if selected is None:
            raise KeyError("audio key {!r} is absent from {}".format(key, path))
        audio = read_array(archive, selected + "/audio")
        if not audio.size or not np.isfinite(audio).all():
            raise ValueError("Zarr audio must contain non-empty finite samples")
        # AQA's training reader uses group attributes. Other AL producers store
        # a one-element or scalar array instead. Do not silently guess a rate.
        attrs_name = selected + "/.zattrs"
        attrs = json.loads(archive.read(attrs_name)) if attrs_name in names else {}
        sample_rate = attrs.get("sampling_rate", attrs.get("sample_rate"))
        if sample_rate is None:
            values = read_array(archive, selected + "/sample_rate", allow_scalar=True)
            if values.size != 1:
                raise ValueError("sample_rate array must contain exactly one value")
            sample_rate = values[0]
        rate = float(sample_rate)
        if isinstance(sample_rate, (bool, np.bool_)) or not np.isfinite(rate) or rate <= 0 or not rate.is_integer():
            raise ValueError("sample_rate must be a positive integer: {!r}".format(sample_rate))
        sample_rate = int(rate)
    return _audio_array_to_wav(audio, sample_rate)


def _read_tar(path: Path, key: str) -> bytes:
    candidates = [key] if key.endswith(".wav") else [key, key + ".wav"]
    with tarfile.open(path, mode="r:*") as archive:
        names = archive.getnames()
        selected = next((name for name in candidates if name in names), None)
        if selected is None:
            by_basename = {Path(name).name: name for name in names}
            selected = next(
                (by_basename[name] for name in candidates if name in by_basename), None
            )
        if selected is None:
            raise KeyError("audio key {!r} is absent from {}".format(key, path))
        handle = archive.extractfile(selected)
        if handle is None:
            raise IOError("cannot read tar member {!r} from {}".format(selected, path))
        return handle.read()


def read_external_audio(reference: ExternalAudioReference, audio_root: Optional[str]) -> bytes:
    path = resolve_audio_path(reference, audio_root)
    key = reference.archive_key
    try:
        if path.name.lower().endswith(".zarr.zip"):
            if not key:
                raise ValueError("archive_key is required for zarr.zip")
            payload = _read_zarr_zip(path, key)
        elif tarfile.is_tarfile(path):
            if not key:
                raise ValueError("archive_key is required for tar audio")
            payload = _read_tar(path, key)
        else:
            if key:
                raise ValueError("archive_key was provided for a direct audio file")
            payload = path.read_bytes()
        if not payload:
            raise ValueError("external audio payload is empty")
        return payload
    except ExternalMediaError:
        raise
    except Exception as exc:
        raise ExternalMediaError(
            "audio_read_error",
            "failed to materialize external audio: {}: {}".format(
                type(exc).__name__, exc
            ),
            **_reference_details(reference, resolved_path=str(path)),
        ) from exc


def materialize_external_audio(
    train_unit: Any,
    references: list[ExternalAudioReference],
    audio_root: Optional[str],
) -> Any:
    """Replace referenced placeholders only after rank and row selection."""

    conversations = getattr(train_unit, "conversations", None)
    if conversations is None:
        raise ExternalMediaError(
            "audio_invalid_reference",
            "external audio requires a train unit with conversations",
            media_type="audio",
            train_unit_type=str(type(train_unit)),
        )
    for reference in references:
        try:
            content = conversations[reference.conversation_index].content[
                reference.content_index
            ]
        except (IndexError, AttributeError) as exc:
            raise ExternalMediaError(
                "audio_invalid_reference",
                "external audio position is outside the train unit",
                **_reference_details(reference),
            ) from exc
        if getattr(content, "type", None) != "audio":
            raise ExternalMediaError(
                "audio_invalid_reference",
                "external audio position does not point to AudioContent",
                **_reference_details(reference),
            )
        content.audio_buffer = read_external_audio(reference, audio_root)
    return train_unit
