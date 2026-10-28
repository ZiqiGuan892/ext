#!/usr/bin/env python3
"""Compare neutral Huanyu audio captures; never infer correctness from cosine alone."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


FORMAT = "huanyu_audio_probe_v1"


def _aligned_chunks(metadata, arrays):
    # Load the adjacent utility without importing the training adapter package
    # or altering sys.path in the diagnostic interpreter.
    path = Path(__file__).with_name("huanyu_probe_alignment.py")
    spec = importlib.util.spec_from_file_location("_huanyu_probe_alignment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.align_native_chunks(metadata, arrays)


def tensor_difference(left, right):
    result = {"left_shape": list(left.shape), "right_shape": list(right.shape)}
    if left.shape != right.shape:
        return dict(result, comparable=False, reason="shape_mismatch")
    a, b = left.astype(np.float64), right.astype(np.float64)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return dict(result, comparable=False, reason="nonfinite_values")
    delta = b - a
    reference_norm = float(np.linalg.norm(a.ravel()))
    other_norm = float(np.linalg.norm(b.ravel()))
    difference_norm = float(np.linalg.norm(delta.ravel()))
    result.update(
        comparable=True, values_exact=bool(np.array_equal(a, b)),
        max_abs=float(np.max(np.abs(delta))) if delta.size else 0.0,
        mean_abs=float(np.mean(np.abs(delta))) if delta.size else 0.0,
        rmse=float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        relative_l2=difference_norm / reference_norm if reference_norm else None,
        cosine=float(np.dot(a.ravel(), b.ravel()) / (reference_norm * other_norm))
        if reference_norm and other_norm else None,
    )
    if a.ndim == 2 and a.shape[0] and a.shape[1]:
        row_error = np.max(np.abs(delta), axis=1)
        rows = np.argsort(-row_error, kind="stable")[:5]
        result["worst_token_rows"] = [
            {"index": int(i), "max_abs": float(row_error[i])} for i in rows
        ]
    return result


def weight_difference(left, right):
    result = {}
    for component in sorted(set(left) | set(right)):
        a, b = left.get(component, {}), right.get(component, {})
        changes = []
        count = 0
        for kind in ("parameters", "buffers"):
            aa, bb = a.get(kind, {}), b.get(kind, {})
            for name in sorted(set(aa) | set(bb)):
                count += 1
                x, y = aa.get(name), bb.get(name)
                if x is None or y is None:
                    changes.append({"kind": kind, "name": name, "reason": "missing",
                                    "left_present": x is not None, "right_present": y is not None})
                    continue
                differing = {key: [x.get(key), y.get(key)] for key in
                             ("shape", "dtype", "sha256", "values_sha256")
                             if x.get(key) != y.get(key)}
                if differing:
                    changes.append({"kind": kind, "name": name, "differences": differing})
        result[component] = {"tensors_checked": count, "different_tensors": changes,
                             "left_training": a.get("training"), "right_training": b.get("training"),
                             "left_source": a.get("source"), "right_source": b.get("source")}
    return result


def read_capture(path):
    path = Path(path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("format") != FORMAT or metadata.get("status") != "complete":
        raise ValueError("Expected a complete Huanyu audio probe capture: {}".format(path))
    tensors = {}
    if metadata.get("tensors"):
        # Use the adjacent file, never an arbitrary path from captured metadata.
        with np.load(str(path.with_suffix(".npz")), allow_pickle=False) as archive:
            for name, record in metadata["tensors"].items():
                value = archive[record["array_key"]]
                if list(value.shape) != record["shape"]:
                    raise ValueError("Capture tensor shape does not match metadata: " + name)
                tensors[name] = value
    return metadata, tensors


def compare(left, left_arrays, right, right_arrays, left_group=None, right_group=None):
    report = {"left_phase": left.get("phase"), "right_phase": right.get("phase"),
              "left_reload_count": left.get("weight_reload_count"),
              "right_reload_count": right.get("weight_reload_count"),
              "weights": weight_difference(left.get("weights", {}), right.get("weights", {})),
              "environment_differences": {key: [left.get("environment", {}).get(key),
                                                  right.get("environment", {}).get(key)]
                  for key in sorted(set(left.get("environment", {})) | set(right.get("environment", {})))
                  if left.get("environment", {}).get(key) != right.get("environment", {}).get(key)}}
    report["run_metadata"] = {side: {key: meta.get(key) for key in (
        "checkpoint", "source", "tensor_parallel_size", "enforce_eager", "mm_processor_cache_gb",
        "audio_config", "encoder_downsample", "projector_downsample", "hook_counts",
    )} for side, meta in (("left", left), ("right", right))}
    aa, bb = left.get("audio_groups", []), right.get("audio_groups", [])
    if not aa or not bb:
        report["activation_comparison"] = "unavailable; weight snapshots only"
        return report
    if (left_group is None) != (right_group is None):
        raise ValueError("Specify both --left-group and --right-group")
    if left_group is not None:
        if not (0 <= left_group < len(aa) and 0 <= right_group < len(bb)):
            raise ValueError("Audio group index outside this capture")
        pairs = [(left_group, right_group)]
    elif len(aa) == len(bb) == 1:
        pairs = [(0, 0)]
    else:
        pairs = []
        for i, group in enumerate(aa):
            matches = [j for j, other in enumerate(bb) if other["feature_id"] == group["feature_id"]]
            left_count = sum(item["feature_id"] == group["feature_id"] for item in aa)
            if len(matches) == 1 and left_count == 1:
                pairs.append((i, matches[0]))
        if not pairs:
            raise ValueError("No unique matching feature groups; select known matching WAV groups explicitly")
    report["groups"] = []
    left_chunks, left_reason = _aligned_chunks(left, left_arrays)
    right_chunks, right_reason = _aligned_chunks(right, right_arrays)
    report["valid_chunk_alignment"] = {"left": left_reason, "right": right_reason}
    for i, j in pairs:
        a, b = aa[i], bb[j]
        same_input = a["feature_id"] == b["feature_id"]
        item = {"left_group": i, "right_group": j, "same_valid_features": same_input,
                "left_feature_id": a["feature_id"], "right_feature_id": b["feature_id"],
                "left_chunk_lengths": a["chunk_lengths"], "right_chunk_lengths": b["chunk_lengths"],
                "audio_embeddings": tensor_difference(left_arrays[a["output_path"]], right_arrays[b["output_path"]])}
        item["audio_embeddings"].update(
            left_dtype=left["tensors"][a["output_path"]]["dtype"],
            right_dtype=right["tensors"][b["output_path"]]["dtype"],
        )
        if len(a["feature_paths"]) == len(b["feature_paths"]):
            item["valid_feature_chunks"] = [
                tensor_difference(left_arrays[x][..., :nx], right_arrays[y][..., :ny])
                for x, y, nx, ny in zip(a["feature_paths"], b["feature_paths"],
                                        a["chunk_lengths"], b["chunk_lengths"])
            ]
        if not same_input:
            item["interpretation"] = "Features differ: downstream differences cannot isolate encoder/projector math"
        if left_chunks is not None and right_chunks is not None and len(a["chunk_lengths"]) == len(b["chunk_lengths"]):
            ai = sum(len(group["chunk_lengths"]) for group in aa[:i])
            bi = sum(len(group["chunk_lengths"]) for group in bb[:j])
            item["valid_module_chunks"] = [
                {stage: tensor_difference(left_chunks[ai + offset][stage], right_chunks[bi + offset][stage])
                 for stage in left_chunks[ai + offset]}
                for offset in range(len(a["chunk_lengths"]))
            ]
        report["groups"].append(item)
    report["unmatched_left_groups"] = sorted(set(range(len(aa))) - {i for i, _ in pairs})
    report["unmatched_right_groups"] = sorted(set(range(len(bb))) - {j for _, j in pairs})
    # A module may process a whole batch or individual chunks. Never pretend
    # encoder/0 on different batches represents the same sample.
    same_batch = [g["feature_id"] for g in aa] == [g["feature_id"] for g in bb]
    paths_a = {key for key in left_arrays if key.startswith(("encoder/", "projector/"))}
    paths_b = {key for key in right_arrays if key.startswith(("encoder/", "projector/"))}
    same_partition = paths_a == paths_b and all(
        left_arrays[key].shape == right_arrays[key].shape for key in paths_a if "/input/" in key
    )
    if same_batch and same_partition and paths_a:
        report["module_tensors"] = {
            key: dict(tensor_difference(left_arrays[key], right_arrays[key]),
                      left_dtype=left["tensors"][key]["dtype"], right_dtype=right["tensors"][key]["dtype"])
            for key in sorted(paths_a)
        }
    else:
        report["module_comparison"] = "Module hooks absent, or call partition/feature batch differs; inspect shapes, do not pair by index"
        report["module_shapes"] = {
            "left": {key: list(left_arrays[key].shape) for key in sorted(paths_a)},
            "right": {key: list(right_arrays[key].shape) for key in sorted(paths_b)},
        }
    report["notes"] = [
        "feature_id identifies valid processed features, not a WAV file or raw waveform",
        "audio_embeddings are get_audio_hidden_states results, not proof of their placement into LLM inputs",
        "No universal cosine/error threshold proves equivalent answers; inspect the earliest difference",
    ]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", help="Serve capture JSON, or before-reload weight snapshot")
    parser.add_argument("right", help="Rollout capture JSON, or after-reload weight snapshot")
    parser.add_argument("--left-group", type=int)
    parser.add_argument("--right-group", type=int)
    parser.add_argument("--out", help="Optional new JSON report; refuses to overwrite")
    args = parser.parse_args(argv)
    try:
        left, la = read_capture(args.left)
        right, ra = read_capture(args.right)
        result = compare(left, la, right, ra, args.left_group, args.right_group)
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.out:
            with Path(args.out).open("x", encoding="utf-8") as handle:
                handle.write(text + "\n")
        print(text)
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, "[huanyu-audio-compare] {}\n".format(error))


if __name__ == "__main__":
    raise SystemExit(main())
