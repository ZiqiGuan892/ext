"""Persist the selected AVQA training dataset's length-filter result.

Use ``bash run.sh --filter`` so model, chat template, and data settings are
composed from the same recipe as training. No Ray cluster or model weights are
loaded. Token lengths are decided by the installed dataset, not a second formula.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time


ROW_INDEX = "__avqa_offline_source_row__"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_files(value):
    values = [value] if isinstance(value, (str, Path)) else list(value)
    paths = [Path(p).expanduser().resolve(strict=True) for p in values]
    if not paths or any(p.suffix != ".parquet" or not p.is_file() for p in paths):
        raise ValueError("Expected one or more existing parquet input files")
    return paths


def filter_split(split, sources, output, build_dataset):
    """Use a temporary index to select untouched source rows after native filtering."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    tables = [pq.read_table(path) for path in sources]
    original = pa.concat_tables(tables)
    if ROW_INDEX in original.column_names:
        raise ValueError(f"Input contains reserved column {ROW_INDEX}")
    if not original.num_rows:
        raise ValueError(f"Empty {split} input")
    indexed = original.append_column(ROW_INDEX, pa.array(range(original.num_rows), type=pa.int64()))
    staging = output / "work" / f"{split}.indexed.parquet"
    staging.parent.mkdir(exist_ok=True)
    pq.write_table(indexed, staging, use_dictionary=False)
    print(f"[avqa-filter] {split}: processing {original.num_rows} rows", flush=True)
    started = time.monotonic()
    dataset = build_dataset(staging)
    kept = [int(i) for i in dataset.dataframe[ROW_INDEX]]
    if len(set(kept)) != len(kept) or any(i < 0 or i >= original.num_rows for i in kept):
        raise ValueError("Dataset returned invalid source row indices")
    if kept != sorted(kept):
        raise ValueError("Dataset changed source order; refusing to silently reshuffle")
    kept_set = set(kept)
    # Write the original Arrow rows, not processor outputs or mutated messages.
    # In particular, preserve nested columns, tagged labels, order and media paths.
    target = output / f"{split}.parquet"
    with (output / f"{split}.rows.jsonl").open("x", encoding="utf-8") as stream:
        offset = 0
        for source, table in zip(sources, tables, strict=True):
            extras = table["extra_info"].to_pylist() if "extra_info" in table.column_names else [{}] * table.num_rows
            for local_row, extra in enumerate(extras):
                index = offset + local_row
                selected = index in kept_set
                stream.write(json.dumps({
                    "source": str(source), "source_row": local_row, "combined_row": index,
                    "problem_id": (extra or {}).get("problem_id"), "kept": selected,
                    "reason": "kept" if selected else "upstream_rejected_overlong_or_processing_error",
                }, ensure_ascii=False) + "\n")
            offset += table.num_rows
    if not kept:
        raise ValueError(f"{split}: all rows rejected; inspect the filtering console log before training")
    temporary = target.with_suffix(".parquet.tmp")
    pq.write_table(original.take(pa.array(kept, type=pa.int64())), temporary, use_dictionary=False)
    temporary.replace(target)
    result = {
        "sources": [{"path": str(path), "sha256": sha256(path), "rows": table.num_rows}
                    for path, table in zip(sources, tables, strict=True)],
        "input_rows": original.num_rows, "kept_rows": len(kept),
        "rejected_rows": original.num_rows - len(kept), "elapsed_s": time.monotonic() - started,
        "output": str(target), "output_sha256": sha256(target),
    }
    print(f"[avqa-filter] {split}: {json.dumps(result, ensure_ascii=False)}", flush=True)
    return result


def file_identity(obj):
    path = Path(inspect.getfile(obj)).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def run_filter(config):
    from omegaconf import OmegaConf

    if not config.data.filter_overlong_prompts:
        raise ValueError("Offline filtering requires data.filter_overlong_prompts=true")
    if int(config.data.max_prompt_length) <= 0:
        raise ValueError("max_prompt_length must be positive")
    for field in ("train_max_samples", "val_max_samples"):
        if int(config.data.get(field) or -1) > 0:
            raise ValueError(f"Do not set data.{field}: offline filtering must retain the full original split")
    architecture = config.actor_rollout_ref.model.architecture
    datasets = {
        "Qwen3OmniMoeForConditionalGeneration": ("pkg://verl_omni.utils.dataset.omni_rl_datasets", "QwenOmniRLHFDataset"),
        "Qwen36HuanyuForTraining": ("pkg://verl_omni_ext.models.qwen3_6_huanyu.dataset", "Qwen36HuanyuOmniRLHFDataset"),
    }
    if (config.data.custom_cls.path, config.data.custom_cls.name) != datasets.get(architecture):
        raise ValueError(f"Use the AVQA training dataset registered for {architecture}")
    workers = int(config.data.filter_overlong_prompts_workers)
    if workers < 1:
        raise ValueError("AVQA_FILTER_WORKERS must be positive")
    sources = {split: source_files(value) for split, value in (
        ("train", config.data.train_files), ("validation", config.data.val_files))}
    output = Path(os.environ["AVQA_FILTER_OUTPUT_DIR"]).expanduser().resolve()
    # Never replace the source dataset or mix partial runs. A failed attempt
    # retains its logs/manifest; use a new directory to retry.
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "starting", "output_dir": str(output), "splits": {}, "workers": workers,
        "max_prompt_length": int(config.data.max_prompt_length), "weights_loaded": False,
        "data_config": OmegaConf.to_container(config.data, resolve=True),
        "model_config": OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True),
        "note": ("Rejections use the installed dataset predicate. "
                 "It may also reject processing errors; see the console log."),
    }
    write_json(output / "filter_manifest.json", manifest)
    OmegaConf.save(config, output / "resolved-config.yaml", resolve=True)
    print(f"[avqa-filter] output={output} workers={workers} "
          f"max_prompt_length={config.data.max_prompt_length}", flush=True)
    try:
        import multiprocess
        import torch
        from importlib.metadata import PackageNotFoundError, version
        from verl.utils.config import omega_conf_to_dataclass
        from verl.utils.dataset.rl_dataset import get_dataset_class

        # Spawn avoids forking after native/tokenizer libraries have initialized.
        # Each worker performs media preprocessing, never an encoder/LLM forward.
        multiprocess.set_start_method("spawn", force=True)
        torch.set_num_threads(1)
        model_config = omega_conf_to_dataclass(config.actor_rollout_ref.model)
        processor = model_config.get_processor()
        if architecture == "Qwen36HuanyuForTraining" and not getattr(processor, "_huanyu_training_processor", False):
            raise TypeError("Expected the real Huanyu training processor, not AutoProcessor/Whisper")
        if getattr(processor, "embedding_cache", None) is not None:
            raise ValueError("Offline length filtering must not use embedding caches")
        dataset_class = get_dataset_class(config.data)
        manifest["implementation"] = {
            "filter": file_identity(dataset_class.maybe_filter_out_long_prompts),
            "dataset": file_identity(dataset_class), "processor": file_identity(type(processor)),
        }
        packages = {}
        for name in ("verl", "verl-omni", "transformers", "datasets", "torch", "torch-npu"):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        manifest["packages"] = packages
        manifest["model_artifacts"] = {}
        for root in {Path(model_config.local_path), Path(model_config.local_tokenizer_path)}:
            for pattern in ("*.json", "*.jinja", "*.model", "merges.txt", "chat_templates/*.jinja"):
                for path in sorted(root.glob(pattern)):
                    if path.is_file():
                        manifest["model_artifacts"][str(path)] = sha256(path)

        def build_dataset(staging):
            return dataset_class(data_files=[str(staging)], tokenizer=model_config.tokenizer,
                                 processor=processor, config=config.data, max_samples=-1)

        for split, paths in sources.items():
            manifest.update(status="filtering", phase=split)
            write_json(output / "filter_manifest.json", manifest)
            manifest["splits"][split] = filter_split(split, paths, output, build_dataset)
            write_json(output / "filter_manifest.json", manifest)
        manifest.update(status="complete", phase="finished")
        write_json(output / "filter_manifest.json", manifest)
        print(f"[avqa-filter] COMPLETE: {output}", flush=True)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(output / "filter_manifest.json", manifest)
        raise


def main():
    # Import the same registration/config modules as the training entrypoint,
    # but never call its main(), run_omni(), or ray.init().
    import hydra
    import verl_omni.trainer.main_omni  # noqa: F401

    entrypoint = hydra.main(config_path="pkg://verl_omni.trainer.config",
                           config_name="omni_trainer", version_base=None)(run_filter)
    entrypoint()


if __name__ == "__main__":
    main()
