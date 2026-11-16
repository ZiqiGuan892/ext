"""Download/convert the published AVQA data and audit it without loading weights.

Run this file directly; importing the Huanyu package would initialize the model
stack. The official converter and choice reward are bundled unchanged beside
this file.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tarfile

CONVERTER_PATH = Path(__file__).resolve().with_name("avqa.py")
REWARD_PATH = Path(__file__).resolve().with_name("choice_reward.py")
REPO = "harryhsing/AVQA-R1-6K"
REVISION = "1c380427f66407f9f4ece354d84f804e4a16e522"
ARCHIVE_SHA256 = "6ad5d4bc00a3367aa36d2b0ed062a0a53c8d701242110c43826541138522f388"
DATA_DEFAULT = "/opt/huawei/dataset/infra_data/data/avqa_r1_6k"
MODEL_DEFAULT = "/opt/huawei/dataset/model_gy/downloaded_models/2748"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
EXCLUDE = re.compile(r".*(?:talker|code2wav|code_predictor|visual|audio_tower|audio_encoder|audio_projection).*")


def load_file(path):
    spec = importlib.util.spec_from_file_location(f"avqa_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unpack(archive, destination):
    """Verify the published archive and never overlay an incomplete extraction."""
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 differs from the published AVQA-R1-6K archive")
    marker = destination / ".avqa-extracted.json"
    if marker.exists():
        if json.loads(marker.read_text())["sha256"] != ARCHIVE_SHA256:
            raise ValueError(f"Different extraction at {destination}")
        return
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Incomplete/unverified extraction: {destination}; use a fresh --data-dir")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(destination, filter="data")
    marker.write_text(json.dumps({"sha256": ARCHIVE_SHA256}), encoding="utf-8")


def prepare(data_dir, archive):
    converter_path = CONVERTER_PATH
    if not converter_path.is_file():
        raise FileNotFoundError(f"Official converter missing: {converter_path}")
    data_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        from huggingface_hub import hf_hub_download

        archive = Path(hf_hub_download(
            repo_id=REPO, repo_type="dataset", revision=REVISION,
            filename="AVQA_R1.tar.gz", local_dir=data_dir / "download",
        ))
    print(f"Verifying and extracting {archive}", flush=True)
    unpack(archive, data_dir / "raw")
    sources = list((data_dir / "raw").rglob("omni_rl_format_train.json"))
    if len(sources) != 1:
        raise ValueError(f"Expected one original train JSON, found {len(sources)}")
    raw_root = sources[0].parent.parent
    converter = load_file(converter_path)
    stats = {}
    for split, source in (("train", sources[0]),
                          ("validation", raw_root / "valid/omni_rl_format_valid.json")):
        stats[split] = converter.convert_split(source, data_dir / f"{split}.parquet", split)
        stats[split]["source_sha256"] = sha256(source)
        print(json.dumps({"split": split, **stats[split]}, ensure_ascii=False), flush=True)
    manifest = {
        "dataset": REPO, "revision": REVISION, "archive_sha256": ARCHIVE_SHA256,
        "converter_sha256": sha256(converter_path), "splits": stats,
        "note": "Counts are before Huanyu tokenizer length filtering. No train/validation reshuffle.",
    }
    (data_dir / "preparation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def check_data(path):
    import pyarrow.parquet as pq

    scorer = load_file(REWARD_PATH)
    converter = load_file(CONVERTER_PATH)
    counts = Counter()
    ids = []
    table = pq.read_table(path)
    if not table.num_rows:
        raise ValueError(f"Empty dataset: {path}")
    for i, row in enumerate(table.to_pylist()):
        if row.get("data_source") != "avqa_r1_6k":
            raise ValueError(f"{path} row {i}: incorrect data_source")
        target = row["reward_model"]["ground_truth"]
        answer = scorer.extract_answer(target)
        options = json.loads(row["extra_info"]["options"])
        # The unchanged scorer treats two missing tags as equal. Reject invalid
        # labels here rather than changing the reward or silently awarding 1.
        if not re.fullmatch(r"[A-Ha-h]", answer) or answer.upper() not in options:
            raise ValueError(f"{path} row {i}: invalid tagged answer {target!r}")
        if scorer.compute_score(solution_str=target, ground_truth=target)["score"] != 1:
            raise ValueError(f"{path} row {i}: reward self-check failed")
        if scorer.compute_score(solution_str="missing answer", ground_truth=target)["score"] != 0:
            raise ValueError(f"{path} row {i}: invalid answer receives reward")
        prompt = row["prompt"]
        if (len(prompt) != 2 or prompt[0]["content"] != converter.SYSTEM_PROMPT
                or not prompt[1]["content"].startswith("<image><audio>")):
            raise ValueError(f"{path} row {i}: prompt differs from the official converter")
        images, audios = row["images"], row["audios"]
        if len(images) != 1 or len(audios) != 1:
            raise ValueError(f"{path} row {i}: expected one image and one audio")
        for media in (images[0]["image"], audios[0]):
            if not Path(media).is_absolute() or not Path(media).is_file():
                raise FileNotFoundError(f"{path} row {i}: missing absolute media path {media}")
        counts[answer] += 1
        ids.append(str(row["extra_info"]["problem_id"]))
    return {"path": str(path), "rows": table.num_rows, "answers": dict(counts),
            "sha256": sha256(path), "unique_problem_ids": len(set(ids))}


def check_model(model):
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if (config.get("model_type") != "qwen3_5_moe"
            or config.get("audio_config", {}).get("model_type") != "huanyu_conformer_0_6b"):
        raise ValueError("Expected Huanyu 2748 config; do not select the Whisper adapter")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        keys = list(json.loads(index.read_text(encoding="utf-8"))["weight_map"])
    else:
        from safetensors import safe_open

        keys = []
        for path in sorted(model.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                keys.extend(handle.keys())
    selected = {
        target: sorted(k for k in keys if k.endswith(f".{target}.weight") and not EXCLUDE.fullmatch(k))
        for target in TARGETS
    }
    if any(not values for values in selected.values()):
        raise ValueError(f"Some official LoRA targets are absent: { {k: len(v) for k, v in selected.items()} }")
    return {"model": str(model), "model_type": config["model_type"],
            "config_sha256": sha256(model / "config.json"),
            "candidate_weight_counts": {key: len(values) for key, values in selected.items()},
            "candidate_weights": selected, "weights_loaded": False,
            "note": "Checkpoint names only; verify actual PEFT trainable parameters in the training log."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--data-dir", type=Path, default=Path(DATA_DEFAULT))
    parser.add_argument("--archive", type=Path, help="Use an already downloaded AVQA_R1.tar.gz")
    parser.add_argument("--model", type=Path, default=Path(MODEL_DEFAULT))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_dir.resolve(), args.archive)
    else:
        from importlib.metadata import PackageNotFoundError, version

        packages = {}
        for package in ("torch", "torch-npu", "transformers", "peft", "verl", "verl-omni", "vllm", "vllm-omni"):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = "missing metadata"
        report = {"python": sys.executable, "packages": packages,
                  "datasets": [check_data(args.data_dir / f"{s}.parquet") for s in ("train", "validation")],
                  "checkpoint": check_model(args.model),
                  "reward_sha256": sha256(REWARD_PATH)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        (args.data_dir / "preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
