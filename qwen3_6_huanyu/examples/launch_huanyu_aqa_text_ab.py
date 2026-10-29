#!/usr/bin/env python3
"""Launch either side of a prepared AQA text comparison (no environment/Ray setup).

Python 3.8+. The case manifest is shared; standalone serving and verl validation
run sequentially when they use the same devices. No optimizer steps are taken.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def load_case(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    case = json.loads((root / "case.json").read_text(encoding="utf-8"))
    if case.get("schema_version") != 1:
        raise ValueError("unsupported comparison case schema")
    return root, case


def hydra_string(value):
    return json.dumps(str(value), ensure_ascii=False)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_launch(root, case, project, mode, port=28889, served_model="qwen3.8"):
    """Return an argv/environment/deploy plan; never execute or import vLLM here."""
    sampling, engine = case["sampling"], case["engine"]
    for key, expected in {"temperature": 0, "top_p": 1, "top_k": -1, "n": 1,
                          "repetition_penalty": 1, "frequency_penalty": 0,
                          "presence_penalty": 0, "min_tokens": 0, "ignore_eos": False}.items():
        if sampling.get(key) != expected:
            raise ValueError("comparison requires sampling.{}={!r}".format(key, expected))
    if engine["dtype"] != "bfloat16" or not isinstance(case["enable_thinking"], bool):
        raise ValueError("comparison requires BF16 and a boolean enable_thinking")
    checkpoint = str(case["model_path"])
    tp = int(engine["tensor_parallel_size"])
    if tp != 4:
        raise ValueError("this comparison launcher currently requires TP=4 on both sides")
    if not 1 <= port <= 65535 or not served_model.strip():
        raise ValueError("invalid port or served model name")
    examples = project / "examples/qwen3_6_huanyu"
    prompt_length = int(engine["prompt_length"])
    max_tokens = int(sampling["max_tokens"])
    context = int(engine["max_model_len"])
    seed = int(sampling["seed"])
    if prompt_length < 1 or max_tokens < 1 or context < prompt_length + max_tokens or not 0 <= seed < 2**63:
        raise ValueError("invalid prompt/response/context budget or seed")
    for key, expected in {"max_num_batched_tokens": context, "max_num_seqs": 1,
                          "gpu_memory_utilization": 0.5, "enforce_eager": True}.items():
        if engine.get(key) != expected:
            raise ValueError("comparison requires engine.{}={!r}".format(key, expected))
    env = dict(os.environ)
    # Reused shells may still carry the old weight/tensor diagnostics. This run
    # compares only text and retains the normal initial Actor synchronization.
    disabled = (
        "HUANYU_CHECKPOINT_VALIDATION_MODE", "HUANYU_WEIGHT_SYNC_AUDIT_DIR",
        "HUANYU_AUDIO_PROBE_DIR", "HUANYU_AUDIO_PROBE_ARM_FILE", "HUANYU_AUDIO_PROBE_FEATURE_ID",
    )
    env.update({key: "" for key in disabled})
    env.update({
        "PROJECT_ROOT": str(project), "MODEL_PATH": checkpoint,
        "HUANYU_PRECISION_PROFILE": "full", "HUANYU_GRAPH_MODE": "0",
        "ROLLOUT_TP": str(tp), "HUANYU_USE_PRECOMPUTE": "0", "MINICPMO_USE_PRECOMPUTE": "0",
        "HUANYU_EMBED_CACHE_DIR": "", "HUANYU_FREEZE_AUDIO": "true",
        "HUANYU_TRAIN_AUDIO_ENCODER": "false", "HUANYU_TRAIN_AUDIO_PROJECTOR": "false",
        "MINICPMO_TUNE_AUDIO": "0", "MINICPMO_TUNE_VISION": "0",
        "ENABLE_THINKING": "1" if case["enable_thinking"] else "0",
        "VLLM_ASCEND_WEIGHT_NZ_MODE": "0", "ASCEND_WEIGHT_NZ_MODE": "0",
        "ASCEND_ENABLE_TRANS_DATA_CACHE": "0",
        "VLLM_ASCEND_ENABLE_NZ": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_V1": "0", "USE_OPTIMIZED_MODEL": "0", "MULTI_STREAM_MEMORY_REUSE": "2",
        "TASK_QUEUE_ENABLE": "0", "ASCEND_LAUNCH_BLOCKING": "0",
        "HCCL_OP_EXPANSION_MODE": env.get("HCCL_OP_EXPANSION_MODE") or "AIV",
    })
    # Optional arithmetic switches must not leak different shell defaults into
    # the two runs. full sets the remaining reference switches to one itself.
    for key in (
        "MINICPMO_HF_NORM_REFERENCE", "MINICPMO_HF_ROUTER_REFERENCE", "MINICPMO_HF_TEXT_ROPE_REFERENCE",
        "MINICPMO_MOE_MATCH_ROLLOUT", "MINICPMO_ISOLATE_FULL_ATTENTION",
        "MINICPMO_VISION_POSITION_REFERENCE", "MINICPMO_NATIVE_VISION_REFERENCE",
        "MINICPMO_MOE_TP_REFERENCE", "MINICPMO_ROW_TP_REFERENCE",
    ):
        env[key] = "0"
    deploy = None
    if mode == "serve":
        native = "/vllm-workspace/vllm-omni"
        old_paths = (
            "/opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy",
            "/opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy",
            "/opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit",
            "/opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit",
        )
        retained = [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                    if p and p not in (native, str(project))
                    and not any(p == old or p.startswith(old + "/") for old in old_paths)]
        env["PYTHONPATH"] = os.pathsep.join([native, str(project)] + retained)
        env["HUANYU_TEXT_AB_DIR"] = ""  # AgentLoop capture is for verl only.
        extras = {
            "model_arch": "MiniCPMO50OmniForConditionalGeneration", "model_stage": "llm",
            "interleave_mm_strings": True,
            "limit_mm_per_prompt": {"image": 96, "audio": 64, "video": 1},
            "media_io_kwargs": {"video": {"fps": 1, "num_frames": 128}},
            "additional_config": {"weight_nz_mode": 0, "rl_config": {"enabled": True}},
            "worker_extension_cls": "verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension",
        }
        engine_values = {
            "tensor_parallel_size": tp, "dtype": engine["dtype"],
            "max_model_len": context, "max_num_batched_tokens": context, "max_num_seqs": 1,
            "gpu_memory_utilization": 0.5, "enforce_eager": True,
            "enable_prefix_caching": False, "enable_chunked_prefill": False,
            "mm_processor_cache_gb": 0, "seed": seed, "generation_config": "vllm",
        }
        # This is the same registered pipeline/stage/extras used by the Huanyu
        # rollout adapter. JSON is valid YAML and needs no YAML dependency.
        deploy = {"pipeline": "qwen3_6_whisper_thinker_only", "stages": [{
            "stage_id": 0, "devices": ",".join(str(i) for i in range(tp)),
            **engine_values, "engine_extras": extras,
            "default_sampling_params": dict(sampling, detokenize=True),
        }]}
        command = [
            "vllm", "serve", checkpoint, "--omni", "--port", str(port),
            "--served-model-name", served_model, "--trust-remote-code", "--stage-init-timeout", "600",
            "--deploy-config", str(root / "serve.deploy.yaml"), "--dtype", str(engine["dtype"]),
            "--tensor-parallel-size", str(tp), "--max-model-len", str(context),
            "--max-num-batched-tokens", str(context), "--max-num-seqs", "1",
            "--gpu-memory-utilization", "0.5", "--seed", str(seed), "--enforce-eager",
            "--generation-config", "vllm", "--no-enable-prefix-caching", "--no-enable-chunked-prefill",
            "--mm-processor-cache-gb", "0", "--chat-template-content-format", "string",
            "--additional-config", json.dumps(extras["additional_config"], separators=(",", ":")),
            "--interleave-mm-strings", "--log-stats",
        ]
        env["HUANYU_TEXT_AB_SERVE_LOG"] = str(root / "serve_console.log")
        shell = ('set -euo pipefail\n'
                 'source "$PROJECT_ROOT/examples/qwen3_6_huanyu/graph_mode.sh"\n'
                 'source "$PROJECT_ROOT/examples/qwen3_6_huanyu/precision_mode.sh"\n'
                 'exec > >(tee -a "$HUANYU_TEXT_AB_SERVE_LOG") 2>&1\n'
                 'exec "$@"\n')
        argv = ["bash", "-c", shell, "huanyu-text-ab-serve", *command]
    else:
        subset = str(case["subset_parquet"])
        env.update({
            "HUANYU_TEXT_AB_DIR": str(root), "TRAIN_FILE": subset, "VAL_FILE": subset,
            "HUANYU_AQA_TRAIN_FILE": subset, "HUANYU_AQA_VAL_FILE": subset,
            "NUM_NPUS": "16", "NNODES": "1", "RM_ENABLE": "1",
            "RM_N_GPUS_PER_NODE": "16", "RM_NNODES": "1", "RM_TP": "4", "RM_SPLIT_NODES": "1",
            "REWARD_POOL_NPUS": "16", "TRAIN_BATCH_SIZE": "4", "PPO_MINI_BATCH_SIZE": "4", "ROLLOUT_N": "4",
            "RM_MODEL_PATH": env.get("RM_MODEL_PATH") or "/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B",
            "MAX_PROMPT_LENGTH": str(prompt_length), "ROLLOUT_PROMPT_LENGTH": str(prompt_length),
            "MINICPMO_MM_PROMPT_BUDGET": str(prompt_length), "MAX_RESPONSE_LENGTH": str(max_tokens),
            "ROLLOUT_MAX_MODEL_LEN": str(context), "MAX_TOKEN_LEN_PER_GPU": str(context),
            "ROLLOUT_TEMPERATURE": "0", "ROLLOUT_TOP_P": "1", "ROLLOUT_TOP_K": "-1",
            "ROLLOUT_REPETITION_PENALTY": "1", "ROLLOUT_FREQUENCY_PENALTY": "0", "ROLLOUT_PRESENCE_PENALTY": "0",
            "ROLLOUT_NUM_WORKERS": "4", "DEBUG_MODE": "0", "DEBUG_SAMPLES": "0",
            "TOTAL_TRAINING_STEPS": "1", "VAL_BEFORE_TRAIN": "true", "TEST_FREQ": "-1",
            "EXPERIMENT_NAME": "huanyu_aqa_text_ab", "HUANYU_SKIP_EDITABLE_INSTALL": "1",
            "RAY_ADDRESS": env.get("RAY_ADDRESS") or "auto",
        })
        command = ["bash", str(examples / "run_huanyu_aqa_audio_train.sh"),
                   "trainer.val_only=true", "trainer.val_before_train=true", "trainer.resume_mode=disable",
                   "trainer.save_freq=-1", "trainer.total_epochs=1", "data.shuffle=false",
                   "data.validation_shuffle=false", "data.val_batch_size=1", "data.train_max_samples=-1",
                   "data.val_max_samples=-1", "data.seed={}".format(seed), "actor_rollout_ref.rollout.load_format=safetensors",
                   "actor_rollout_ref.rollout.free_cache_engine=false", "actor_rollout_ref.rollout.dtype=bfloat16",
                   "actor_rollout_ref.rollout.max_num_seqs=1", "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
                   "actor_rollout_ref.rollout.enable_chunked_prefill=false",
                   "actor_rollout_ref.rollout.enable_prefix_caching=false",
                   "actor_rollout_ref.rollout.agent.num_workers=4",
                   "actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent",
                   "actor_rollout_ref.rollout.temperature=0", "actor_rollout_ref.rollout.top_p=1",
                   "actor_rollout_ref.rollout.top_k=-1", "actor_rollout_ref.rollout.ignore_eos=false",
                   "++actor_rollout_ref.rollout.repetition_penalty=1",
                   "actor_rollout_ref.rollout.val_kwargs.n=1", "actor_rollout_ref.rollout.val_kwargs.temperature=0",
                   "actor_rollout_ref.rollout.val_kwargs.top_p=1", "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
                   "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.frequency_penalty=0",
                   "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.presence_penalty=0",
                   "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.min_tokens=0",
                   "++actor_rollout_ref.rollout.engine_kwargs.q36_sampling.seed={}".format(seed),
                   "actor_rollout_ref.rollout.seed={}".format(seed),
                   "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.generation_config=vllm",
                   "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0",
                   "++data.apply_chat_template_kwargs.enable_thinking={}".format(str(case["enable_thinking"]).lower()),
                   "trainer.validation_data_dir=" + hydra_string(root / "validation_dump"),
                   "trainer.rollout_data_dir=" + hydra_string(root / "rollout_dump"),
                   "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_TEXT_AB_DIR=" + hydra_string(root)]
        command.extend("++ray_kwargs.ray_init.runtime_env.env_vars.{}=\"\"".format(key) for key in disabled)
        argv = command
    return {"mode": mode, "case_id": case["case_id"], "command": command,
            "argv": argv, "env": env, "deploy": deploy}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "rollout"))
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--project-root", default=os.environ.get("PROJECT_ROOT"))
    parser.add_argument("--port", type=int, default=28889)
    parser.add_argument("--served-model", default="qwen3.8")
    parser.add_argument("--dry-run", action="store_true", help="Print the requested launch plan without starting processes")
    args = parser.parse_args(argv)
    try:
        if not args.project_root:
            raise ValueError("set PROJECT_ROOT or --project-root")
        root, case = load_case(args.case_dir)
        project = Path(args.project_root).expanduser().resolve(strict=True)
        plan = build_launch(root, case, project, args.mode, args.port, args.served_model)
        print("[huanyu-text-ab] " + " ".join(shlex.quote(arg) for arg in plan["command"]), flush=True)
        if args.dry_run:
            return 0
        if not Path(case["model_path"]).is_dir() or not Path(case["subset_parquet"]).is_file():
            raise ValueError("prepared model path or subset Parquet is absent on this machine")
        if file_hash(case["subset_parquet"]) != case["subset_parquet_sha256"]:
            raise ValueError("prepared subset Parquet changed")
        if file_hash(root / "manifest.jsonl") != case["manifest_sha256"]:
            raise ValueError("prepared case manifest changed")
        for name, expected in case["checkpoint_file_sha256"].items():
            path = Path(case["model_path"]) / name
            if (file_hash(path) if path.is_file() else None) != expected:
                raise ValueError("checkpoint configuration changed: " + name)
        target = root / (args.mode + ".launch.json")
        # Deliberately do not persist the ambient environment (it may contain credentials).
        snapshot = {key: value for key, value in plan.items() if key != "env"}
        snapshot["selected_environment"] = {
            key: plan["env"][key] for key in (
                "HUANYU_PRECISION_PROFILE", "HUANYU_GRAPH_MODE", "HCCL_OP_EXPANSION_MODE",
                "VLLM_ASCEND_WEIGHT_NZ_MODE", "USE_OPTIMIZED_MODEL",
                "HUANYU_TRAIN_AUDIO_ENCODER", "HUANYU_TRAIN_AUDIO_PROJECTOR",
            )
        }
        with target.open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if plan["deploy"]:
            with (root / "serve.deploy.yaml").open("x", encoding="utf-8") as stream:
                json.dump(plan["deploy"], stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        return subprocess.call(plan["argv"], env=plan["env"])
    except (OSError, ValueError, KeyError) as exc:
        print("[huanyu-text-ab] ERROR: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
