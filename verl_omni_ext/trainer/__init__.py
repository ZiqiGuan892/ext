"""训练范式注册目录 + main_omni env 注入 patch。

L2 patch：扩展 verl 的 ``get_ppo_ray_runtime_env`` 白名单，让
``MINICPMO_*`` / ``GENRM_*`` / ``HCCL_*`` 等环境变量透传到 Ray worker。

原因：verl 的 ``get_ppo_ray_runtime_env()`` 白名单（``PPO_RAY_RUNTIME_ENV``）
不含 verl-omni 专用变量（``MINICPMO_EMBED_CACHE_DIR`` 等），launch script
``export`` 的变量不进 Ray worker。

方案：patch ``verl.trainer.constants_ppo.get_ppo_ray_runtime_env``，在原函数
返回的 runtime_env 里追加 verl-omni 专用变量。这个函数在 verl 包内，不被
``runpy -m`` 覆盖（不像 ``verl_omni.trainer.main_omni.run_omni``）。

代码1 在 ``main_omni.py`` 的 ``run_omni`` 里直接写 ``config.ray_kwargs``，
代码2 作为 out-of-tree 插件仓不能改上游源码，改用扩展白名单函数。

See memory: feedback_ray_env_vars_propagation.md
"""
from verl_omni_ext._patchkit import idempotent_patch

# ============================================================================
# L2 patch：扩展 get_ppo_ray_runtime_env 白名单
# ============================================================================
@idempotent_patch(
    name="inject_omni_env_vars",
    target_module="verl.trainer.constants_ppo",
    target_attr="get_ppo_ray_runtime_env",
    fingerprint="verl>=0.2.0",
    expected_signature=None,
    probe_signature=True,
)
def _patch_get_ppo_ray_runtime_env(original_fn, __patch_signature__=None):
    """扩展 get_ppo_ray_runtime_env 白名单，透传 verl-omni 专用 env vars。"""

    import os

    _ENV_PREFIXES = (
        "MINICPMO_",
        "GENRM_",
        "HCCL_",
        "ACL_DEVICE_SYNC_TIMEOUT",
        "TORCH_DIST_TIMEOUT",
    )

    def _patched_get_ppo_ray_runtime_env(config=None):
        runtime_env = original_fn(config)

        injected = []
        for key, val in os.environ.items():
            if any(key.startswith(prefix) for prefix in _ENV_PREFIXES):
                runtime_env["env_vars"][key] = val
                injected.append(f"{key}={val}")

        if injected:
            print(
                f"[verl-omni-ext] env injection: {len(injected)} vars added to "
                f"runtime_env.env_vars: {injected}",
                flush=True,
            )

        return runtime_env

    return _patched_get_ppo_ray_runtime_env


# 模块 import 时显式调用，执行 patch
_patch_get_ppo_ray_runtime_env()


# ============================================================================
# L2 patch：offload_fsdp2_model_to_cpu 不需要 patch
# ============================================================================
# 代码1 无条件安装 patch_audio_encoder_fsdp_unshard → FSDP2 nn.Embedding
# weight 在 forward 后被 unshard（变回普通 Tensor）→ model.cpu() 成功。
#
# 代码2 之前 precompute 激活时跳过了 unshard hook → FSDP2 nn.Embedding
# weight 一直是 DTensor → model.cpu() 的 FSDP2 _apply() swap 失败。
#
# 修复（log 20）：forward.py 无条件安装 unshard hook（与代码1 一致）。
# 因此 offload_fsdp2_model_to_cpu 不需要 patch——原始 model.cpu() 能工作。