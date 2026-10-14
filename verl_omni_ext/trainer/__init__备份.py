"""训练范式注册目录 + main_omni env 注入 patch。

L2 patch：把 MINICPMO_*/VLLM_*/GENRM_*/HCCL_* 等环境变量注入
``config.ray_kwargs.ray_init.runtime_env.env_vars``，让 Ray worker 能收到
launch script export 的变量。

原因：verl 的 ``get_ppo_ray_runtime_env()`` 白名单不含 verl-omni 专用变量
（``MINICPMO_EMBED_CACHE_DIR`` 等），launch script ``export`` 的变量不进 Ray
worker。代码1 在 ``main_omni.py`` 内直接实现了 ``_inject_omni_env_vars``，
代码2 作为 out-of-tree 插件仓不能改上游源码，改用 ``_patchkit`` patch
``verl_omni.trainer.main_omni.run_omni``。

移植自代码1 ``verl_omni/trainer/main_omni.py:348-402`` 的
``_inject_omni_env_vars``，改为 ``_patchkit.idempotent_patch`` 封装的 L2 patch。

See memory: feedback_ray_env_vars_propagation.md
"""
from verl_omni_ext._patchkit import idempotent_patch
print("[verl-omni-ext] trainer/__init__.py is being imported", flush=True)

# ============================================================================
# L2 patch：main_omni.run_omni env 注入
# ============================================================================
@idempotent_patch(
    name="inject_omni_env_vars",
    target_module="verl_omni.trainer.main_omni",
    target_attr="run_omni",
    fingerprint="verl_omni>=0.2.0",
    expected_signature=None,
    probe_signature=True,
)
def _patch_run_omni_env_injection(original_run_omni, __patch_signature__=None):
    """Patch run_omni to inject MINICPMO_*/VLLM_*/GENRM_*/HCCL_* env into Ray runtime_env.

    原因：verl 的 ``get_ppo_ray_runtime_env()`` 白名单不含 verl-omni 专用变量
    （``MINICPMO_EMBED_CACHE_DIR`` / ``GENRM_*`` / ``HCCL_*`` 等），launch script
    ``export`` 的变量不进 Ray worker。

    方案：在 ``run_omni(config)`` 调用 original 之前，把 env 注入
    ``config.ray_kwargs.ray_init.runtime_env.env_vars``。verl 的 run_ppo 和
    verl-omni 的 launch_ray_task_runner 都在 ray.init() 前读这个字段。

    注意：
    - OmegaConf struct mode 拒绝 ``setdefault`` 创建不存在的 key，需要临时关 struct
    - 不用 ``DictConfig.setdefault``（返回 plain dict，写入丢失），用直接属性赋值
      (see memory: feedback_omegaconf_setdefault_loses_writes.md)
    """
    import os
    from omegaconf import OmegaConf as _OC

    _ENV_KEYS = (
        # 预计算
        "MINICPMO_EMBED_CACHE_DIR",
        "MINICPMO_MM_TRACE",
        "MINICPMO_MM_TRACE_FIRST_N",
        "MINICPMO_ENABLE_MM_FORWARD",
        "MINICPMO_ALLOW_PACKED_MM_FORWARD",
        "MINICPMO_VIDEO_MAX_FRAMES",
        "MINICPMO_VIDEO_FPS",
        "MINICPMO_MAX_SLICE_NUMS",
        # hang probe
        "MINICPMO_HANG_PROBE",
        "MINICPMO_HANG_TIMEOUT_SEC",
        "MINICPMO_HANG_ABORT_AFTER",
        "MINICPMO_HANG_ABORT_GRACE_SEC",
        "MINICPMO_HANG_DUMP_DIR",
        # FSDP2
        "MINICPMO_FSDP2_PIN_MEMORY",
        # GenRM
        "GENRM_TEMPERATURE",
        "GENRM_TOP_P",
        "GENRM_MAX_TOKENS",
        "GENRM_ENABLE_THINKING",
        "GENRM_STRIP_THINK",
        "GENRM_MAX_RETRIES",
        "GENRM_TIMEOUT",
        "GENRM_ENDPOINT",
        "GENRM_FALLBACK_ON_ERROR",
        "GENRM_LOG_RESPONSE",
        "GENRM_RESPONSE_MAX_CHARS",
        "GENRM_MODEL_NAME",
        "GENRM_SEED",
        # HCCL timeout
        "HCCL_CONNECT_TIMEOUT",
        "HCCL_EXEC_TIMEOUT",
        "ACL_DEVICE_SYNC_TIMEOUT",
        "TORCH_DIST_TIMEOUT",
    )

    def _patched_run_omni(config, task_runner_class=None):
        # OmegaConf struct mode rejects setdefault for keys not already present
        # in the config schema. Temporarily disable struct mode to create them.
        # Use direct attribute/index assignment instead of setdefault —
        # DictConfig.setdefault returns a plain Python dict for missing keys,
        # which is NOT managed by OmegaConf, so writes to it are silently lost.
        _was_struct = _OC.is_struct(config)
        _OC.set_struct(config, False)
        try:
            if "ray_kwargs" not in config:
                config.ray_kwargs = {}
            if "ray_init" not in config.ray_kwargs:
                config.ray_kwargs.ray_init = {}
            ray_init = config.ray_kwargs.ray_init
            if "runtime_env" not in ray_init:
                ray_init.runtime_env = {}
            runtime_env = ray_init.runtime_env
            if "env_vars" not in runtime_env:
                runtime_env.env_vars = {}
            env_vars = runtime_env.env_vars

            injected = []
            for key in _ENV_KEYS:
                val = os.environ.get(key)
                if val is not None:
                    env_vars[key] = val
                    injected.append(f"{key}={val}")
        finally:
            _OC.set_struct(config, _was_struct)

        if injected:
            print(
                f"[verl-omni-ext] env injection: injected {len(injected)} vars into "
                f"config.ray_kwargs.ray_init.runtime_env.env_vars: {injected}",
                flush=True,
            )

        return original_run_omni(config, task_runner_class)

    return _patched_run_omni


# 模块 import 时显式调用，执行 patch
_patch_run_omni_env_injection()