#!/usr/bin/env bash
# Run on EVERY node. VC_TASK_INDEX=0 starts training; others only join Ray.
# Legacy modes use two nodes; STEM supports 8/16 nodes through its job wrapper.
# Deploy to ${PROJECT_ROOT}/examples/qwen3_6_huanyu/ with the Huanyu package.
# Defaults: frozen Huanyu, 35B sampleall, precomputed q128 cache, actor16 + RM16.
# HUANYU_JOB_MODE=aqa_unfreeze selects online AQA training via its own entry.
# AQA length and gradient-sync settings are supplied by environment/Hydra args.
# Precision defaults: the complete full policy, including FP32 training/GDN,
# prefill/gating/conv/L2/barrier, private hybrid cache, stable linear, layerwise
# IPC reload, logical KV zeroing, isolated media and FSDP pin_memory=0.
# Apply the same policy before Ray on BOTH nodes and explicitly in runtime_env
# for training. full requires graph mode=0; explicit profile=custom permits A/B.
# Uses the selected vLLM-Omni source directly; never copies/replaces checkouts.
# Never stops an existing Ray cluster.
set -euo pipefail

# BEGIN HUANYU audio policy
huanyu_resolve_audio_train_mode() {
    local legacy="${HUANYU_FREEZE_AUDIO-false}" default_train key value
    case "$legacy" in
        true) default_train=false ;;
        false) default_train=true ;;
        *) echo '[huanyu-audio-mode] FATAL: HUANYU_FREEZE_AUDIO must be true or false' >&2; return 2 ;;
    esac
    for key in HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR; do
        value="${!key-$default_train}"
        case "$value" in
            true|false) export "$key=$value" ;;
            *) echo "[huanyu-audio-mode] FATAL: $key must be true or false" >&2; return 2 ;;
        esac
    done
    if [[ "$HUANYU_TRAIN_AUDIO_ENCODER" == true || "$HUANYU_TRAIN_AUDIO_PROJECTOR" == true ]]; then
        export HUANYU_FREEZE_AUDIO=false MINICPMO_TUNE_AUDIO=1
    else
        export HUANYU_FREEZE_AUDIO=true MINICPMO_TUNE_AUDIO=0
    fi
}
# END HUANYU audio policy
# BEGIN HUANYU graph policy
huanyu_configure_graph_mode() {
    if [[ "${HUANYU_PRECISION_PROFILE:-full}" == full && ( "${HUANYU_GRAPH_MODE:-0}" == 1 || "${HUANYU_GRAPH_MODE:-0}" == true ) ]]; then
        echo "[huanyu-graph] FATAL: HUANYU_PRECISION_PROFILE=full cannot capture Python precision reference kernels; use HUANYU_GRAPH_MODE=0, or explicitly select profile=custom for a separate comparison" >&2
        return 2
    fi
    case "${HUANYU_GRAPH_MODE:-0}" in
        1|true)
            export HUANYU_GRAPH_MODE=1 HUANYU_ENFORCE_EAGER=false
            export VLLM_TORCH_COMPILE=1
            export TORCH_COMPILE_DISABLE=0 NPU_GRAPH_CAPTURE_DISABLE=0
            ;;
        0|false)
            export HUANYU_GRAPH_MODE=0 HUANYU_ENFORCE_EAGER=true
            export VLLM_TORCH_COMPILE=0
            export TORCH_COMPILE_DISABLE=1 NPU_GRAPH_CAPTURE_DISABLE=1
            ;;
        *)
            echo "[huanyu-graph] FATAL: HUANYU_GRAPH_MODE must be 0/1 or false/true" >&2
            return 2
            ;;
    esac

    # Explicit enable/disable values also override flags inherited from an
    # already-running Raylet. Merely unsetting the driver's environment cannot
    # remove those worker-side flags. Keep values as Hydra-quoted strings.
    HUANYU_GRAPH_ARGS=(
        "actor_rollout_ref.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER}"
        "reward.reward_model.rollout.enforce_eager=${HUANYU_ENFORCE_EAGER}"
        "++ray_kwargs.ray_init.runtime_env.env_vars.HUANYU_GRAPH_MODE=\"${HUANYU_GRAPH_MODE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_TORCH_COMPILE=\"${VLLM_TORCH_COMPILE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.TORCH_COMPILE_DISABLE=\"${TORCH_COMPILE_DISABLE}\""
        "++ray_kwargs.ray_init.runtime_env.env_vars.NPU_GRAPH_CAPTURE_DISABLE=\"${NPU_GRAPH_CAPTURE_DISABLE}\""
    )
}
# END HUANYU graph policy
# BEGIN HUANYU precision policy
huanyu_configure_precision_mode() {
    local _key _value _arg _tp="${ROLLOUT_TP:-4}"
    local _profile="${HUANYU_PRECISION_PROFILE:-full}"
    case "$_profile" in
        full|custom) ;;
        *) echo "[huanyu-precision] FATAL: HUANYU_PRECISION_PROFILE must be full or custom" >&2; return 2 ;;
    esac
    if [[ "$_profile" == full && ( "${HUANYU_GRAPH_MODE:-0}" == 1 || "${HUANYU_GRAPH_MODE:-0}" == true ) ]]; then
        echo "[huanyu-precision] FATAL: profile=full uses Python precision reference kernels and cannot capture inference graphs; set HUANYU_GRAPH_MODE=0, or explicitly select HUANYU_PRECISION_PROFILE=custom for a separate comparison" >&2
        return 2
    fi
    export HUANYU_PRECISION_PROFILE="$_profile"
    local -a _changed=()
    local -a _on=(
        MINICPMO_HUANYU_GDN_FP32_TRAINING MINICPMO_GDN_FP32_REFERENCE
        MINICPMO_HUANYU_LAYERWISE_RELOAD MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE
        MINICPMO_ISOLATE_MEDIA_ENCODERS
    )
    local -a _full=(
        MINICPMO_GDN_CACHE_BARRIER MINICPMO_GDN_PREFILL_REFERENCE
        MINICPMO_GDN_L2NORM_REFERENCE MINICPMO_GDN_CONV_REFERENCE
        MINICPMO_GDN_GATING_REFERENCE MINICPMO_PRIVATE_HYBRID_CACHE
        MINICPMO_STABLE_LINEAR
    )
    local -a _off=(
        MINICPMO_HF_NORM_REFERENCE
        MINICPMO_HF_ROUTER_REFERENCE MINICPMO_HF_TEXT_ROPE_REFERENCE
        MINICPMO_MOE_MATCH_ROLLOUT MINICPMO_ISOLATE_FULL_ATTENTION
        MINICPMO_VISION_POSITION_REFERENCE MINICPMO_NATIVE_VISION_REFERENCE
    )
    for _key in "${_on[@]}" "${_full[@]}" "${_off[@]}"; do
        if [[ "$_profile" == full && " ${_on[*]} ${_full[*]} " == *" $_key "* ]]; then
            if [[ -v "$_key" && "${!_key}" != 1 ]]; then
                _changed+=("$_key=${!_key}->1")
            fi
            printf -v "$_key" '%s' 1
        elif [[ ! -v "$_key" ]]; then
            case " ${_on[*]} " in
                *" $_key "*) printf -v "$_key" '%s' 1 ;;
                *) printf -v "$_key" '%s' 0 ;;
            esac
        fi
        case "${!_key}" in
            0|1) export "$_key" ;;
            *) echo "[huanyu-precision] FATAL: $_key must be 0 or 1" >&2; return 2 ;;
        esac
    done
    if [[ "$_profile" == full ]]; then
        if [[ -v MINICPMO_FSDP2_PIN_MEMORY && "$MINICPMO_FSDP2_PIN_MEMORY" != 0 ]]; then
            _changed+=("MINICPMO_FSDP2_PIN_MEMORY=$MINICPMO_FSDP2_PIN_MEMORY->0")
        fi
        export MINICPMO_FSDP2_PIN_MEMORY=0
    else
        export MINICPMO_FSDP2_PIN_MEMORY="${MINICPMO_FSDP2_PIN_MEMORY:-0}"
    fi
    case "$MINICPMO_FSDP2_PIN_MEMORY" in
        0|1) ;;
        *) echo "[huanyu-precision] FATAL: MINICPMO_FSDP2_PIN_MEMORY must be 0 or 1" >&2; return 2 ;;
    esac
    # Hydra applies the last CLI override. Use that exact TP in both preflight
    # arithmetic validation and the worker environment, not a stale shell TP.
    for _arg in "$@"; do
        _key="${_arg%%=*}"
        while [[ "$_key" == +* ]]; do _key="${_key#+}"; done
        if [[ "$_key" == actor_rollout_ref.rollout.tensor_model_parallel_size ]]; then
            _tp="${_arg#*=}"
            _tp="${_tp#\"}"; _tp="${_tp%\"}"
            _tp="${_tp#\'}"; _tp="${_tp%\'}"
        fi
    done
    if [[ ! "$_tp" =~ ^[1-9][0-9]*$ ]]; then
        echo "[huanyu-precision] FATAL: rollout tensor_model_parallel_size must be a positive integer" >&2
        return 2
    fi
    export ROLLOUT_TP="$_tp" MINICPMO_HUANYU_ROLLOUT_TP="$_tp"
    for _key in MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE; do
        _value="${!_key:-0}"
        case "$_value" in
            0) ;;
            4)
                if [[ "$_tp" != 4 ]]; then
                    echo "[huanyu-precision] FATAL: $_key=4 requires actual rollout TP=4, got $_tp" >&2
                    return 2
                fi ;;
            *) echo "[huanyu-precision] FATAL: $_key supports only 0 or 4" >&2; return 2 ;;
        esac
        export "$_key=$_value"
    done
    export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_6_huanyu.engine_hooks
    export HUANYU_USE_PRECOMPUTE="${HUANYU_USE_PRECOMPUTE:-0}"
    case "$HUANYU_USE_PRECOMPUTE" in
        0|1) ;;
        *) echo "[huanyu-precision] FATAL: HUANYU_USE_PRECOMPUTE must be 0 or 1" >&2; return 2 ;;
    esac
    # The two-node job snapshots these defaults/caller overrides across the
    # site env file. Cache mode, TP and HCCL are resolved later by the launcher.
    HUANYU_PRECISION_KEYS=(HUANYU_PRECISION_PROFILE "${_on[@]}" "${_full[@]}" "${_off[@]}" \
        MINICPMO_FSDP2_PIN_MEMORY MINICPMO_MOE_TP_REFERENCE MINICPMO_ROW_TP_REFERENCE)
    # Opt-in job policy is snapshotted across site setup and explicitly sent to
    # fresh Ray workers even when their raylet was started with other defaults.
    export HUANYU_JOB_DETERMINISM="${HUANYU_JOB_DETERMINISM:-0}"
    case "$HUANYU_JOB_DETERMINISM" in
        0|1) ;;
        *) echo '[huanyu-precision] FATAL: HUANYU_JOB_DETERMINISM must be 0 or 1' >&2; return 2 ;;
    esac
    HUANYU_PRECISION_KEYS+=(HUANYU_JOB_DETERMINISM)
    # Preserve optional timing through site setup and existing Ray clusters.
    export HUANYU_GEN_TIMING="${HUANYU_GEN_TIMING:-0}"
    export HUANYU_GEN_TIMING_ENGINE_EVERY="${HUANYU_GEN_TIMING_ENGINE_EVERY:-100}"
    case "$HUANYU_GEN_TIMING" in
        0|1) ;;
        *) echo '[huanyu-timing] FATAL: HUANYU_GEN_TIMING must be 0 or 1' >&2; return 2 ;;
    esac
    if [[ ! "$HUANYU_GEN_TIMING_ENGINE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
        echo '[huanyu-timing] FATAL: HUANYU_GEN_TIMING_ENGINE_EVERY must be a positive integer' >&2
        return 2
    fi
    HUANYU_PRECISION_KEYS+=(HUANYU_GEN_TIMING HUANYU_GEN_TIMING_ENGINE_EVERY)
    if [[ "$HUANYU_JOB_DETERMINISM" == 1 ]]; then
        export HCCL_DETERMINISTIC=strict ASCEND_LAUNCH_BLOCKING=1 NCCL_DETERMINISTIC=1
        export LCCL_DETERMINISTIC=1 VLLM_BATCH_INVARIANT=0
        HUANYU_PRECISION_KEYS+=(ASCEND_LAUNCH_BLOCKING NCCL_DETERMINISTIC LCCL_DETERMINISTIC VLLM_BATCH_INVARIANT)
    fi
    HUANYU_PRECISION_ARGS=()
    if [[ "$HUANYU_JOB_DETERMINISM" == 1 ]]; then
        # verl's full_determinism option also enables batch-invariant kernels.
        # This job intentionally selects the independent worker policy instead.
        HUANYU_PRECISION_ARGS+=(
            'actor_rollout_ref.rollout.full_determinism=false'
            '++reward.reward_model.rollout.full_determinism=false'
            '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enabled=true'
            '++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.rl_config.enable_batch_invariant=false'
        )
    fi
    # Keep the outer actor engine kwargs consistent with stage-0 engine_extras:
    # native deploy merging must not restore the generic colocated extension.
    # This applies to the actor only; the reward model keeps its own extension.
    HUANYU_PRECISION_ARGS+=(
        "++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.worker_extension_cls=verl_omni_ext.models.qwen3_6_huanyu.worker_extension.HuanyuColocateWorkerExtension"
    )
    for _key in "${HUANYU_PRECISION_KEYS[@]}" \
        MINICPMO_HUANYU_ROLLOUT_TP VLLM_OMNI_EXTERNAL_MODULES HUANYU_USE_PRECOMPUTE \
        HCCL_OP_EXPANSION_MODE HCCL_DETERMINISTIC; do
        _value="${!_key:-}"
        # Do not introduce an empty HCCL setting: native CANN may distinguish
        # an empty value from an unset variable. Forward only a defined policy.
        if [[ "$_key" == HCCL_* && -z "$_value" ]]; then continue; fi
        # Hydra quoted strings preserve the driver's actual numerical policy
        # when reusing an already-running Raylet.
        _value="${_value//\\/\\\\}"
        _value="${_value//\"/\\\"}"
        HUANYU_PRECISION_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=\"${_value}\"")
    done
    if (( ${#_changed[@]} )); then
        echo "[huanyu-precision] profile=full overrides inherited values: ${_changed[*]}"
    fi
    # Repeated source calls in one shell revalidate/rebuild argv but only repeat
    # the detailed report if its actual values changed. Child launchers report
    # independently, and the worker Python installers report active execution.
    local _report="profile=$_profile actual_TP=$_tp cache=$HUANYU_USE_PRECOMPUTE graph=${HUANYU_GRAPH_MODE:-0}"
    for _key in "${HUANYU_PRECISION_KEYS[@]}"; do _report+=" $_key=${!_key}"; done
    if [[ "${_HUANYU_PRECISION_LAST_REPORT:-}" != "$_report" ]]; then
        echo "[huanyu-precision] profile=$_profile actual_TP=$_tp cache=$HUANYU_USE_PRECOMPUTE graph=${HUANYU_GRAPH_MODE:-0} hooks=$VLLM_OMNI_EXTERNAL_MODULES"
        for _key in "${HUANYU_PRECISION_KEYS[@]}"; do
            echo "[huanyu-precision] effective $_key=${!_key}"
        done
        _HUANYU_PRECISION_LAST_REPORT="$_report"
    fi
}
# END HUANYU precision policy
# BEGIN HUANYU native policy
huanyu_select_native_source() {
    local root="${HUANYU_VLLM_SOURCE:-${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}}"
    local explicit="${1:-}" old_source="${HUANYU_VLLM_SOURCE:-}" old_root="${HUANYU_VLLM_OMNI_ROOT:-}"
    local entry retained="" required
    local -a entries=()
    # Retire source choices inherited from old prepare/activate.sh files.
    if [[ "$explicit" != --explicit-source ]]; then
        case "${root%/}" in
            */vllm-omni-xy|*/vllm-omni-xy-*) root=/vllm-workspace/vllm-omni ;;
        esac
    fi
    if [[ "$root" != /* || "$root" == *:* || "$root" == *$'\n'* ]]; then
        echo 'FATAL: HUANYU_VLLM_SOURCE must be an absolute directory without colons/newlines' >&2
        return 2
    fi
    for required in vllm_omni/__init__.py \
        vllm_omni/model_executor/models/minicpmo_5_0/huanyu_audio.py \
        vllm_omni/model_executor/models/minicpmo_5_0/minicpmo_5_0_omni_llm.py; do
        [[ -r "$root/$required" ]] || {
            echo "FATAL: Huanyu-enabled source is incomplete: $root/$required" >&2
            return 1
        }
    done
    root="$(cd -- "$root" && pwd -P)" || return $?
    IFS=: read -r -a entries <<< "${PYTHONPATH:-}"
    for entry in "${entries[@]}"; do
        case "${entry%/}" in
            ""|"$root"|"$old_source"|"$old_root"|\
            /vllm-workspace/vllm-omni|/vllm-workspace/vllm-omni/*|\
            */vllm-omni-xy|*/vllm-omni-xy/*|*/vllm-omni-xy-*|*/vllm-omni-xy-*/*) continue ;;
        esac
        # A differently named checkout must not shadow the installed package.
        [[ -f "$entry/vllm_omni/__init__.py" ]] && continue
        retained+="${retained:+:}$entry"
    done
    # These are expected-origin checks, not import-path overrides.
    export HUANYU_VLLM_SOURCE="$root" HUANYU_VLLM_OMNI_ROOT="$root"
    if [[ "$explicit" == --explicit-source ]]; then
        export PYTHONPATH="$root${retained:+:$retained}"
    elif [[ -n "$retained" ]]; then
        export PYTHONPATH="$retained"
    else
        unset PYTHONPATH
    fi
}
# END HUANYU native policy
# BEGIN HUANYU persisted-native policy
huanyu_clear_persisted_native_source() {
    local profile marker backup
    # Do not write new shell-startup overrides. Disable only our own old file;
    # the guarded line already in .bashrc becomes inert. Keep a recoverable copy.
    profile="${HOME:?}/.huanyu-vllm-source.sh"
    if [[ -f "$profile" && ! -L "$profile" ]]; then
        umask 077
        exec 8>"$HOME/.huanyu-vllm-source.lock"
        flock -x 8
        if [[ -f "$profile" && ! -L "$profile" ]] && \
            IFS= read -r marker < "$profile" && [[ "$marker" == '# Generated by native_source.sh --persist' ]]; then
            backup="$(mktemp -- "$profile.disabled.XXXXXX")"
            mv -fT -- "$profile" "$backup"
            printf '[huanyu-env] disabled old source override; backup=%s\n' "$backup"
        fi
    fi
}
# END HUANYU persisted-native policy
# BEGIN HUANYU environment policy
huanyu_prepare_environment() {
set -ex
huanyu_configure_precision_mode "$@" || {
    return 2 2>/dev/null || exit 2
}

# Huanyu 环境；在每个节点 ray start 前 source 本文件。
# source 前 export PROJECT_ROOT 可覆盖部署目录；默认使用 infra_data 下的新项目。
# ext 从 PROJECT_ROOT 加载，verl / verl-omni 继续使用公共安装。
export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
# 插件由 PROJECT_ROOT 选择；vLLM-Omni 使用已有安装，源码变量仅用于核对导入位置。
export QWEN36_HUANYU_ROOT="${PROJECT_ROOT}"
huanyu_select_native_source || {
    return 1 2>/dev/null || exit 1
}
unset QWEN36_GZQ_ROOT

# PYTHONPATH 加入 Python 包的父目录，兼容直接布局及外层插件仓库目录。
_huanyu_ext_python_root=""
for _huanyu_candidate in \
    "${PROJECT_ROOT}" \
    "${PROJECT_ROOT}/verl-omni_ext" \
    "${PROJECT_ROOT}/verl-omni-ext" \
    "${PROJECT_ROOT}/verl_omni_ext"; do
    if [ -f "${_huanyu_candidate}/verl_omni_ext/__init__.py" ]; then
        _huanyu_ext_python_root="${_huanyu_candidate}"
        break
    fi
done
if [ -z "${_huanyu_ext_python_root}" ]; then
    echo "ERROR: ${PROJECT_ROOT} 下未找到可导入的 verl_omni_ext/__init__.py。" >&2
    return 1 2>/dev/null || exit 1
fi
if [ ! -f "${HUANYU_VLLM_OMNI_ROOT}/vllm_omni/__init__.py" ]; then
    echo "ERROR: ${HUANYU_VLLM_OMNI_ROOT}/vllm_omni/__init__.py 不存在；不能回退到其他 vLLM-Omni 版本。" >&2
    return 1 2>/dev/null || exit 1
fi

# source 同时切换工作目录：Python 会优先搜索 cwd，不能停留在另一个副本。
# 之后可直接使用相同的相对路径训练命令，无需额外手动 cd。
if ! cd -- "${PROJECT_ROOT}"; then
    echo "ERROR: 无法进入 PROJECT_ROOT=${PROJECT_ROOT}" >&2
    return 1 2>/dev/null || exit 1
fi

# 清除旧 ext 根、嵌套目录、路径别名，以及旧 vLLM-Omni 路径。
# 随后仅把本次选定的包父目录放回首部；保留 CANN 和公共依赖路径。
_huanyu_retained_pythonpath=""
IFS=: read -r -a _huanyu_python_entries <<< "${PYTHONPATH:-}"
for _huanyu_path in "${_huanyu_python_entries[@]}"; do
    case "${_huanyu_path%/}" in
        ""|"${PROJECT_ROOT%/}"|"${PROJECT_ROOT%/}"/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq_new|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq_new/*|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36/*|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36_gzq|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36_gzq/*|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb_data/code/hufenyu/vllm-omni-xy/*|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy|\
        /opt/huawei/dataset/ccb/data/code/hufenyu/vllm-omni-xy/*|\
        /vllm-workspace/vllm-omni|\
        /vllm-workspace/vllm-omni/*|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/hfy_guiyang/vllm-omni-xy-qwen-vit/*|\
        /opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit|\
        /opt/huawei/dataset/infra_data/code/vllm-omni-xy-qwen-vit/*)
            continue ;;
    esac
    _huanyu_retained_pythonpath="${_huanyu_retained_pythonpath:+${_huanyu_retained_pythonpath}:}${_huanyu_path}"
done
export PYTHONPATH="${_huanyu_ext_python_root}:/opt/huawei/algorithm/:/opt/huawei/dataset/ccb_data/code/package_install/verl${_huanyu_retained_pythonpath:+:${_huanyu_retained_pythonpath}}"
huanyu_select_native_source || { return 1 2>/dev/null || exit 1; }
echo "[huanyu env] ext Python root: ${_huanyu_ext_python_root}"
echo "[huanyu env] expected installed vLLM-Omni: ${HUANYU_VLLM_OMNI_ROOT} (no source PYTHONPATH injection)"
unset _huanyu_candidate _huanyu_ext_python_root
unset _huanyu_retained_pythonpath _huanyu_python_entries _huanyu_path

# 覆盖旧环境残留的输出路径；如需自定义，在 source 本文件之后再覆盖。
# 基础启动脚本会读取这两个变量，确保输出写入本次主目录。
export DEFAULT_LOCAL_DIR="${PROJECT_ROOT}/checkpoints"
export TENSORBOARD_DIR="${PROJECT_ROOT}/tensorboard_log"
# 两个代码目录之间切换时，迁移继承的诊断目录并保留其相对路径。
# 其他用户指定的独立诊断目录保持不变。
case "${MINICPMO_ROLLOUT_DIAG_DIR:-}" in
    /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq_new/*)
        export MINICPMO_ROLLOUT_DIAG_DIR="${PROJECT_ROOT}/${MINICPMO_ROLLOUT_DIAG_DIR#*/verl_omni_qwen36_gzq_new/}" ;;
    /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36_gzq/*|\
    /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36_gzq/*)
        export MINICPMO_ROLLOUT_DIAG_DIR="${PROJECT_ROOT}/${MINICPMO_ROLLOUT_DIAG_DIR#*/verl_omni_qwen36_gzq/}" ;;
    /opt/huawei/dataset/ccb_data/code/hufenyu/verl_omni_qwen36/*|\
    /opt/huawei/dataset/ccb/data/code/hufenyu/verl_omni_qwen36/*)
        export MINICPMO_ROLLOUT_DIAG_DIR="${PROJECT_ROOT}/${MINICPMO_ROLLOUT_DIAG_DIR#*/verl_omni_qwen36/}" ;;
esac

pip install mathruler
pip install zarr

export TORCH_EXTENSIONS_DIR=/cache/.torch_extension
export NPUS_PER_NODE=16
export MODEL_PATH="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
export RM_MODEL_PATH="${RM_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B}"
export TRAIN_FILE="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/train.parquet}"
export VAL_FILE="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/validation.parquet}"
export Q36_CKPT_DIR="${MODEL_PATH}"
echo "[huanyu-paths] project=$PROJECT_ROOT actor=$MODEL_PATH reward=$RM_MODEL_PATH"
echo "[huanyu-paths] train=$TRAIN_FILE validation=$VAL_FILE"

# 这里**曾经**有一行 `export MINICPMO_DISABLE_IMAGE_EXPAND=1`（2026-10-10 调试残留）。
# 它的位置决定了它的危险：本文件在 `ray start` **之前**被 source，于是每个 Ray worker
# 都把它当默认值，而 driver / run 脚本里看到的是 `<unset>` —— 谁跑训练都看不见它，
# 跑出来的却全是「修复前形态」。那一轮（10-10 21:19）的代价是 12/12 条图像 rollout
# 全在 _assert_bounds_agree 抛 bound mismatch，图像样本直接对不上位。
#
# 现在这层保护在包内：worker_env.reconcile_worker_env() 在插件 import 时把 worker 侧的
# `=1` 撤销（并打印一行说明）。真要切回旧形态，必须再显式加 ACK，两个开关同时存在：
#
# export MINICPMO_LEGACY_IMAGE_SPAN_ACK=1     # 只有它也在 worker env 里，上面那个才被尊重
export MINICPMO_DISABLE_IMAGE_EXPAND=0

MASTER_ADDR="${VC_WORKER_HOSTS:-${MASTER_ADDR:-}}"
MASTER_ADDR="${MASTER_ADDR%%,*}"
if [[ -n "$MASTER_ADDR" ]]; then
    MASTER_ADDR=$(getent hosts "$MASTER_ADDR" | awk '{print $1; exit}')
fi
RAY_PORT=6925
RAY_DASHBOARD_PORT=8960

############################## COMMON FUN ##########################

rm_exist_dir(){
    if [ -d "$1" ] ; then
      rm -rf "$1"
    fi
}

print_stage(){
  echo "###################  $1  #############################"
}

print_stage "ENV PRINT"
echo "$@"

#############################  PRE CHECK ##########################
print_stage "PRE CHECK"
WORKER_RANK=${VC_TASK_INDEX:-0}
DEPLOY_LOG_PATH="${PROJECT_ROOT}/logs/$WORKER_RANK"
MASTER_INFO_FILE="${PROJECT_ROOT}/logs/0/master_info.txt"
WORKER_LOG_PATH="$DEPLOY_LOG_PATH/worker_log_$WORKER_RANK"

mkdir -p "$DEPLOY_LOG_PATH"

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni_ext
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=1
export USE_OPTIMIZED_MODEL=0
# One switch for eager/graph inference; the training backend reapplies it and
# explicitly forwards it to Ray, so later per-run changes also reach workers.
huanyu_configure_graph_mode || {
    return 2 2>/dev/null || exit 2
}
export WANDB_MODE=${WANDB_MODE:-offline}
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NPUS_PER_NODE}
export VLLM_NO_SUBCLASS_INSPECT=1

ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
fi
if [ -f "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/../nnal/atb/set_env.sh"
fi

export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

# =============================================================================
# 必须在 ray start 之前设的 env（见下面为什么）
# =============================================================================
# WorkerDict / vLLMOmniHttpServer 都是 Ray actor，env 继承自 raylet，而 raylet
# 是下面那两条 `ray start` fork 出来的。所以 `ray start` 之后再 export 的东西
# 只活在 driver 进程里，训练 rank 上根本看不到。
#
# verl 确实有 runtime_env 透传，但那是 verl/trainer/constants_ppo.py 里一份
# 写死的白名单（TOKENIZERS_PARALLELISM / NCCL_DEBUG / VLLM_* / VERL_* ...），
# 一个 HCCL_* 都不在里面。log 25 打印的 runtime_env 可以逐字核对。
#
# 这直接坑掉了基础脚本 §2 的整段 HCCL 配置 —— 它在 line 105-116 export，
# 而基础脚本是 line 101 之后才被调用的。后果（log 25 实测）：
#   * HCCL_CONNECT_TIMEOUT 一直是默认 120s，不是 7200。log 20 那个"静默 174 秒
#     后 EI0006"就是 120s 默认超时，不是什么 adapter_hccp.cc 硬编码。
#   * HCCL_SOCKET_IFNAME 从未生效，HCCL 一直在自己猜网卡。
#   * MINICPMO_MM_TRACE / TORCH_DISTRIBUTED_DEBUG 同理，两轮诊断的观测都是瞎的。
#
# 反例：MINICPMO_ENABLE_MM_FORWARD=1 一直好使，因为它本来就在 ray start 之前。
#
# 新增变量往这里加，不要往下面的实验脚本里加。
# -----------------------------------------------------------------------------

# 通信超时。EI0006 的直接原因：多模态视觉负载跨卡差 7 倍（slices=7 vs 1），
# 快的卡先到 all_reduce，在 120s 内等不到慢的卡就判定建链失败。
#
# CONNECT 保持 7200：上面那条 EI0006 就是它太小造成的，别动。
# 其余三个从 7200/3600 降到分钟级 —— 它们不是"容错余量"，是"挂了多久才报错"。
# log 29 在 14:28 卡死、16:04 被人工 kill，95 分钟里没有任何一层抛异常，就是因为
# 7200s 一直没到点；「没报错」当时被我们误读成「没有未完成的通信」。
# 取值依据：plog 实测算子之间最长的合法间隔是 10.1s（反传计算），300s 有 30 倍余量。
#
# EXEC 从 600 降到 300 是 log 31 的教训：那一轮 18:16:53 十五张卡进了 AICPU 的
# AllGather kernel，EXEC_TIMEOUT 本该在 18:26:53 点名是哪次集合通信卡住，但 rank 0
# 的 hang probe 在 18:19:05 就把进程杀了，Ascend 侧那份报告从来没机会打出来。
# 现在的排期是：120s/240s 出 Python 栈（全卡，见下面的 cross-rank trigger），
# 300s 出 HCCL 的算子级报错，480+45s 才自杀 —— 三份证据都拿到再退出。
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-300}
export ACL_DEVICE_SYNC_TIMEOUT=${ACL_DEVICE_SYNC_TIMEOUT:-900}
export TORCH_DIST_TIMEOUT=${TORCH_DIST_TIMEOUT:-600}

# 网卡名。不设的话 HCCL/GLOO 自己挑，多网卡机器上可能挑错。
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export HCCL_SOCKET_IFNAME=${SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}
export TP_SOCKET_IFNAME=${SOCKET_IFNAME}

# 注意：**故意不设** HCCL_IF_BASE_PORT=64000（基础脚本 line 107 有）。
# 它和上面 HCCL_*_SOCKET_PORT_RANGE="auto" 是两套互斥的端口分配机制，而 auto
# 这条路是目前唯一在 32 卡上真正建起链的（同节点 46/46 成功）。要换先单独验。

# NPU 运行时调优，取自基础脚本 §1，同样因为时序问题从未到达训练 rank
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-2}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export ASCEND_SECURITY_CHECK_DISABLED=${ASCEND_SECURITY_CHECK_DISABLED:-1}
export ASCEND_GLOBAL_LOG_LEVEL=${ASCEND_GLOBAL_LOG_LEVEL:-3}
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

# 基础脚本 line 101 的 `unset` 同样只作用于 driver，这里补一次。
# max_split_size_mb:128 会把分配器切碎，而 MoE 专家权重是大块张量。
unset PYTORCH_NPU_ALLOC_CONF

# 诊断开关。排障期间默认打开，跑通之后再关回去：
#   MINICPMO_MM_TRACE=1        每次多模态前向 / 每次 apm unshard 都打一行
#   MINICPMO_MM_TRACE_FIRST_N  即使 MM_TRACE=0 也无条件打头 N 次（默认 8）。
#                              旧代码是 `call_no == 1`，而 compute_log_prob 每卡
#                              恰好 2 个 micro-batch —— 每次都挂在第 2 轮，而第 2
#                              轮从来没有埋点。log 24 那个"32 卡全是 n=1，所以
#                              通信次数对齐"的结论就是这么来的，是门控的假象。
#   TORCH_DISTRIBUTED_DEBUG    DETAIL 用 gloo 校验每次集合通信的跨 rank 一致性。
#                              它本身会拖慢，只在怀疑通信次数不一致时开。
export MINICPMO_MM_TRACE=${MINICPMO_MM_TRACE:-1}
export MINICPMO_MM_TRACE_FIRST_N=${MINICPMO_MM_TRACE_FIRST_N:-8}
[ -n "${TORCH_DISTRIBUTED_DEBUG}" ] && export TORCH_DISTRIBUTED_DEBUG

# =============================================================================
# 挂起取证（verl_omni/utils/hang_probe.py）
# =============================================================================
# py-spy 在这个容器里是 EPERM（没有 CAP_SYS_PTRACE），gdb / ray stack 同理，
# 所以只能让进程自己 dump 自己的栈。hang_probe 在 verl_omni/__init__.py 里安装，
# 每个 Ray worker 都会经过那里。两种取栈方式：
#
#   1) 手动：在任意节点上 `kill -USR1 <pid>`，立刻输出全部线程的 Python 栈。
#      整节点一把梭：
#        for p in $(pgrep -f "ray::WorkerDict"); do kill -USR1 $p; done
#      对 futex 里睡着的进程照样有效（信号会打断等待）。
#
#   2) 自动：MINICPMO_HANG_TIMEOUT_SEC 秒内埋点没有推进就自动 dump。
#      120s：合法的埋点间隔实测最大 ~10s（一个 micro-batch 的前反传），120s 已有
#      10 倍余量，而且远早于上面 HCCL_EXEC_TIMEOUT=300，HCCL 报超时之前 Python
#      侧栈就已经落盘了。原来的 300s 只是为了躲 torch_npu 那个 600s 看门狗。
#
#   3) MINICPMO_HANG_ABORT_AFTER=4：**训练开始之后**连续四次 dump 才自杀（exit 99）。
#      「训练开始之后」是 log 31 补的：原来 dump 计数器是全局的，rank 0 在 18:14:04
#      因为启动宽限期（120×4=480s）dump 过一次，等 18:19:05 真卡住时那已经算第 2 次
#      → 当场自杀。结果是唯一一次跑进 compute_log_prob 的实验只留下一张栈，而且是
#      「在等另外十五张卡」的那张。现在启动期的 dump 不再花训练期的预算。
#      4 × 120s = 480s，越过上面 HCCL_EXEC_TIMEOUT=300 的报错点。设 0 恢复只 dump 不退。
#
#   4) MINICPMO_HANG_ABORT_GRACE_SEC=45：决定自杀之后先等 45 秒再 os._exit。
#      自杀本身就是 Ray 拆任务的扳机，而真正想看的东西（同伴卡的栈、HCCL 正要抛的
#      算子级报错）都排在自己那份 dump 后面。log 31 就是没等这 45 秒。
#
#   5) 跨卡联动 dump：谁先到阈值，谁就写一个 $MINICPMO_HANG_DUMP_DIR/HANG_TRIGGER，
#      同节点其余各卡在一个轮询周期内一起出栈（没有开关，watchdog 开着就有）。
#      集合通信挂起只看一张卡的栈是没用的 —— log 31 死在「15 张卡在 AICPU kernel 里
#      等，rank 0 进不去」，而我们手里只有 rank 0 那一张。
#      => 最坏情况 480 + 45 ≈ 8.7 分钟，带 16 张栈 + HCCL 算子级报错退出，不用盯屏。
#
# stderr 只打主线程（538 个线程 × 32 卡的全量会淹掉 driver 日志），
# 全量写到 MINICPMO_HANG_DUMP_DIR 下的每卡一个文件。
export MINICPMO_HANG_PROBE=${MINICPMO_HANG_PROBE:-1}
export MINICPMO_HANG_TIMEOUT_SEC=${MINICPMO_HANG_TIMEOUT_SEC:-120}
export MINICPMO_HANG_ABORT_AFTER=${MINICPMO_HANG_ABORT_AFTER:-4}
export MINICPMO_HANG_ABORT_GRACE_SEC=${MINICPMO_HANG_ABORT_GRACE_SEC:-45}
export MINICPMO_HANG_DUMP_DIR="${DEPLOY_LOG_PATH}/hang_dump"
mkdir -p "${MINICPMO_HANG_DUMP_DIR}"
# 上一轮留下的 trigger 会让本轮每张卡一启动就白 dump 一次。hang_probe 内部已用进程
# 启动时间挡了一道，这里再清一次，重跑同一个目录时更稳。
rm -f "${MINICPMO_HANG_DUMP_DIR}/HANG_TRIGGER"

# === 多模态对齐兜底的响亮化（verl_omni_ext/models/qwen3_6_whisper/align_guard.py）===
# 两条兜底会在不抛异常的前提下改变模型看到的序列，是「复读退化」的头号嫌疑：
#   1) mrope 位置编码：text_len 为负时把整段序列的 (t,h,w) 换成 flat 1D；
#   2) merge：视觉/音频特征数与占位符数不等时，取最后 N 个占位符位置塞特征。
# 两者的失效形态都不是崩，而是「数字慢慢不对」，所以默认 1：一经触发立刻 raise，
# 把问题炸在启动后几分钟内、还带着 tag/pid/长度信息。
# 置 0 = 只响亮打印 + 计数（[verl-omni] ALIGN-FALLBACK <tag> #N pid=...），
# 并沿用旧的兜底行为，用来跑完一整轮取证。
export MINICPMO_MM_ALIGN_STRICT=${MINICPMO_MM_ALIGN_STRICT:-1}

# PYTHONPATH 已在文件开头设置，必须在各节点 ray start 之前 source。
ulimit -s unlimited

# === Ray 心跳超时（避免 update_actor 长时间运行导致节点被标记 dead）===
export RAY_health_check_failure_threshold=60
export RAY_health_check_period_ms=30000

# === GenRM dump（Judge 判分结果离线检查）===
export GENRM_DUMP_DIR="${PROJECT_ROOT}/genrm_dump"
export GENRM_DUMP_MAX=50

}
# END HUANYU environment policy


# Some schedulers wrap this node-level entry in torchrun/torch.distributed.launch
# with one Bash process per NPU. Elect local rank 0 BEFORE Python, setup, logs or
# flock; peers exit successfully so the outer launcher does not kill the owner.
# VC_TASK_INDEX remains the node rank: global RANK=0 would exclude other nodes.
_HUANYU_LOCAL_RANK="${LOCAL_RANK-}"
_HUANYU_HAS_LOCAL_RANK=0
if [[ -v LOCAL_RANK ]]; then _HUANYU_HAS_LOCAL_RANK=1; fi
_HUANYU_NODE_ARGS=()
while (( $# )); do
    case "$1" in
        --local-rank=*|--local_rank=*) _HUANYU_ARG_RANK="${1#*=}"; shift ;;
        --local-rank|--local_rank)
            (( $# >= 2 )) || { echo 'FATAL: missing local rank argument value' >&2; exit 2; }
            _HUANYU_ARG_RANK="$2"; shift 2 ;;
        *) _HUANYU_NODE_ARGS+=("$1"); shift; continue ;;
    esac
    if (( _HUANYU_HAS_LOCAL_RANK )) && [[ "$_HUANYU_LOCAL_RANK" != "$_HUANYU_ARG_RANK" ]]; then
        echo 'FATAL: conflicting local rank values in LOCAL_RANK / launcher arguments' >&2
        exit 2
    fi
    _HUANYU_LOCAL_RANK="$_HUANYU_ARG_RANK"
    _HUANYU_HAS_LOCAL_RANK=1
done
set -- "${_HUANYU_NODE_ARGS[@]}"
if (( _HUANYU_HAS_LOCAL_RANK )); then
    [[ "$_HUANYU_LOCAL_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || {
        echo 'FATAL: local rank must be a non-negative integer without leading zeroes' >&2; exit 2;
    }
    if [[ "$_HUANYU_LOCAL_RANK" != 0 ]]; then
        echo "[huanyu-job] local_rank=$_HUANYU_LOCAL_RANK node_rank=${VC_TASK_INDEX:-unknown} skipping duplicate node launcher; local_rank=0 owns setup/Ray"
        exit 0
    fi
    echo "[huanyu-job] local_rank=0 node_rank=${VC_TASK_INDEX:-unknown} selected node launcher pid=$$"
    # Ray/verl creates its own distributed workers. The outer launcher's ranks
    # must not make setup probes or the Ray driver look like a DDP worker.
    unset LOCAL_RANK RANK WORLD_SIZE LOCAL_WORLD_SIZE GROUP_RANK GROUP_WORLD_SIZE ROLE_RANK ROLE_WORLD_SIZE
    for _key in ${!TORCHELASTIC_@}; do unset "$_key"; done
fi

# Normal jobs opt in to extra native/device/cluster probes. Preparation always
# validates the environment; --start-ray also verifies registration through GCS.
# Keep the previous readiness switch as an alias; the new switch takes priority.
_PREFLIGHT="${HUANYU_JOB_PREFLIGHT:-${HUANYU_RAY_READINESS_CHECK:-0}}"
case "$_PREFLIGHT" in
    0|1) ;;
    *) echo 'FATAL: HUANYU_JOB_PREFLIGHT (or HUANYU_RAY_READINESS_CHECK) must be 0 or 1' >&2; exit 2 ;;
esac
_PREPARE_ONLY="${HUANYU_PREPARE_ONLY:-0}"
case "$_PREPARE_ONLY" in
    0) ;;
    1)
        [[ -r "${HUANYU_PREPARE_SNAPSHOT:-}" && -n "${HUANYU_PREPARED_ENV_FILE:-}" &&
           -d "${HUANYU_PREPARE_LOG_DIR:-}" ]] || {
            echo 'FATAL: use prepare_huanyu_env.sh for environment-only preparation' >&2
            exit 2
        }
        ;;
    *) echo 'FATAL: HUANYU_PREPARE_ONLY must be 0 or 1' >&2; exit 2 ;;
esac

_PREPARE_RAY=0
if (( _PREPARE_ONLY )); then
    _PREPARE_RAY="${HUANYU_PREPARE_START_RAY:-0}"
    [[ "$_PREPARE_RAY" == 0 || "$_PREPARE_RAY" == 1 ]] || { echo 'FATAL: invalid prepare Ray selection' >&2; exit 2; }
fi
_MANAGE_RAY=$(( ! _PREPARE_ONLY || _PREPARE_RAY ))

export PROJECT_ROOT="${PROJECT_ROOT:-/opt/huawei/dataset/infra_data/verl_omni_qwen36_gzq_new}"
_HUANYU_EXAMPLE_DIR="${PROJECT_ROOT}/examples/qwen3_6_huanyu"
cd "$PROJECT_ROOT"
if [[ "${HUANYU_JOB_MODE:-frozen_cache}" == aqa_unfreeze ]]; then
    export TRAIN_FILE="${TRAIN_FILE:-${HUANYU_AQA_TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/train.parquet}}"
    export VAL_FILE="${VAL_FILE:-${HUANYU_AQA_VAL_FILE:-/opt/huawei/dataset/infra_data/data/aqa_dataset/validation.parquet}}"
elif [[ "${HUANYU_JOB_MODE:-}" == stem ]]; then
    # Explicit common paths take priority over STEM aliases and directory defaults.
    export STEM_DATA_DIR="${STEM_DATA_DIR:-/opt/huawei/dataset/infra_data/data/rl_stem_data}"
    export TRAIN_FILE="${TRAIN_FILE:-${STEM_TRAIN_FILE:-$STEM_DATA_DIR/train.parquet}}"
    if [[ -z "${VAL_FILE:-}" && -z "${STEM_VAL_FILE:-}" &&
          ! -f "$STEM_DATA_DIR/val.parquet" && -f "$STEM_DATA_DIR/validation.parquet" ]]; then
        export STEM_VAL_FILE="$STEM_DATA_DIR/validation.parquet"
    fi
    export VAL_FILE="${VAL_FILE:-${STEM_VAL_FILE:-$STEM_DATA_DIR/val.parquet}}"
fi
# Resolve deployment paths once. Site environment files may still contain old
# defaults; neither they nor downstream wrappers should replace caller paths.
declare -A _HUANYU_PATH_CONFIG=(
    [PROJECT_ROOT]="$PROJECT_ROOT"
    [MODEL_PATH]="${MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/2748}"
    [RM_MODEL_PATH]="${RM_MODEL_PATH:-/opt/huawei/dataset/model_gy/downloaded_models/Qwen3.5-35B-A3B}"
    [TRAIN_FILE]="${TRAIN_FILE:-/opt/huawei/dataset/infra_data/data/train.parquet}"
    [VAL_FILE]="${VAL_FILE:-/opt/huawei/dataset/infra_data/data/validation.parquet}"
)
if (( _PREPARE_ONLY )); then
    for _key in DEFAULT_LOCAL_DIR TENSORBOARD_DIR GENRM_DUMP_DIR HUANYU_EMBED_CACHE_DIR RAY_ADDRESS; do
        if [[ -v "$_key" ]]; then _HUANYU_PATH_CONFIG["$_key"]="${!_key}"; fi
    done
fi
for _key in "${!_HUANYU_PATH_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PATH_CONFIG[$_key]}"
done
# The downstream training entry also invokes python3; keep installation and
# probes in that same interpreter environment.
_PYTHON=python3
_HELPER="${_HUANYU_EXAMPLE_DIR}/job_bootstrap.py"
_ENV_FILE="${HUANYU_ENV_FILE:-}"
_VLLM_SOURCE="${HUANYU_VLLM_SOURCE:-${HUANYU_VLLM_OMNI_ROOT:-/vllm-workspace/vllm-omni}}"
_RANK="${VC_TASK_INDEX:-}"
_WAIT="${RAY_CLUSTER_WAIT_TIMEOUT:-900}"
_INTERVAL="${RAY_CLUSTER_POLL_INTERVAL:-5}"
_PORT="${RAY_PORT:-6925}"
_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8960}"
_JOB_MODE="${HUANYU_JOB_MODE:-frozen_cache}"
# Resolve reward selection before pool validation and preserve it across site
# setup. An explicit public mode wins over the old per-entry switches.
_REWARD_MODE="${HUANYU_REWARD_MODE:-}"
if [[ -z "$_REWARD_MODE" ]]; then
    if [[ "$_JOB_MODE" == stem && -n "${STEM_REWARD_MODE:-}" ]]; then
        _REWARD_MODE="$STEM_REWARD_MODE"
    elif [[ -n "${RM_ENABLE:-}" ]]; then
        case "$RM_ENABLE" in
            0) _REWARD_MODE=rule_based ;;
            1) _REWARD_MODE=RM ;;
            *) echo 'FATAL: RM_ENABLE must be 0 or 1' >&2; exit 2 ;;
        esac
    elif [[ "$_JOB_MODE" == stem ]]; then
        _REWARD_MODE=rule_based
    else
        _REWARD_MODE=RM
    fi
fi
case "${_REWARD_MODE,,}" in
    rm|genrm) _REWARD_MODE=RM; _REWARD_ENABLED=1 ;;
    rule|rule_based|ruled_based) _REWARD_MODE=rule_based; _REWARD_ENABLED=0 ;;
    *) echo 'FATAL: HUANYU_REWARD_MODE must be RM or rule_based' >&2; exit 2 ;;
esac
export HUANYU_REWARD_MODE="$_REWARD_MODE" RM_ENABLE="$_REWARD_ENABLED"
_RAY_READINESS_CHECK="$_PREFLIGHT"
_CLUSTER_NODES="${HUANYU_CLUSTER_NODES:-2}"
if [[ "$_JOB_MODE" == stem ]]; then
    [[ "$_CLUSTER_NODES" == 1 || "$_CLUSTER_NODES" == 2 || "$_CLUSTER_NODES" == 8 || "$_CLUSTER_NODES" == 16 ]] || {
        echo 'FATAL: STEM jobs require HUANYU_CLUSTER_NODES=1, 2, 8 or 16 (16 NPUs per node)' >&2; exit 2;
    }
else
    [[ "$_CLUSTER_NODES" == 2 ]] || {
        echo 'FATAL: frozen_cache and aqa_unfreeze jobs require HUANYU_CLUSTER_NODES=2' >&2; exit 2;
    }
fi
# Defaults for direct invocation; a profile such as the 16K wrapper supplies
# these first. Snapshot before sourcing the site env so it cannot replace the
# caller's pool sizes. Physical Ray capacity remains 16 NPUs on each node.
declare -A _HUANYU_POOL_CONFIG=(
    [NUM_NPUS]="${NUM_NPUS:-16}" [NNODES]="${NNODES:-1}"
    [ROLLOUT_TP]="${ROLLOUT_TP:-4}" [RM_TP]="${RM_TP:-4}"
    [RM_N_GPUS_PER_NODE]="${RM_N_GPUS_PER_NODE:-16}" [RM_NNODES]="${RM_NNODES:-1}"
    [RM_SPLIT_NODES]="${RM_SPLIT_NODES:-1}"
)
if (( ! _REWARD_ENABLED )); then _HUANYU_POOL_CONFIG[RM_NNODES]=0; fi
for _key in NUM_NPUS NNODES ROLLOUT_TP RM_TP RM_N_GPUS_PER_NODE RM_NNODES; do
    if [[ "$_key" == RM_NNODES && "${_HUANYU_POOL_CONFIG[$_key]}" == 0 && "$_REWARD_ENABLED" == 0 ]]; then continue; fi
    if [[ ! "${_HUANYU_POOL_CONFIG[$_key]}" =~ ^[1-9][0-9]?$ ]]; then
        echo "FATAL: $_key must be a positive integer (without leading zeroes)" >&2
        exit 2
    fi
done
_ACTOR_PER_NODE=${_HUANYU_POOL_CONFIG[NUM_NPUS]}
_ACTOR_NODES=${_HUANYU_POOL_CONFIG[NNODES]}
_RM_PER_NODE=${_HUANYU_POOL_CONFIG[RM_N_GPUS_PER_NODE]}
_RM_NODES=${_HUANYU_POOL_CONFIG[RM_NNODES]}
if (( ! _PREPARE_ONLY )); then
if [[ "$_JOB_MODE" == stem ]]; then
    if (( _ACTOR_PER_NODE != 16 || _ACTOR_NODES + _RM_NODES > _CLUSTER_NODES ||
          _ACTOR_PER_NODE % ${_HUANYU_POOL_CONFIG[ROLLOUT_TP]} != 0 ||
          (_REWARD_ENABLED && (_RM_PER_NODE != 16 || _RM_NODES < 1 ||
           _RM_PER_NODE % ${_HUANYU_POOL_CONFIG[RM_TP]} != 0)) )); then
        echo 'FATAL: STEM requires 16-card node-local Actor/RM pools within the physical nodes, with each TP dividing 16' >&2
        exit 2
    fi
else
if (( _ACTOR_PER_NODE > 16 || _RM_PER_NODE > 16 || _ACTOR_NODES > 2 || _RM_NODES > 2 ||
      _ACTOR_PER_NODE * _ACTOR_NODES + _RM_PER_NODE * _RM_NODES > 32 ||
      _ACTOR_PER_NODE % ${_HUANYU_POOL_CONFIG[ROLLOUT_TP]} != 0 ||
      (_REWARD_ENABLED && (_RM_NODES < 1 || _RM_PER_NODE % ${_HUANYU_POOL_CONFIG[RM_TP]} != 0)) )); then
    echo 'FATAL: pool sizes must fit within 32 on two 16-card nodes; each enabled per-node pool must be divisible by its TP' >&2
    exit 2
fi
# Each per-node pool is STRICT_PACK. Check it can be packed into two physical
# nodes; checking only the total would accept impossible [12,12]+[8].
_POOLS_FIT=0
for (( _a=0; _a<=_ACTOR_NODES; _a++ )); do
    for (( _r=0; _r<=_RM_NODES; _r++ )); do
        if (( _a * _ACTOR_PER_NODE + _r * _RM_PER_NODE <= 16 &&
              (_ACTOR_NODES - _a) * _ACTOR_PER_NODE + (_RM_NODES - _r) * _RM_PER_NODE <= 16 )); then
            _POOLS_FIT=1
        fi
    done
done
if (( ! _POOLS_FIT )); then
    echo 'FATAL: per-node pools cannot fit on two 16-card nodes; for actor24 + RM8 use actor12x2 and RM4x2' >&2
    exit 2
fi
if [[ "$_JOB_MODE" == aqa_unfreeze ]] && (( _REWARD_ENABLED )) &&
    (( _ACTOR_PER_NODE != 16 || _ACTOR_NODES != 1 || _RM_PER_NODE != 16 || _RM_NODES != 1 )); then
    echo 'FATAL: AQA jobs require actor16 + RM16' >&2
    exit 2
fi
fi  # STEM assigns whole nodes; the two-node multimodal mode also permits smaller pools.
fi  # Pool packing is checked when launching the job, not during standalone setup.
case "$_JOB_MODE" in
    frozen_cache) _TRAIN_ENTRY=run_qwen36_huanyu_gspo_npu_omni_mix_genrm.sh ;;
    aqa_unfreeze) _TRAIN_ENTRY=run_huanyu_aqa_audio_train.sh ;;
    stem) _TRAIN_ENTRY=run_huanyu_stem.sh ;;
    *) echo 'FATAL: HUANYU_JOB_MODE must be frozen_cache, aqa_unfreeze or stem' >&2; exit 2 ;;
esac
# Keep the component choices through site environment setup and into existing
# Ray workers. The frozen-cache entry always freezes both audio components.
if [[ "$_JOB_MODE" == frozen_cache || "$_JOB_MODE" == stem ]]; then
    export HUANYU_FREEZE_AUDIO=true
    export HUANYU_TRAIN_AUDIO_ENCODER=false HUANYU_TRAIN_AUDIO_PROJECTOR=false
fi
huanyu_resolve_audio_train_mode || exit $?
declare -A _HUANYU_AUDIO_CONFIG=()
for _key in HUANYU_FREEZE_AUDIO HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR MINICPMO_TUNE_AUDIO; do
    _HUANYU_AUDIO_CONFIG["$_key"]="${!_key}"
done
[[ "$_PREPARE_ONLY" == 1 || -f "$PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" ]] || {
    echo "FATAL: missing training entry $PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" >&2
    exit 1
}
_REQUIRED_COMMANDS=("$_PYTHON" git pgrep flock)
if (( _MANAGE_RAY )); then _REQUIRED_COMMANDS+=(ray timeout); fi
for _command in "${_REQUIRED_COMMANDS[@]}"; do
    command -v "$_command" >/dev/null || { echo "FATAL: missing command $_command" >&2; exit 1; }
done
if (( _MANAGE_RAY )); then
    if [[ ! "$_RANK" =~ ^(0|[1-9][0-9]?)$ ]] || (( _RANK >= _CLUSTER_NODES )); then
        echo "FATAL: set VC_TASK_INDEX=0..$((_CLUSTER_NODES - 1)); rank 0 is head; no implicit rank 0" >&2
        exit 2
    fi
    if [[ -n "${VC_WORKER_NUM:-}" && "$VC_WORKER_NUM" != "$_CLUSTER_NODES" ]]; then
        echo "FATAL: this launcher expects $_CLUSTER_NODES nodes, each with 16 NPUs" >&2
        exit 2
    fi
fi
for _key in _WAIT _INTERVAL _PORT _DASHBOARD_PORT; do
    if [[ ! "${!_key}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FATAL: $_key must be a positive integer" >&2
        exit 2
    fi
done
if (( _PORT > 65535 || _DASHBOARD_PORT > 65535 || _PORT == _DASHBOARD_PORT )); then
    echo 'FATAL: Ray and dashboard ports must be distinct values in 1..65535' >&2
    exit 2
fi

# Platform jobs are non-interactive and may never read the image's .bashrc
# jemalloc export. Set it before Python/pip and the Ascend setup scripts (which
# also import torch), not just before Ray. This restores the allocator setup in
# the known working job entry; a native import must still exit successfully.
_HUANYU_JOB_JEMALLOC="${HUANYU_JEMALLOC_LIBRARY:-${Q36_JEMALLOC_LIBRARY:-}}"
_HUANYU_PRELOAD_LIBS=()
_HUANYU_ORIGINAL_PRELOAD="${LD_PRELOAD:-}"
read -r -a _HUANYU_PRELOAD_LIBS <<< "${_HUANYU_ORIGINAL_PRELOAD//:/ }"
if [[ -z "$_HUANYU_JOB_JEMALLOC" ]]; then
    for _library in "${_HUANYU_PRELOAD_LIBS[@]}"; do
        if [[ "$_library" == /* && "${_library##*/}" == libjemalloc*.so* && -f "$_library" && -r "$_library" ]]; then
            _HUANYU_JOB_JEMALLOC="$_library"
            break
        fi
    done
fi
if [[ -z "$_HUANYU_JOB_JEMALLOC" ]]; then
    for _library in /usr/lib/"$(uname -m)"-linux-gnu/libjemalloc.so.2 \
        /usr/lib64/libjemalloc.so.2 /usr/lib/libjemalloc.so.2 \
        /lib/"$(uname -m)"-linux-gnu/libjemalloc.so.2; do
        if [[ -f "$_library" && -r "$_library" ]]; then
            _HUANYU_JOB_JEMALLOC="$_library"
            break
        fi
    done
fi
if [[ "$_HUANYU_JOB_JEMALLOC" != /* || ! -f "$_HUANYU_JOB_JEMALLOC" || ! -r "$_HUANYU_JOB_JEMALLOC" \
    || "$_HUANYU_JOB_JEMALLOC" == *[[:space:]:]* ]]; then
    echo 'FATAL: set HUANYU_JEMALLOC_LIBRARY to a readable absolute libjemalloc.so path on BOTH nodes (no spaces/colons)' >&2
    exit 1
fi
huanyu_preload_allocator() {
    local library
    local -a libraries=()
    local current_preload="${LD_PRELOAD:-}"
    read -r -a libraries <<< "${current_preload//:/ }"
    # Keep unrelated preloads, but do not stack multiple jemalloc versions.
    export LD_PRELOAD="$_HUANYU_JOB_JEMALLOC"
    for library in "${libraries[@]}"; do
        if [[ "${library##*/}" != libjemalloc*.so* && "$library" != "$_HUANYU_JOB_JEMALLOC" ]]; then
            export LD_PRELOAD="$LD_PRELOAD:$library"
        fi
    done
}
huanyu_preload_allocator
[[ -f "$_HELPER" && ( -z "$_ENV_FILE" || -f "$_ENV_FILE" ) ]] || {
    echo "FATAL: missing job_bootstrap.py or explicit HUANYU_ENV_FILE" >&2; exit 1;
}
echo "[huanyu-job] allocator before Python/env setup: LD_PRELOAD=$LD_PRELOAD"
if (( _MANAGE_RAY )); then
    _IPS="$("$_PYTHON" -B "$_HELPER" resolve "${VC_WORKER_HOSTS:-}" --expected-nodes "$_CLUSTER_NODES")"
    IFS=, read -r -a _NODE_IPS <<< "$_IPS"
    _MASTER_IP="${_NODE_IPS[0]}"
    _NODE_IP="${_NODE_IPS[$_RANK]}"
fi

# Filter only scheduler --NAME=value metadata; preserve Hydra argv boundaries.
_TRAIN_ARGS=()
for _arg in "$@"; do
    case "$_arg" in
        --cfg=*|--package=*|--config-path=*|--config-name=*|--config-dir=*|--experimental-rerun=*|--info=*)
            _TRAIN_ARGS+=("$_arg") ;;
        --*=*) echo "[huanyu-job] ignoring scheduler argument ${_arg%%=*}" ;;
        *) _TRAIN_ARGS+=("$_arg") ;;
    esac
done

# Preserve exported training settings in every mode, including frozen_cache.
# Site setup provides defaults; it must not replace the caller's experiment.
declare -A _HUANYU_PROFILE_CONFIG=()
for _key in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
        ROLLOUT_N ACTOR_MODEL_DTYPE REF_MODEL_DTYPE ACTOR_OFFLOAD_POLICY \
        ACTOR_FORWARD_PREFETCH ACTIVATION_OFFLOAD FILTER_OVERLONG \
        MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH ROLLOUT_PROMPT_LENGTH \
        ROLLOUT_MAX_MODEL_LEN ROLLOUT_MAX_NUM_BATCHED_TOKENS MAX_TOKEN_LEN_PER_GPU MINICPMO_MM_PROMPT_BUDGET AQA_DYNAMIC_BSZ \
        ROLLOUT_TEMPERATURE ROLLOUT_TOP_P ROLLOUT_TOP_K ROLLOUT_REPETITION_PENALTY \
        ROLLOUT_FREQUENCY_PENALTY ROLLOUT_PRESENCE_PENALTY ROLLOUT_LOGPROBS_MODE \
        ROLLOUT_GPU_MEM_UTIL HUANYU_GRAPH_MODE PROJECT_NAME EXPERIMENT_NAME TOTAL_TRAINING_STEPS \
        UPDATE_WEIGHTS_BUCKET_MB TOTAL_EPOCHS TRAINER_LOGGER SAVE_FREQ RESUME_MODE ENABLE_THINKING \
        DEFAULT_LOCAL_DIR TENSORBOARD_DIR GENRM_DUMP_DIR ROLLOUT_DATA_DIR VALIDATION_DATA_DIR WANDB_MODE \
        TEST_FREQ VAL_BEFORE_TRAIN ROLLOUT_MAX_NUM_SEQS ROLLOUT_NUM_WORKERS HUANYU_RUN_DIR HUANYU_RUN_BASE \
        HUANYU_EMBED_CACHE_DIR HUANYU_JOB_PROFILE HUANYU_REWARD_MODE Q36_BACKWARD_PREFETCH \
        HUANYU_STEM_TEXT_TRAIN HUANYU_CHECKPOINT_VALIDATION_MODE \
        HUANYU_FULL_WEIGHT_AUDIT_DIR HUANYU_FULL_WEIGHT_AUDIT_RUN_ID HUANYU_WEIGHT_SYNC_AUDIT_DIR \
        HUANYU_AUDIO_PROBE_DIR HUANYU_TEXT_AB_DIR HUANYU_EMBEDDING_AB_DIR HUANYU_EMBEDDING_AB_SIDE \
        ${!ACTOR_@} ${!REF_@} ${!ROLLOUT_@} ${!PPO_@} ${!RM_@} ${!STEM_@} ${!MAX_@} \
        ${!GENRM_@} ${!HUANYU_GEN_@} ${!HUANYU_FORWARD_@} ${!HUANYU_NPU_@}; do
    if [[ -v "$_key" ]]; then _HUANYU_PROFILE_CONFIG["$_key"]="${!_key}"; fi
done
if [[ "$_JOB_MODE" == aqa_unfreeze ]]; then
    _HUANYU_PROFILE_CONFIG[HUANYU_GRAPH_MODE]="${HUANYU_GRAPH_MODE:-0}"
    for _key in HUANYU_AQA_TRAIN_FILE HUANYU_AQA_VAL_FILE; do
        if [[ -v "$_key" ]]; then _HUANYU_PROFILE_CONFIG["$_key"]="${!_key}"; fi
    done
fi

# Own the numerical policy at this entry, even with an older/custom site env.
# The default full profile overrides stale per-kernel opt-outs. custom keeps
# explicit caller values for comparisons; site env must not replace the policy.
huanyu_configure_precision_mode "${_TRAIN_ARGS[@]}" || exit $?
declare -A _HUANYU_PRECISION_CONFIG=()
for _key in "${HUANYU_PRECISION_KEYS[@]}"; do
    _HUANYU_PRECISION_CONFIG["$_key"]="${!_key}"
done

if (( _PREPARE_ONLY )); then
    # The outer environment entry captures early errors as well as these steps.
    _LOG_DIR="$HUANYU_PREPARE_LOG_DIR"
else
    export EXPERIMENT_NAME="${EXPERIMENT_NAME:-huanyu_sampleall_cache_32card}"
    _LOG_DIR="${HUANYU_JOB_LOG_DIR:-$PROJECT_ROOT/runs/$EXPERIMENT_NAME/cluster}"
    mkdir -p "$_LOG_DIR"
    exec > >(tee -a "$_LOG_DIR/rank_$_RANK.log") 2>&1
fi
# Local lock stays held by this launcher, never inherited by a detached raylet.
exec 9>"${TMPDIR:-/tmp}/huanyu-32card-${UID}.lock"
flock -n 9 || { echo 'FATAL: another Huanyu job launcher is already running on this node' >&2; exit 1; }
if (( _PREPARE_ONLY )); then
    echo "[huanyu-env] prepare mode=$_JOB_MODE start_ray=$_PREPARE_RAY log=$_LOG_DIR/prepare.log"
else
    echo "[huanyu-job] rank=$_RANK platform_ip=$_NODE_IP ray_node_ip=auto head=$_MASTER_IP:$_PORT log=$_LOG_DIR/rank_$_RANK.log"
fi

_LIVE_RAY=0
if pgrep -x raylet >/dev/null; then _LIVE_RAY=1; fi
export HUANYU_VLLM_SOURCE="$_VLLM_SOURCE"
huanyu_select_native_source || exit $?
_VLLM_SOURCE="$HUANYU_VLLM_SOURCE"
echo "[huanyu-job] expected installed vLLM-Omni=$_VLLM_SOURCE (no source PYTHONPATH injection)"
if (( _LIVE_RAY )); then
    echo '[huanyu-job] reusing local Ray; editable installation skipped; existing workers keep their loaded code'
else
    # Install this project's plugin, not the shared verl / verl-omni repositories.
    [[ -f "$PROJECT_ROOT/pyproject.toml" || -f "$PROJECT_ROOT/setup.py" ]] || {
        echo "FATAL: no packaging file under PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1;
    }
    git config --global --add safe.directory "$PROJECT_ROOT"
    "$_PYTHON" -m pip install --no-build-isolation --no-deps -e "$PROJECT_ROOT"
fi

# Source the established Huanyu settings. It reads optional/unset variables and
# enables xtrace itself; restore our shell options afterwards.
export HUANYU_GRAPH_MODE="${HUANYU_GRAPH_MODE:-0}"
set +u
if [[ -n "$_ENV_FILE" ]]; then source "$_ENV_FILE"; else huanyu_prepare_environment "$@"; fi
set +x
set -euo pipefail
# A custom/older site env may reset these to /vllm-workspace/vllm-omni.
# Restore the expected installed origin and remove old source-path overrides.
export HUANYU_VLLM_SOURCE="$_VLLM_SOURCE"
huanyu_select_native_source || exit $?
for _key in "${!_HUANYU_PATH_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PATH_CONFIG[$_key]}"
done
for _key in "${!_HUANYU_PROFILE_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PROFILE_CONFIG[$_key]}"
done
if [[ "$_JOB_MODE" == aqa_unfreeze || "$_JOB_MODE" == stem ]]; then
    # Only discard legacy site defaults. An identical path explicitly exported
    # by the caller is intentional and must reach the training backend.
    if [[ -z "${_HUANYU_PROFILE_CONFIG[DEFAULT_LOCAL_DIR]+set}" &&
          "${DEFAULT_LOCAL_DIR:-}" == "$PROJECT_ROOT/checkpoints" ]]; then
        unset DEFAULT_LOCAL_DIR
    fi
    if [[ -z "${_HUANYU_PROFILE_CONFIG[TENSORBOARD_DIR]+set}" &&
          "${TENSORBOARD_DIR:-}" == "$PROJECT_ROOT/tensorboard_log" ]]; then
        unset TENSORBOARD_DIR
    fi
fi
export Q36_CKPT_DIR="$MODEL_PATH"
cd "$PROJECT_ROOT"
huanyu_preload_allocator
echo "[huanyu-job] allocator for native check/Ray: LD_PRELOAD=$LD_PRELOAD"
export RAY_PORT="$_PORT" RAY_DASHBOARD_PORT="$_DASHBOARD_PORT"
export HUANYU_JOB_PREFLIGHT="$_PREFLIGHT"
export HUANYU_RAY_READINESS_CHECK="$_RAY_READINESS_CHECK"
if (( _MANAGE_RAY )); then export RAY_ADDRESS="$_MASTER_IP:$_PORT"; fi
export HUANYU_JOB_MODE="$_JOB_MODE" HUANYU_CLUSTER_NODES="$_CLUSTER_NODES"
export RAY_USAGE_STATS_ENABLED=0 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
# Apply after site settings, before ray start on every node. These control
# process startup, not the total AgentLoopWorker count. Existing raylets retain
# their environment; prepare snapshots also carry these values for manual starts.
export RAY_worker_maximum_startup_concurrency=32
export RAY_prestart_worker_first_driver=1
export RAY_enable_worker_prestart=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export MINICPMO_USE_PRECOMPUTE=0 MINICPMO_TUNE_VISION=0
export HUANYU_STEM_TEXT_TRAIN=0
for _key in "${!_HUANYU_AUDIO_CONFIG[@]}"; do
    export "$_key=${_HUANYU_AUDIO_CONFIG[$_key]}"
done
case "$_JOB_MODE" in
    frozen_cache)
        export HUANYU_USE_PRECOMPUTE=1 HUANYU_FREEZE_AUDIO=true
        export HUANYU_EMBED_CACHE_DIR="${HUANYU_EMBED_CACHE_DIR:-/opt/huawei/dataset/infra_data/data/embed_cache_huanyu_sampleall_q128}"
        export MINICPMO_TUNE_AUDIO=0 HUANYU_AQA_AUDIO_TRAIN=0
        [[ -r "$HUANYU_EMBED_CACHE_DIR/manifest.json" ]] || {
            echo "FATAL: cache manifest not readable on rank $_RANK: $HUANYU_EMBED_CACHE_DIR/manifest.json" >&2
            exit 1
        }
        ;;
    aqa_unfreeze)
        # Apply before Ray starts on BOTH nodes, after the site environment.
        # Empty the cache path as well as disabling it, including runtime_env
        # below for workers launched by an already-running frozen-cache raylet.
        export HUANYU_USE_PRECOMPUTE=0
        export HUANYU_EMBED_CACHE_DIR=""
        export HUANYU_AQA_AUDIO_TRAIN=1
        export Q36_BACKWARD_PREFETCH=off
        ;;
    stem)
        export HUANYU_USE_PRECOMPUTE=0 HUANYU_EMBED_CACHE_DIR=""
        export HUANYU_AQA_AUDIO_TRAIN=0 HUANYU_STEM_TEXT_TRAIN=1
        export Q36_BACKWARD_PREFETCH=off
        ;;
esac
export DEBUG_MODE=0 DEBUG_SAMPLES=0
for _key in "${!_HUANYU_POOL_CONFIG[@]}"; do
    export "$_key=${_HUANYU_POOL_CONFIG[$_key]}"
done
export NPUS_PER_NODE=16
export HUANYU_REWARD_MODE="$_REWARD_MODE" RM_ENABLE="$_REWARD_ENABLED"
if (( ! _REWARD_ENABLED )); then
    export RM_ENABLE=0 RM_NNODES=0 REWARD_POOL_NPUS=0
else
    export REWARD_POOL_NPUS=$((RM_N_GPUS_PER_NODE * RM_NNODES)) RM_ENABLE=1
fi
export NUM_GPUS_ACTOR_ROLLOUT_REWARD=16
export WANDB_MODE="${WANDB_MODE:-offline}"
for _key in "${!_HUANYU_PRECISION_CONFIG[@]}"; do
    export "$_key=${_HUANYU_PRECISION_CONFIG[$_key]}"
done
# Source our helper again after site settings and the final pool/cache mode.
# This also prevents an older env file from selecting the Whisper engine hook.
huanyu_configure_precision_mode "${_TRAIN_ARGS[@]}" || exit $?
echo "[huanyu-job] precision before Ray: train_GDN=${MINICPMO_HUANYU_GDN_FP32_TRAINING} rollout_GDN=${MINICPMO_GDN_FP32_REFERENCE} reload=${MINICPMO_HUANYU_LAYERWISE_RELOAD} KV_zero=${MINICPMO_HYBRID_BLOCK_ZERO_REFERENCE} isolate_media=${MINICPMO_ISOLATE_MEDIA_ENCODERS} actual_TP=${MINICPMO_HUANYU_ROLLOUT_TP}"
echo "[huanyu-job] inference hooks: ${VLLM_OMNI_EXTERNAL_MODULES}"
echo "[huanyu-job] mode=$_JOB_MODE reward=$HUANYU_REWARD_MODE freeze_audio=$HUANYU_FREEZE_AUDIO precompute=$HUANYU_USE_PRECOMPUTE train_entry=$_TRAIN_ENTRY"
echo "[huanyu-job] train_audio_encoder=$HUANYU_TRAIN_AUDIO_ENCODER train_audio_projector=$HUANYU_TRAIN_AUDIO_PROJECTOR"
echo "[huanyu-job] physical=16x${_CLUSTER_NODES} actor=${NUM_NPUS}x${NNODES} rollout_TP=$ROLLOUT_TP RM=${RM_N_GPUS_PER_NODE}x${RM_NNODES} RM_TP=$RM_TP requested=$((NUM_NPUS * NNODES + REWARD_POOL_NPUS))"
if (( _PREFLIGHT || _PREPARE_ONLY )); then
    "$_PYTHON" -B "$PROJECT_ROOT/verl_omni_ext/models/qwen3_6_huanyu/runtime_imports.py" \
        --check --expected-root "$_VLLM_SOURCE" || exit $?
    echo '[huanyu-job] native import check exited successfully'
    "$_PYTHON" -B "$_HELPER" check-devices || exit $?
else
    echo '[huanyu-job] skipping native/device/cluster preflight (HUANYU_JOB_PREFLIGHT=0)'
fi

wait_for_cluster() {
    local -a options=()
    if (( _PREPARE_RAY )); then
        options+=(--gcs)
    fi
    local status=0
    timeout --foreground "${_WAIT}s" "$_PYTHON" -B "$_HELPER" wait-cluster \
        --address "$RAY_ADDRESS" --expected-ips "$_IPS" --local-ip "$_NODE_IP" \
        --timeout "$_WAIT" --interval "$_INTERVAL" --mode "$1" "${options[@]}" || status=$?
    if (( status )); then
        echo "FATAL: Ray readiness check failed/timed out (exit $status); training not started" >&2
        return "$status"
    fi
}

if (( _MANAGE_RAY )); then
    # Apply after site setup, matching the standalone Ray launcher on every node.
    unset HCCL_LOGIC_SUPERPOD_ID
    "$_PYTHON" -B "$_HELPER" clear-superpod-bashrc "${HOME:?HOME must be set}/.bashrc"
    echo "[huanyu-job] Ray rank=$_RANK address=$RAY_ADDRESS local_ip=$_NODE_IP"
    if (( _RANK == 0 )); then
        if (( ! _LIVE_RAY )); then
            ray start --head --port="$RAY_PORT" \
                --dashboard-host=0.0.0.0 --dashboard-port="$RAY_DASHBOARD_PORT" \
                --resources='{"NPU": 16}' 9>&-
        fi
        if (( _PREFLIGHT || _PREPARE_RAY )); then wait_for_cluster all || exit $?; fi
    else
        timeout --foreground "${_WAIT}s" "$_PYTHON" -B "$_HELPER" wait-head \
            --address "$RAY_ADDRESS" --timeout "$_WAIT" --interval "$_INTERVAL" || exit $?
        if (( ! _LIVE_RAY )); then
            echo '[huanyu-job] joining as worker; no training on this node'
            ray start --address="$RAY_ADDRESS" --resources='{"NPU": 16}' 9>&-
        fi
        if (( _PREPARE_RAY || (_PREFLIGHT && _LIVE_RAY) )); then wait_for_cluster local || exit $?; fi
    fi
fi

if (( _PREPARE_ONLY )); then
    if (( _LIVE_RAY )); then
        echo '[huanyu-env] existing Ray was left running; activation does not update its inherited environment'
    fi
    export HUANYU_SKIP_EDITABLE_INSTALL=1
    huanyu_clear_persisted_native_source
    "$_PYTHON" -S "${_HUANYU_EXAMPLE_DIR}/prepared_env.py" write \
        --snapshot "$HUANYU_PREPARE_SNAPSHOT" --output "$HUANYU_PREPARED_ENV_FILE" \
        --stack-limit "$(ulimit -S -s)"
    echo "[huanyu-env] READY activation=$HUANYU_PREPARED_ENV_FILE"
    exit 0
fi

if (( _RANK == 0 )); then
    # Explicitly propagate cache settings and the allocator even when the Raylet
    # predates this job. Neither Huanyu nor LD_PRELOAD is in the shared ext
    # environment whitelist; Ray can also strip its own allocator preload.
    _RUNTIME_ARGS=()
    for _key in HUANYU_USE_PRECOMPUTE HUANYU_EMBED_CACHE_DIR HUANYU_FREEZE_AUDIO \
        HUANYU_TRAIN_AUDIO_ENCODER HUANYU_TRAIN_AUDIO_PROJECTOR \
        HUANYU_AQA_AUDIO_TRAIN HUANYU_STEM_TEXT_TRAIN MINICPMO_USE_PRECOMPUTE MINICPMO_TUNE_AUDIO \
        MINICPMO_TUNE_VISION WANDB_MODE LD_PRELOAD HUANYU_REWARD_MODE; do
        _value="$("$_PYTHON" -c 'import json, os, sys; print(json.dumps(os.environ[sys.argv[1]]))' "$_key")"
        _RUNTIME_ARGS+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_key}=${_value}")
    done
    if [[ "$_RAY_READINESS_CHECK" == 1 ]]; then
        echo "[huanyu-job] all $_CLUSTER_NODES nodes registered ($((_CLUSTER_NODES * 16)) NPU); head starting $_JOB_MODE Huanyu training (preflight=$_PREFLIGHT)"
    else
        echo "[huanyu-job] head submitting $_JOB_MODE training to $RAY_ADDRESS (preflight=$_PREFLIGHT); expected capacity=$((_CLUSTER_NODES * 16)) NPU, not prechecked"
    fi
    bash "$PROJECT_ROOT/examples/qwen3_6_huanyu/$_TRAIN_ENTRY" \
        "${_TRAIN_ARGS[@]}" \
        "++actor_rollout_ref.model.override_config.huanyu_freeze_audio=${HUANYU_FREEZE_AUDIO}" \
        "++actor_rollout_ref.model.override_config.huanyu_train_audio_encoder=${HUANYU_TRAIN_AUDIO_ENCODER}" \
        "++actor_rollout_ref.model.override_config.huanyu_train_audio_projector=${HUANYU_TRAIN_AUDIO_PROJECTOR}" \
        "${_RUNTIME_ARGS[@]}" "${HUANYU_PRECISION_ARGS[@]}"
else
    if (( _LIVE_RAY )); then
        if [[ "$_RAY_READINESS_CHECK" == 1 ]]; then
            echo '[huanyu-job] worker registered in the selected cluster; waiting, no training on this node'
        else
            echo '[huanyu-job] keeping existing worker raylet; cluster membership unchecked, no training on this node'
        fi
    fi
    # ray start returns after startup, just like the working manual command.
    # Keep the node job and its lock alive without creating a Ray probe driver.
    echo '[huanyu-job] worker launcher staying alive; monitoring local raylet only'
    while pgrep -x raylet >/dev/null; do sleep 30; done
    echo 'FATAL: worker raylet exited' >&2
    exit 1
fi
