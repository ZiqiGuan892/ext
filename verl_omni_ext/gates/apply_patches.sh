#!/bin/bash
# ============================================================================
# Gate patch 应用脚本
#
# 给 vllm-omni 打 gate patch，加 VLLM_OMNI_EXTERNAL_MODULES 扩展点。
# 打完后，新模型的 pipeline 定义可以放在 verl_omni_ext 里，
# 不需要每次加模型都改 vllm-omni 源码树。
#
# 用法：
#   bash verl_omni_ext/gates/apply_patches.sh /path/to/vllm-omni
#
# 验证（gate off = 原行为不变）：
#   unset VLLM_OMNI_EXTERNAL_MODULES
#   python -c "import vllm_omni"  # 应该和没打补丁一样
# ============================================================================
set -euo pipefail

VLLM_OMNI_DIR="${1:?Usage: $0 <vllm-omni-source-dir>}"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_FILE="$PATCH_DIR/vllm_omni_external_modules.patch"

echo "=== Applying gate patches to vllm-omni ==="
echo "  vllm-omni source: $VLLM_OMNI_DIR"
echo "  patch file: $PATCH_FILE"
echo ""

# 检查是否已打过补丁
if grep -q "VLLM_OMNI_EXTERNAL_MODULES" "$VLLM_OMNI_DIR/vllm_omni/model_executor/models/registry.py" 2>/dev/null; then
    echo "⚠ Patch already applied, skipping."
    echo "  To re-apply: cd $VLLM_OMNI_DIR && git checkout -- vllm_omni/model_executor/models/registry.py"
    exit 0
fi

# 应用补丁
cd "$VLLM_OMNI_DIR"
git apply "$PATCH_FILE" 2>/dev/null || {
    echo "⚠ git apply failed, trying patch command..."
    patch -p1 < "$PATCH_FILE"
}

echo ""
echo "✓ Patch applied successfully."
echo ""
echo "Usage:"
echo "  export VLLM_OMNI_EXTERNAL_MODULES=verl_omni_ext.models.qwen3_5_moe.vllm_omni"
echo "  # 然后 vllm-omni 启动时会自动加载 ext 包里的 pipeline 定义"
echo ""
echo "Verify (gate off = no behavior change):"
echo "  unset VLLM_OMNI_EXTERNAL_MODULES"
echo "  python -c 'import vllm_omni'  # should work as before"
