"""
槽位①: Qwen3.5-MoE Thinker 训练适配器

@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")

时序陷阱（必须记住）：
  from_pretrained 在 configure_model 之前执行（fsdp/omni_impl.py:185）。
  所以"必须在模型加载前生效"的补丁不能放这里，必须放包 import 期。
  这里只放"打在已加载实例上"的补丁（如 forward 重定向、device fix）。
"""
from typing import Any

from verl_omni.pipelines.model_base import OmniModelBase

from .patches import apply_qwen3_5_vision_device_fix


@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")
class Qwen35MoeThinkerAdapter(OmniModelBase):
    """Qwen3.5-MoE Thinker 训练适配器"""

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """不剥离任何模块——全量训练 ViT，与 verl_npu 基线对齐"""
        return []

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        """标准 hf_processor——Qwen3.5 的 processor 类名在 verl.utils.hf_processor 白名单内"""
        from verl.utils.hf_processor import hf_processor
        return hf_processor(model_config.local_path, trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        """标准 hf_tokenizer"""
        from verl.utils.hf_tokenizer import hf_tokenizer
        return hf_tokenizer(model_config.local_path, trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_model(cls, module, model_config):
        """打 ViT 位置编码设备补丁

        FSDP2 CPUOffload 下：参数报 cpu、激活在 npu、索引张量建在错的设备上。
        这个补丁打的是 from_pretrained 返回的实例，所以放在 configure_model 里恰好正确。
        """
        module = super().configure_model(module, model_config)
        apply_qwen3_5_vision_device_fix(module)
        return module