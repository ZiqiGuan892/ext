"""micro_batch["routed_experts"] → 与训练前向 token 布局逐行对齐的 replay targets。

verl v1 引擎里 ``routed_experts`` 是 jagged nested tensor ``[B, j, L, K]``
（``workers/utils/padding.py`` 按与 ``input_ids`` 相同的 unpad indices /
cu_seqlens 构造）。Huanyu 只走 padded 前向（``use_remove_padding=false``）：
``FSDPEngineWithLMHead.prepare_model_inputs`` 把 nested ``input_ids``
``to_padded_tensor`` 成右 pad 的 ``[B, S]``，MoE block 内
``hidden.reshape(-1, H)`` 即 ``[B*S]`` 行主序。本模块按同一规则把每个样本
的有效行放到 ``b*S + [0, len_b)``，pad 位为零行（→ 回退原生路由）。

packed（``use_remove_padding=true``）布局显式拒绝：Huanyu 前向本身不支持。

样本级对齐契约（fail-closed）：vLLM 返回 ``P+R-1`` 行（最后一个采样 token
未前向）。因此一个样本要么**全有记录**（有效行 ≥ len_b-1），要么**全无记录**
（abort 等无 routed_experts 的样本，整体回退原生路由并计数）；部分记录说明
vLLM prompt 与训练 prompt 不等长（如 >30 s 音频的 chunk 标记差异），行已错位。
"""

import torch

from .controller import has_duplicate_rows


def build_padded_targets(routed, input_ids: torch.Tensor, *, num_layers: int, num_experts: int):
    """返回 (targets [B*S, L, K] int64, stats dict)。

    ``routed``：jagged nested ``[B, j, L, K]``（已在 input_ids 同设备）。
    ``input_ids``：模型实际收到的 padded ``[B, S]``。
    """
    if not isinstance(routed, torch.Tensor) or not routed.is_nested:
        raise RuntimeError(
            "expert_replay: micro_batch['routed_experts'] must be the verl v1 jagged nested tensor; "
            f"got {type(routed).__name__} nested={getattr(routed, 'is_nested', None)}"
        )
    if input_ids.dim() != 2:
        raise RuntimeError(
            f"expert_replay: packed input_ids {tuple(input_ids.shape)} unsupported; "
            "Huanyu R3 requires use_remove_padding=false (padded [B, S])"
        )
    values = routed.values()
    if values.dim() != 3 or values.shape[1] != num_layers:
        raise RuntimeError(
            f"expert_replay: routed_experts values {tuple(values.shape)} != [*, {num_layers}, K]; "
            "capture layer count disagrees with the actor's fused MoE blocks"
        )
    batch, seq = input_ids.shape
    lengths = routed.offsets().diff()
    if lengths.numel() != batch:
        raise RuntimeError(f"expert_replay: routed_experts batch {lengths.numel()} != input_ids batch {batch}")
    if int(lengths.max()) > seq:
        raise RuntimeError(f"expert_replay: routed_experts sample length {int(lengths.max())} > padded S={seq}")

    values = values.to(device=input_ids.device, dtype=torch.int64)
    # 有效行 v（属于样本 b、样本内位置 p）→ padded 行 b*S + p。
    sample = torch.repeat_interleave(torch.arange(batch, device=values.device), lengths.to(values.device))
    starts = routed.offsets().to(values.device)[:-1]
    position = torch.arange(values.shape[0], device=values.device) - starts[sample]
    targets = values.new_zeros((batch * seq, num_layers, values.shape[-1]))
    targets[sample * seq + position] = values

    if values.numel():
        low, high = int(values.min()), int(values.max())
        if low < 0 or high >= num_experts:
            raise RuntimeError(
                f"expert_replay: routed expert ids in [{low}, {high}] outside [0, {num_experts}); "
                "rollout returned physical/garbage ids"
            )

    # 样本级对齐：一行在所有层都无重复才算有记录。
    recorded = ~has_duplicate_rows(values).any(dim=-1)  # [nnz]
    recorded_per_sample = torch.zeros(batch, dtype=torch.int64, device=values.device)
    recorded_per_sample.index_add_(0, sample, recorded.to(torch.int64))
    lengths_dev = lengths.to(values.device)
    full = recorded_per_sample >= (lengths_dev - 1).clamp_min(0)
    empty = recorded_per_sample == 0
    partial = ~(full | empty)
    if bool(partial.any()):
        bad = int(partial.nonzero()[0])
        raise RuntimeError(
            f"expert_replay: sample {bad} has {int(recorded_per_sample[bad])} recorded routing rows "
            f"but {int(lengths[bad])} tokens (expected >= {int(lengths[bad]) - 1}); vLLM prompt length "
            "differs from the training prompt — routing rows are shifted"
        )
    stats = {
        "samples": batch,
        "unrecorded_samples": int((empty & (lengths_dev > 1)).sum()),
        "recorded_rows": int(recorded.sum()),
        "tokens": int(lengths.sum()),
    }
    return targets, stats
