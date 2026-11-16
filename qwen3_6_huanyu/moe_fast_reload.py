"""O5：把融合 MoE 专家权重按「整张 3D full-load」一次灌入 vLLM 的处理后布局。

vLLM ``RoutedExperts.load_weights`` 对每个 3D 融合张量会 ``unbind`` 成逐专家
2D 张量再逐个调用 ``weight_loader``：40 层 × 256 专家 × 3 shard = 30720 次/卡；
layerwise reload 下每次调用还会在 meta 上先跑一遍原 loader 计数，再在 finalize
时重放一遍。而 ``weight_loader`` 的 ``full_load``（``loaded_weight.ndim==3``）分支
本就支持整张专家维一次写入——两条路径对 unquantized / is_act_and_mul / 无 EP 的层
**逐字节等价**（见仓库 CPU 测试与 on-device VERIFY）。本模块复刻 ``load_weights``
的 ``is_fused`` 分支但跳过 unbind：3 次/层。

**命名空间**：来料名是 actor checkpoint 命名空间（``model.language_model.layers.N.
mlp.experts.gate_up_proj``），vLLM 模块在 rollout 命名空间（``thinker.llm.model.
layers.N.mlp.experts``，经 omni 模型 ``load_weights`` 的前缀重映射后才派发）。两者
用 :class:`KeyScheme` 对齐到同一个键，再把名字改写到模块的 ``layer_name`` 下——
等价于 vLLM 顶层前缀映射 + ``RoutedExperts.load_weights`` 的 ``qual_name`` 拼接。

任何不满足前置条件的张量/模块回落 ``model.load_weights``（始终正确）；回落被计数，
``strict=True`` 时直接 raise（静默回落曾让 O5 在 11-12/11-13 两轮实测中零生效）。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Callable, Hashable

_LAYER_TAIL_RE = re.compile(r"^(?:.*\.)?layers\.(\d+)\.(.+)$")
_FUSED_SUFFIXES = ("gate_up_proj", "down_proj", "w13", "w2", "w13_weight", "w2_weight")
_REQUIRED_ATTRS = ("w13_weight", "w2_weight", "weight_loader", "moe_config",
                   "layer_name", "get_expert_mapping", "_orient_fused_weight")


class MoeFastFallbackError(RuntimeError):
    """strict 模式下，融合专家张量本应走快路径却回落。"""


def ineligibility_reason(module) -> str | None:
    """None 表示可走快路径；否则返回第一条不满足的前置条件（用于诊断计数）。"""
    missing = [attr for attr in _REQUIRED_ATTRS if not hasattr(module, attr)]
    if missing:
        return f"missing:{missing[0]}"
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None or "Unquantized" not in type(quant_method).__name__:
        return f"quant_method:{type(quant_method).__name__}"
    moe_config = module.moe_config
    if not getattr(moe_config, "is_act_and_mul", False):
        return "not_act_and_mul"
    parallel = getattr(moe_config, "moe_parallel_config", None)
    if parallel is not None and getattr(parallel, "ep_size", 1) != 1:
        return f"ep_size:{getattr(parallel, 'ep_size', None)}"
    manager = getattr(module, "expert_map_manager", None)
    if manager is not None and getattr(manager, "num_fused_shared_experts", 0) != 0:
        return "fused_shared_experts"
    quant_config = getattr(module, "quant_config", None)
    if quant_config is not None and callable(getattr(quant_config, "get_name", None)):
        if quant_config.get_name() == "gpt_oss_mxfp4":
            return "gpt_oss_mxfp4"
    return None


def _module_is_eligible(module) -> bool:
    return ineligibility_reason(module) is None


def _is_candidate(module) -> bool:
    """看起来是融合专家模块（有 w13/w2）——用于区分「不合格」与「根本不是专家层」。"""
    return hasattr(module, "w13_weight") and hasattr(module, "w2_weight")


# ---- 命名空间对齐 -----------------------------------------------------------

@dataclass(frozen=True)
class KeyScheme:
    """把 vLLM 模块路径与 checkpoint 张量名映射到同一个键。

    ``module_key(path)`` → 键或 None；``tensor_key(name)`` → (键, 参数后缀) 或 None。
    """

    name: str
    module_key: Callable[[str], Hashable | None]
    tensor_key: Callable[[str], tuple[Hashable, str] | None]


def _layer_tail_module_key(path):
    match = _LAYER_TAIL_RE.match(path)
    return (int(match.group(1)), match.group(2)) if match else None


def _layer_tail_tensor_key(name):
    match = _LAYER_TAIL_RE.match(name)
    if match is None:
        return None
    module_tail, _, suffix = match.group(2).rpartition(".")
    if not module_tail:
        return None
    return (int(match.group(1)), module_tail), suffix


LAYER_TAIL = KeyScheme("layer_tail", _layer_tail_module_key, _layer_tail_tensor_key)


def prefix_map_scheme(pairs, name="prefix_map") -> KeyScheme:
    """给前缀映射明确、层号不唯一的模型用：checkpoint 前缀 → 模块路径前缀。"""
    pairs = tuple(pairs)

    def tensor_key(tensor_name):
        for src, dst in pairs:
            if tensor_name.startswith(src):
                module_path, _, suffix = (dst + tensor_name[len(src):]).rpartition(".")
                return module_path, suffix
        return None

    return KeyScheme(name, lambda path: path, tensor_key)


KEY_SCHEMES: dict[str, KeyScheme] = {LAYER_TAIL.name: LAYER_TAIL}


class ExpertModuleIndex:
    """合格专家模块的键 → 模块索引，附带不合格原因与歧义键诊断。"""

    def __init__(self, model, scheme: KeyScheme = LAYER_TAIL):
        self.scheme = scheme
        self.modules: dict[Hashable, object] = {}
        self.ineligible: Counter = Counter()
        self.ambiguous: set = set()
        self.candidates = 0
        for path, module in model.named_modules():
            if not _is_candidate(module):
                continue
            self.candidates += 1
            reason = ineligibility_reason(module)
            if reason is not None:
                self.ineligible[reason] += 1
                continue
            key = scheme.module_key(getattr(module, "layer_name", None) or path)
            if key is None:
                self.ineligible["unkeyed_path"] += 1
                continue
            # 同键多模块（如 MTP 与主干共用 layers.N.）：无法确定归属，整键弃用。
            if key in self.ambiguous or (key in self.modules and self.modules[key] is not module):
                self.ambiguous.add(key)
                self.modules.pop(key, None)
                continue
            self.modules[key] = module

    def __bool__(self):
        return bool(self.modules)

    def resolve(self, name: str):
        """返回 (module, 改写到模块命名空间的 qual_name)；无法对齐时返回 None。"""
        keyed = self.scheme.tensor_key(name)
        if keyed is None:
            return None
        key, suffix = keyed
        module = self.modules.get(key)
        if module is None:
            return None
        return module, f"{module.layer_name}.{suffix}"

    def describe(self) -> dict:
        return {
            "scheme": self.scheme.name,
            "indexed_modules": len(self.modules),
            "candidate_modules": self.candidates,
            "ineligible": dict(self.ineligible),
            "ambiguous_keys": len(self.ambiguous),
        }


def build_fast_moe_index(model, scheme: KeyScheme = LAYER_TAIL) -> ExpertModuleIndex:
    return ExpertModuleIndex(model, scheme)


# ---- 快路径 -----------------------------------------------------------------

def _is_fused_tensor(name: str, tensor) -> bool:
    if not (hasattr(tensor, "dim") and tensor.dim() == 3):
        return False
    leaf = name.removesuffix(".weight").rpartition(".")[2]
    return leaf in _FUSED_SUFFIXES


def _fast_load_one(module, qual_name: str, tensor) -> int:
    """复刻 load_weights 的 is_fused 分支，但每 shard 只调一次 weight_loader。

    ``qual_name`` 必须已在模块命名空间（``f"{layer_name}.{suffix}"``）。返回实际的
    weight_loader 调用数；0 表示需回落（未改动任何参数前返回）。
    """
    if tensor.dim() != 3:
        return 0
    mapping = module.get_expert_mapping(include_fused=True)
    matched = False
    plan = []  # 先全部校验再统一写入，保证全有或全无
    for param_name, weight_name, expert_id, shard_id in mapping:
        if weight_name not in qual_name:
            if matched:
                break
            continue
        matched = True
        resolved_name = qual_name.replace(weight_name, param_name)
        param_attr = resolved_name.removeprefix(f"{module.layer_name}.")
        param = getattr(module, param_attr, None)
        if param is None:
            return 0
        uses_weight_layout = "scale" not in resolved_name
        fused = module._orient_fused_weight(
            tensor, module.is_fused_checkpoint_transposed and uses_weight_layout
        )
        if shard_id in ("w1", "w3"):
            experts_shard = fused.chunk(2, dim=1)[expert_id]
        else:
            experts_shard = fused
        if experts_shard.shape[0] != param.data.shape[0]:
            return 0  # EP / 专家数不匹配 → 回落
        plan.append((param, resolved_name, shard_id, experts_shard))
    if not matched or not plan:
        return 0
    for param, resolved_name, shard_id, experts_shard in plan:
        ok = param.weight_loader(
            param=param,
            loaded_weight=experts_shard,
            weight_name=resolved_name,
            shard_id=shard_id,
            expert_id=0,  # full_load 下忽略，仅为签名完整
            return_success=True,
        )
        if not ok:
            raise RuntimeError(
                f"moe_fast full-load weight_loader rejected {resolved_name} shard={shard_id}; "
                "preconditions passed but copy failed — refusing silent fallback double-load"
            )
    return len(plan)


def make_moe_fast_consumer(model, *, scheme: KeyScheme = LAYER_TAIL, strict: bool = False):
    """构造 bucket 级 consumer：消化合格的融合专家张量，其余原样返回。

    返回 ``(consume, stats)``：``consume(items) -> remaining``。``stats`` 累计
    ``fast_tensors`` / ``fast_calls`` / ``fallback_fused_tensors`` 与索引诊断。
    索引为空时 consume 为 None；``strict`` 下索引为空、有不合格专家模块、或任何
    融合张量回落都 raise :class:`MoeFastFallbackError`。
    """
    index = build_fast_moe_index(model, scheme)
    stats = {"fast_tensors": 0, "fast_calls": 0, "fallback_fused_tensors": 0, **index.describe()}
    if strict and (not index or index.ineligible or index.ambiguous):
        raise MoeFastFallbackError(f"moe_fast index unusable under strict mode: {index.describe()}")
    if not index:
        return None, stats

    def consume(items):
        remaining = []
        for name, tensor in items:
            if not _is_fused_tensor(name, tensor):
                remaining.append((name, tensor))
                continue
            resolved = index.resolve(name)
            calls = _fast_load_one(resolved[0], resolved[1], tensor) if resolved else 0
            if not calls:
                stats["fallback_fused_tensors"] += 1
                if strict:
                    reason = "unresolved name" if resolved is None else "fast-load preconditions failed"
                    raise MoeFastFallbackError(
                        f"moe_fast fallback for {name} ({reason}); index={index.describe()}"
                    )
                remaining.append((name, tensor))
                continue
            stats["fast_tensors"] += 1
            stats["fast_calls"] += calls
        return remaining

    return consume, stats
