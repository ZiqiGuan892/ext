# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""等 D2H 落地，再让 FSDP2 在 host 上重新补齐分片。

这个补丁拦下的 bug
------------------
16 卡、FSDP2、``param_offload=true``。音频编码器里 4 个 ``LayerNorm(40)`` 的权重
在每次权重同步送给 rollout 时，**只有最后一个元素（下标 39）是错的**，而且每次
同步的错值都不一样；同一个编码器里的 ``LayerNorm(20)`` 全程逐字节正确。
（完整证据链见 ``perf_report/analysis_11_14/REPORT.md``。）

决定性的线索是**哪些张量会坏**。FSDP2 对每个参数各自沿 dim0 分片并补齐，16 卡下：

===== ===== ==================== ======
numel chunk 最后一块             结果
===== ===== ==================== ======
40    3     rank13 只装 1 / 3    **坏在 39**
41    3     rank13 只装 2 / 3    **坏在 39、40**
17    2     rank8  只装 1 / 2    **坏在 16**
20    2     10 卡各 2，整除       干净
33    3     11 卡各 3，整除       干净
16    1     16 卡各 1，整除       干净
===== ===== ==================== ======

坏的**当且仅当**最后一块分片装不满、需要补齐。

补齐分片为什么特殊
------------------
``offload_fsdp2_model_to_cpu`` 做的是 ``model.to("cpu", non_blocking=True)``。
对 FSDP2 模块，``FSDPModule._apply`` 的次序是：

1. ``super()._apply(...)`` —— 给**每个**参数排一发异步 D2H（真实模型是 GB 级，
   这些拷贝会在队列里排很久）；
2. 然后逐个 ``FSDPParam.reset_sharded_param()``。

而 ``reset_sharded_param`` 对需要补齐的分片会走（torch
``_fully_shard/_fsdp_param.py:909``）::

    padded_local_tensor = local_tensor.new_zeros(padded_sharded_size)
    padded_local_tensor.narrow(dim=shard_dim, start=0, length=length).copy_(local_tensor)

``local_tensor`` 就是第 1 步那发 D2H 的**目的地**，这里的 ``copy_`` 是一次
**host 侧读**。verl 自己在 ``offload_fsdp2_model_to_cpu`` 的注释里写得很清楚：
“The CPU tensors are not safe for host access until the D2H copy finishes.
A caller that reads the tensors on the host must synchronize first.” —— 而这次
host 读发生在 ``.to()`` **内部**，在 ``.to()`` 返回之后再同步已经来不及。
整除的分片不进这个分支、没有 host 读，所以安然无恙。

于是补齐分片读到的是还没写入的页，``new_zeros`` 之后的 ``copy_`` 把垃圾抄进了
CPU 主副本。这份 CPU 主副本既是下次 ``load_fsdp_model_to_gpu`` 的来源，也是
``get_per_tensor_param`` 导出给 rollout 的来源 —— 所以**trainer 自己的权重和
推理端的权重一起坏**，而且坏完还会被原样存住。

这个补丁做什么
--------------
只在 ``reset_sharded_param`` 真的要做 host 侧补齐时（分片在 CPU 上、且尺寸不等于
补齐后尺寸）先 ``synchronize()`` 一次，然后调用原实现。补齐参数在本模型是个位数，
所以同步次数是个位数；且只要同步过一次，后续的 ``synchronize()`` 在空闲设备上
几乎是空操作。

为什么打在 torch 而不是 verl
----------------------------
L2（第三方）而不是 L3：错的是 torch 这段「``_apply`` 之后在 host 上重新补齐」的
时序假设，而**所有**调用方都踩同一个坑 —— ``offload_fsdp2_model_to_cpu``、
``fsdp2_load_full_state_dict`` 末尾的 ``model.to("cpu", non_blocking=True)``、
以及任何对 FSDP2 模块的 ``.to("cpu")``。改 verl 的某一个 helper 只能堵住其中一处。

另一个等效修复是把 ``offload_fsdp2_model_to_cpu`` 改成 ``non_blocking=False``
（``perf_report/analysis_11_14/code/repro_uneven_shard.py --fix sync_offload``
同样能让复现脚本转干净），但它把整个 offload 串行化，且堵不住 init 那条路径。

真机验证
--------
``repro_uneven_shard.py``（16 卡、不加载任何 checkpoint、约 45 秒）：不打补丁
5 轮里 40/41/17 稳定坏在末尾元素；打上之后 5 轮全干净，20/33/16 两种情况下都干净。
全栈 ``run_wsync_probe.sh`` 复测见 REPORT.md。

关掉
----
``HUANYU_FSDP2_PAD_SYNC=0`` 保留上游行为（= 把 bug 放回来），用于 A/B。
"""

import inspect
import os

from verl_omni_ext._patchkit import idempotent_patch

_ENV = "HUANYU_FSDP2_PAD_SYNC"
_TARGET_MODULE = "torch.distributed.fsdp._fully_shard._fsdp_param"
_TARGET_ATTR = "FSDPParam"
_TARGET_METHOD = "reset_sharded_param"
_PATCH_NAME = "fsdp2_reset_sharded_param_pad_sync"
# 上游那次 host 侧读所在的行（torch 2.9 的行号；源码断言不依赖行号）
_UPSTREAM_SITE = f"{_TARGET_MODULE}.{_TARGET_ATTR}.{_TARGET_METHOD} (new_zeros + narrow().copy_())"

# 原实现里「在 host 上重新补齐」那段的指纹。上游哪天不再这么做，本补丁就没有
# 意义了，必须立刻报出来，而不是继续打一个什么都不保护的 synchronize。
_SOURCE_SENTINELS = ("new_zeros", ".copy_(", "padded_sharded_param_size")

_disabled_notice_printed = False


@idempotent_patch(
    name=_PATCH_NAME,
    target_module=_TARGET_MODULE,
    target_attr=_TARGET_ATTR,
    fingerprint="torch>=2.4,<3 (FSDP2 fully_shard)",
    expected_signature=_TARGET_METHOD,
    probe_signature=True,
)
def _patch_reset_sharded_param(original, __patch_signature__=None):
    """包一层：host 侧补齐之前先等 D2H 落地。"""
    from verl.utils.device import get_torch_device

    # 前置断言 1：原实现确实还在 host 上做补齐拷贝。缺了任何一个指纹，说明
    # 上游改了做法，这个 synchronize 不再对应任何东西。
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as error:
        raise AssertionError(
            f"{_UPSTREAM_SITE} 取不到源码（{error!r}），无法确认它仍在 host 上重新"
            f"补齐分片；拒绝打上一个不知道在保护什么的补丁。"
        ) from error
    missing = [token for token in _SOURCE_SENTINELS if token not in source]
    if missing:
        raise AssertionError(
            f"{_UPSTREAM_SITE} 的实现里找不到 {missing}——上游已经换了补齐分片的"
            f"做法，本补丁语义失效；拒绝打上。"
        )

    # 前置断言 2：判定条件依赖的属性还在。
    if "padded_sharded_param_size" not in source:  # pragma: no cover - 由上面兜住
        raise AssertionError(f"{_UPSTREAM_SITE} 不再暴露 padded_sharded_param_size")

    def reset_sharded_param(self):
        local = getattr(getattr(self, "sharded_param", None), "_local_tensor", None)
        if local is not None and local.device.type == "cpu" and not local.is_meta:
            try:
                needs_host_pad = local.size() != self.padded_sharded_param_size
            except Exception:  # noqa: BLE001
                # 读不到补齐尺寸就按最坏情况处理。这个兜底只会**多**同步，
                # 不会少同步，所以不存在「静默少打一次保护」的失效形态。
                needs_host_pad = True
            if needs_host_pad:
                synchronize = getattr(get_torch_device(), "synchronize", None)
                if synchronize is not None:
                    # 原实现接下来要在 host 上读 local：等它真的写完。
                    synchronize()
        return original(self)

    print(
        "[verl-omni] Patched FSDP2 FSDPParam.reset_sharded_param -> synchronize before the "
        "host-side re-pad. Without it, model.to('cpu', non_blocking=True) on an FSDP2 module "
        "lets the host copy_ read a D2H destination that has not landed, silently corrupting "
        "the tail elements of every unevenly sharded parameter (see analysis_11_14). "
        f"Set {_ENV}=0 to restore stock behaviour. "
        f"[patchkit name={_PATCH_NAME} sig={__patch_signature__}]",
        flush=True,
    )
    return reset_sharded_param


def apply_fsdp2_pad_sync_patch() -> bool:
    """安装补丁。返回 True=打上了/已打过，False=关掉了或此处不适用。

    必须在任何对 FSDP2 模块的 ``.to("cpu")`` 之前生效。包 import 期调用是够早的：
    ``OmniModelConfig.__post_init__`` 经注册表解析适配器时就会 import 本包，远早于
    ``_build_fsdp_module``（与 ``fsdp_cpu_offload_pin`` 同一时序）。
    """
    global _disabled_notice_printed
    if os.environ.get(_ENV, "1") == "0":
        if not _disabled_notice_printed:
            _disabled_notice_printed = True
            print(
                f"[verl-omni] {_ENV}=0：保留上游 FSDP2 的 host 侧补齐时序"
                f"（{_UPSTREAM_SITE}）。非整除分片的尾部元素会在 offload 时读到未落地的"
                "D2H 目的地——这是明确要求的原样行为（A/B 用），不是失败。",
                flush=True,
            )
        return False

    applied = _patch_reset_sharded_param()
    if not applied:
        # strict 模式下装饰器已经 raise（到不了这行）。非 strict 下补一行：
        # 这个 bug 的失效形态是「数字慢慢不对」，没打上必须看得见。
        print(
            f"[verl-omni] Could not patch {_UPSTREAM_SITE} — 本补丁没打上，FSDP2 "
            "非整除分片的尾部元素仍可能在 CPU offload 时被静默写坏。"
            "检查 _patchkit 的 failed 记录里的 reason。",
            flush=True,
        )
    return applied
