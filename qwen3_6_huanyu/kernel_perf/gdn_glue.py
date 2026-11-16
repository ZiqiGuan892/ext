"""GDN 融合管道的辅助算子（l2norm / chunk cumsum / kkt / solve_tril）。

最初 vendored 的 flash-linear-attention-npu triton 实现在本机
triton-ascend 3.5.0 上会踩坏后续自定义 AscendC 算子（507015 aicore
exception，"scalar access out of bounds"）。根因经逐实验定位：vendored
kernel 使用的 ``tl.make_block_ptr``（带 boundary_check）在该版本生成越界
访存——同一数学改写为掩码指针访存后数值正确且不再影响后续算子（内置
torch_npu 算子自始至终不受影响）。

本模块是修复后的实现：
- l2norm fwd/bwd：triton kernel，纯掩码指针访存，无 make_block_ptr、无
  autotune（固定单 config，杜绝 bench 期间的异常配置）；
- chunk_local_cumsum / chunk_scaled_dot_kkt / solve_tril：torch 内置算子
  向量化实现（cumsum、bmm、分块倍增求逆），与 HF 参考同结构、无 python
  逐 chunk 循环。

布局约定与上游 flash 管道一致：
- l2norm 输入任意形状，沿最后一维归一化，返回 (out, rstd)；
- chunk_local_cumsum：g [B,T,H]，chunk 内（反）累加，输出同形；
- chunk_scaled_dot_kkt_fwd：k [B,H,T,K]，g（已 chunk 累加）[B,T,H]，
  beta [B,T,H] → A [B,T,H,CS] fp32，严格下三角；
- solve_tril：A [B,T,H,CS] 严格下三角 → (I+A)^{-1}，同形。

仅支持 uniform（cu_seqlens=None，每条 batch 行等长）场景；Huanyu padded
训练（use_remove_padding=false）与补齐后的 fused 前向均属此类。varlen
显式 raise，不静默降级。

CPU import 契约：模块级只 import torch；triton 延迟到 l2norm 首次调用。
"""

import torch

# triton/tl 以模块全局惰性赋值：@triton.jit 编译期从模块 __globals__ 解析 tl，
# 不能用函数内局部 import（会 NameError）；CPU 环境不触发 _ensure_triton，
# 保持 import 本模块无 NPU 依赖的契约。
triton = None
tl = None

_BT = 32  # l2norm kernel 行块（固定 config，不用 autotune）
_NUM_WARPS = 4


def _ensure_triton():
    global triton, tl
    if triton is None:
        import triton as _triton
        import triton.language as _tl

        triton, tl = _triton, _tl


_l2norm_kernels = {}


def _get_l2norm_kernels():
    if _l2norm_kernels:
        return _l2norm_kernels
    _ensure_triton()

    @triton.jit
    def _l2norm_fwd_kernel(x, y, rstd, eps, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BT + tl.arange(0, BT)
        cols = tl.arange(0, BD)
        mask = (rows < T)[:, None] & (cols < D)[None, :]
        x_ptr = x + rows[:, None] * D + cols[None, :]
        b_x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)
        b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
        tl.store(y + rows[:, None] * D + cols[None, :],
                 (b_x * b_rstd[:, None]).to(y.dtype.element_ty), mask=mask)
        tl.store(rstd + rows, b_rstd, mask=rows < T)

    @triton.jit
    def _l2norm_bwd_kernel(y, rstd, dy, dx, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BT + tl.arange(0, BT)
        cols = tl.arange(0, BD)
        mask = (rows < T)[:, None] & (cols < D)[None, :]
        b_y = tl.load(y + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        b_dy = tl.load(dy + rows[:, None] * D + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        b_rstd = tl.load(rstd + rows, mask=rows < T, other=0.0).to(tl.float32)
        # y = x * rstd → dx = rstd * (dy - y * sum(dy * y))
        b_dx = b_rstd[:, None] * (b_dy - b_y * tl.sum(b_dy * b_y, 1)[:, None])
        tl.store(dx + rows[:, None] * D + cols[None, :],
                 b_dx.to(dx.dtype.element_ty), mask=mask)

    _l2norm_kernels["fwd"] = _l2norm_fwd_kernel
    _l2norm_kernels["bwd"] = _l2norm_bwd_kernel
    return _l2norm_kernels


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6):
    """沿最后一维 l2 归一化，返回 (out, rstd fp32)。布局与上游一致。"""
    _ensure_triton()
    kernel = _get_l2norm_kernels()["fwd"]
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    y = torch.empty_like(x2)
    T, D = x2.shape
    BD = triton.next_power_of_2(D)
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    kernel[(triton.cdiv(T, _BT),)](x2, y, rstd, eps, T, D=D, BD=BD, BT=_BT,
                                   num_warps=_NUM_WARPS)
    return y.view(shape), rstd.view(shape[:-1])


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    """l2norm 反向：dx = rstd * (dy - y * sum(dy * y))。"""
    _ensure_triton()
    kernel = _get_l2norm_kernels()["bwd"]
    shape = y.shape
    y2 = y.reshape(-1, shape[-1])
    dy2 = dy.reshape(-1, shape[-1])
    dx = torch.empty_like(y2)
    T, D = y2.shape
    BD = triton.next_power_of_2(D)
    kernel[(triton.cdiv(T, _BT),)](y2, rstd.reshape(-1), dy2, dx, T, D=D, BD=BD,
                                   BT=_BT, num_warps=_NUM_WARPS)
    return dx.view(shape)


class _L2NormFn(torch.autograd.Function):
    """triton l2norm 的 autograd 包装（raw kernel 无梯度，必须经 Function）。"""

    @staticmethod
    def forward(ctx, x, eps):
        y, rstd = l2norm_fwd(x, eps)
        ctx.save_for_backward(y, rstd)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        return l2norm_bwd(y, rstd, dy.contiguous()), None


def l2norm_apply(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """可微分的 l2norm（训练路径必须用这个，raw l2norm_fwd 不连梯度）。"""
    return _L2NormFn.apply(x, eps)


def chunk_local_cumsum(g: torch.Tensor, chunk_size: int, reverse: bool = False,
                       cu_seqlens=None, chunk_indices_out=None, head_first: bool = False) -> torch.Tensor:
    """chunk 内（反向）累加。g [B,T,H]，输出同形同 dtype。"""
    if cu_seqlens is not None:
        raise NotImplementedError("gdn_glue 仅支持 uniform（cu_seqlens=None）场景")
    if head_first:
        raise NotImplementedError("gdn_glue 不支持 head_first 布局")
    B, T, H = g.shape
    if T % chunk_size:
        raise ValueError(f"T={T} 必须是 chunk_size={chunk_size} 的整数倍（调用方负责补齐）")
    gc = g.reshape(B, T // chunk_size, chunk_size, H)
    if reverse:
        gc = gc.flip(dims=[2]).cumsum(dim=2).flip(dims=[2])
    else:
        gc = gc.cumsum(dim=2)
    return gc.reshape(B, T, H)


def chunk_scaled_dot_kkt_fwd(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
                             cu_seqlens=None, chunk_indices=None, chunk_size: int = 64,
                             output_dtype=torch.float32) -> torch.Tensor:
    """A = tril_{-1}(beta * (K K^T) * exp(g_i - g_j))，逐 chunk，输出 [B,T,H,CS]。

    k [B,H,T,K]；g（已 chunk 累加）[B,T,H] fp32；beta [B,T,H]。
    与 vendored triton 语义一致：exp 差值 clamp 到 [-50, 50]，严格下三角。
    """
    if cu_seqlens is not None:
        raise NotImplementedError("gdn_glue 仅支持 uniform（cu_seqlens=None）场景")
    B, H, T, K = k.shape
    if T % chunk_size:
        raise ValueError(f"T={T} 必须是 chunk_size={chunk_size} 的整数倍（调用方负责补齐）")
    NT = T // chunk_size
    kc = k.float().reshape(B, H, NT, chunk_size, K)
    gc = g.float().transpose(1, 2).reshape(B, H, NT, chunk_size)
    bc = beta.float().transpose(1, 2).reshape(B, H, NT, chunk_size)
    dot = kc @ kc.transpose(-1, -2)
    decay = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).clamp(-50.0, 50.0).exp()
    A = dot * decay * bc.unsqueeze(-1)
    A = A.tril(-1)
    return A.reshape(B, H, T, chunk_size).transpose(1, 2).contiguous().to(output_dtype)


# 分块求逆的基块边长。**不要**调到 CS（=64）：整块幂零展开在真实数据上
# 会把训练打成 NaN，根因与取舍见 README「WY 求逆的数值稳定性」。
_SOLVE_TRIL_BASE = 8


def _block_masks(side: int, base: int, like: torch.Tensor):
    """分块倍增用的两组掩码：基块的块对角、各级待合并块对的左下角块。

    全程在整块 [*,side,side] 上做，不切分块视图——掩码乘法比
    ``unflatten/diagonal/movedim`` + 逐级 ``cat`` 重拼便宜得多（NPU 实测
    2048 chunk 下 1.8ms → 0.8ms），代价是 matmul 按整块 64×64 计 flop。
    """
    index = torch.arange(side, device=like.device)
    diagonal = (index[:, None] // base == index[None, :] // base).to(like.dtype)
    merges = []
    width = base
    while width < side:
        row, col = index[:, None] // width, index[None, :] // width
        # 块 (2p+1, 2p)：分块次对角线上按步长 2 取，即合并第 p 对的 A21。
        merges.append(((row == col + 1) & (col % 2 == 0)).to(like.dtype))
        width *= 2
    return diagonal, merges


def _solve_tril_squares(squares: torch.Tensor) -> torch.Tensor:
    """[N,CS,CS] 严格下三角 → (I+A)^{-1}，FP32。两级分块，见 solve_tril_npu。"""
    side = squares.shape[-1]
    base = min(_SOLVE_TRIL_BASE, side)
    diagonal, merges = _block_masks(side, base, squares)
    eye = torch.eye(side, device=squares.device, dtype=squares.dtype)

    # 一级：边长 base 的块对角部分。严格下三角故块内 A^base=0，
    # (I+A)^{-1} = Σ(-A)^i = (I-A)(I+A^2)...(I+A^{base/2})，2·(log2(base)-1) 个
    # matmul。base 小是关键：中间幂次元素量级随 base 指数增长而结果恒为
    # O(1)，差额全靠浮点抵消吃掉——base=64 时 A^32 可达 1e18，fp32 一位
    # 有效数字都剩不下。
    nilpotent = squares * diagonal
    inverse = eye - nilpotent
    power = nilpotent
    exponent = 1
    while exponent * 2 < base:
        power = power @ power
        exponent *= 2
        inverse = inverse @ (eye + power)

    # 二级：2×2 分块倍增。[[A11,0],[A21,A22]]^{-1} = [[T11,0],[-T22·A21·T11,T22]]，
    # 即在已算好的块对角 T 上减去 T·A21·T，每级 2 个 matmul。
    for merge in merges:
        inverse = inverse - inverse @ ((squares * merge) @ inverse)
    return inverse


class _SolveTrilFn(torch.autograd.Function):
    """分块倍增求逆 + 闭式反向。

    T=(I+A)^{-1} 有 dT = -T·dA·T，反向因此是
    ``dA = tril(-(T^T·Ḡ·T^T), -1)``——存下 T 就只剩两个 matmul，不必保存前向
    的 10 个中间量，也把倍增递归整个移出反向图。与 MindSpeed wy_fast.py 的
    ``-(A·dA·A)``（掩码后）、CANN ``prepare_wy_repr_bwd_da`` 同式。

    掩掉非严格下三角：A 由 chunk_scaled_dot_kkt_fwd 的 ``.tril(-1)`` 产出，
    只有严格下三角是自由变量，上游反向本来也会把其余位置清零。
    """

    @staticmethod
    def forward(ctx, squares):
        inverse = _solve_tril_squares(squares)
        ctx.save_for_backward(inverse)
        return inverse

    @staticmethod
    def backward(ctx, grad_output):
        inverse, = ctx.saved_tensors
        # 反向在前向的 autocast 作用域之外执行，显式关掉以保证 FP32
        # （与 rms_norm.py 的自定义 Function 同约定）。
        with torch.autocast(device_type=inverse.device.type, enabled=False):
            transposed = inverse.transpose(-1, -2).float()
            grad = -(transposed @ grad_output.float() @ transposed)
        return grad.tril(-1)


def solve_tril_npu(A: torch.Tensor, cu_seqlens=None, chunk_indices_out=None,
                   output_dtype=torch.float32) -> torch.Tensor:
    """逐 chunk 求 (I + A)^{-1}（A 严格下三角），FP32 计算，输出与 A 同形。

    不用 torch.linalg.solve_triangular——其 NPU 实现在 [B*H*NT,64,64] 上实测
    ~107ms（2048 token）。改为两级分块：

    1. ``_SOLVE_TRIL_BASE`` 边长的对角块用幂零展开求逆（全 batch 并行）；
    2. 2×2 分块倍增拼到 CS：[[A11,0],[A21,A22]] 的逆是
       [[T11,0],[-T22·A21·T11, T22]]，每级 2 个 batched matmul。

    两级都以掩码在整块 [N,CS,CS] 上做，不切分块视图（见 _block_masks）。
    总 matmul 数与基块无关，恒为 2·log2(CS)（CS=64 → 10 个），与整块幂零
    展开相同，但中间量级受基块边长封顶，不再有灾难性抵消。基块边长对精度/
    tile 开销的取舍见 README；fp32 的 Ascend fractal 是 16×8，故 8 最省。

    反向走 ``_SolveTrilFn`` 的闭式解，不穿过倍增递归。
    """
    if cu_seqlens is not None:
        raise NotImplementedError("gdn_glue 仅支持 uniform（cu_seqlens=None）场景")
    B, T, H, CS = A.shape
    if T % CS:
        raise ValueError(f"T={T} 必须是 chunk 边长 CS={CS} 的整数倍")
    if CS & (CS - 1):
        raise ValueError(f"chunk 边长 CS={CS} 必须是 2 的幂（分块倍增求逆约定）")
    NT = T // CS
    Ac = A.transpose(1, 2).reshape(B * H * NT, CS, CS).float()
    result = _SolveTrilFn.apply(Ac)
    return result.reshape(B, H, T, CS).transpose(1, 2).contiguous().to(output_dtype)
