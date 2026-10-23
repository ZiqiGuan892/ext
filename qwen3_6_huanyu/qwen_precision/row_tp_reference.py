"""Opt-in actor output projections with rollout TP4 BF16 boundaries."""
import os
from types import MethodType

import torch.nn.functional as F

from .moe_tp_reference import reduce_tp4


def row_forward(self, x):
    width = self.in_features // 4
    if width * 4 != self.in_features or self.bias is not None:
        raise ValueError('TP4 output reference requires a divisible, bias-free projection')
    parts = [F.linear(x[..., rank*width:(rank+1)*width],
                      self.weight[:, rank*width:(rank+1)*width]) for rank in range(4)]
    return reduce_tp4(parts)


def install_actor(model):
    mode = os.environ.get('MINICPMO_ROW_TP_REFERENCE', '0')
    if mode == '0':
        return 0
    if mode != '4':
        raise ValueError('MINICPMO_ROW_TP_REFERENCE supports only 0 or 4')
    if os.environ.get('HCCL_OP_EXPANSION_MODE') != 'AIV' or os.environ.get('HCCL_DETERMINISTIC') != 'strict':
        raise ValueError('TP4 output reference requires the measured strict AIV reduction')
    import torch.nn as nn
    count = 0
    for name, child in model.named_modules():
        if (isinstance(child, nn.Linear) and name.endswith(('linear_attn.out_proj', 'self_attn.o_proj'))
                and not getattr(child, '_q36_row_tp_reference', False)):
            if child.in_features % 4 or child.bias is not None:
                raise ValueError(f'Unsupported TP4 output projection: {name}')
            child._q36_original_row_forward = child.forward
            child.forward = MethodType(row_forward, child)
            child._q36_row_tp_reference = True
            count += 1
    print(f'[q36-precision] actor TP4 output projection reference: {count} layers', flush=True)
    return count