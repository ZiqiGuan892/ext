"""Opt-in native bilinear arithmetic in the actual frozen HF vision forward."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
from types import MethodType

import torch


def native_bilinear(grid_thw, side, merge, dtype, device):
    indices, weights = [], []
    for t,h,w in grid_thw.tolist():
        hg=torch.linspace(0,side-1,h,device=device,dtype=torch.float32)
        wg=torch.linspace(0,side-1,w,device=device,dtype=torch.float32)
        hf,wf=hg.long(),wg.long();hc=(hf+1).clamp(max=side-1);wc=(wf+1).clamp(max=side-1)
        dh,dw=torch.meshgrid(hg-hf,wg-wf,indexing='ij')
        hfg,wfg=torch.meshgrid(hf,wf,indexing='ij');hcg,wcg=torch.meshgrid(hc,wc,indexing='ij')
        w11=dh*dw;w10=dh-w11;w01=dw-w11;w00=1-dh-w01
        idx=torch.stack((hfg*side+wfg,hfg*side+wcg,hcg*side+wfg,hcg*side+wcg)).reshape(4,-1)
        coef=torch.stack((w00,w01,w10,w11)).reshape(4,-1)
        order=torch.arange(h*w,device=device).reshape(h//merge,merge,w//merge,merge).permute(0,2,1,3).reshape(-1)
        indices.append(idx[:,order].repeat(1,t));weights.append(coef[:,order].repeat(1,t).to(dtype))
    return torch.cat(indices,dim=1),torch.cat(weights,dim=1)


def prepare_helper():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
    if hasattr(hf,'_q36_vision_position_scope'):
        return hf._q36_vision_position_scope
    scope=ContextVar('q36_vision_position_scope',default=None)
    original=hf.get_vision_bilinear_indices_and_weights
    def helper(grid_thw,num_grid_per_side,spatial_merge_size,kwargs=None):
        active=scope.get()
        if active is None:
            return original(grid_thw,num_grid_per_side,spatial_merge_size,kwargs)
        dtype,device,owner=active
        owner._q36_position_helper_calls=getattr(owner,'_q36_position_helper_calls',0)+1
        if getattr(owner,'_q36_native_positions_disabled',False):
            return original(grid_thw,num_grid_per_side,spatial_merge_size,kwargs)
        owner._q36_native_position_calls=getattr(owner,'_q36_native_position_calls',0)+1
        if kwargs is not None:
            kwargs.pop('bilinear_indices',None);kwargs.pop('bilinear_weights',None)
        return native_bilinear(grid_thw,num_grid_per_side,spatial_merge_size,dtype,device)
    hf._q36_vision_position_scope=scope
    hf.get_vision_bilinear_indices_and_weights=helper
    return scope


@contextmanager
def native_position_scope(visual):
    scope=prepare_helper();weight=visual.pos_embed.weight
    token=scope.set((weight.dtype,weight.device,visual))
    try:
        yield
    finally:
        scope.reset(token)


def install_actor(model):
    if os.environ.get('MINICPMO_VISION_POSITION_REFERENCE','0')!='1':
        return False
    visual=getattr(getattr(model,'model',None),'visual',None)
    if visual is None or getattr(visual,'_q36_native_positions',False):
        return False
    if any(p.requires_grad for p in visual.parameters()):
        raise ValueError('Native vision position reference requires a frozen tower')
    prepare_helper()
    original=visual.forward
    @wraps(getattr(original,'__func__',original))
    def forward(self,*args,**kwargs):
        before=getattr(self,'_q36_native_position_calls',0)
        with native_position_scope(self):
            output=original(*args,**kwargs)
        if not getattr(self,'_q36_native_positions_disabled',False) and getattr(self,'_q36_native_position_calls',0)==before:
            raise RuntimeError('Actual vision forward did not call the interpolation helper')
        return output
    visual.forward=MethodType(forward,visual)
    visual._q36_native_positions=True
    print('[q36-precision] frozen actor actual bilinear helper uses native BF16 boundaries',flush=True)
    return True