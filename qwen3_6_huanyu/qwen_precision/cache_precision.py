"""Common cache algorithms ported from qwen3_6_whisper.cache_precision.

Huanyu owns policy selection, hook installation and bounded execution logs.
The allocation, budget and page-mapping arithmetic matches the source.
"""

import torch


def private_buffers(runner,config):
 from vllm.v1.kv_cache_interface import AttentionSpec,MambaSpec
 specs=runner._get_layer_kv_cache_specs(config)
 if not any(isinstance(s,MambaSpec) for s in specs.values()) or not any(isinstance(s,AttentionSpec) for s in specs.values()):
  raise RuntimeError('Private hybrid cache requires both attention and Mamba layers')
 if not getattr(runner,'_q36_private_cache_budget_applied',False):
  raise RuntimeError('Private cache allocation requires the matching physical-memory budget')
 transfer=runner.vllm_config.kv_transfer_config
 if transfer is not None and getattr(transfer,'kv_connector',None)!='ExampleHiddenStatesConnector':
  raise RuntimeError('Private policy cache has not been validated with KV migration')
 if runner.use_sparse or runner.use_compress:
  raise RuntimeError('Private policy cache expects ordinary dense attention')
 required=sum(config.num_blocks*s.page_size_bytes for s in specs.values())
 if required>runner._q36_private_cache_available_bytes:
  raise RuntimeError(f'Private cache requires {required} bytes, budget is {runner._q36_private_cache_available_bytes}')
 runner.hybrid_with_attn_and_mamba=True
 buffers={name:torch.zeros(config.num_blocks*spec.page_size_bytes,dtype=torch.int8,device=runner.device)
          for name,spec in specs.items()}
 # Separate tensors also prevent block-zeroing in an attention layer from
 # touching a live Mamba state owned by a different cache group.
 if len({v.data_ptr() for v in buffers.values()})!=len(buffers):
  raise RuntimeError('Private cache allocations unexpectedly alias')
 return buffers


def planner_budget(available,groups):
 from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
 totals=[]
 for group in groups:
  spec=group.kv_cache_spec
  totals.append(sum((spec.kv_cache_specs[name] if isinstance(spec,UniformTypeKVCacheSpecs) else spec).page_size_bytes
                    for name in group.layer_names))
 return int(available*max(totals)/sum(totals)) if totals else available


def zero_attention_blocks(runner,block_ids):
 """Expand logical scheduler blocks to the backend's physical cache pages."""
 from vllm.v1.kv_cache_interface import AttentionSpec
 runner._huanyu_zero_execution=None
 if not block_ids or not hasattr(runner,'kv_cache_config'):return
 ids=torch.as_tensor(block_ids,device=runner.device,dtype=torch.long)
 touched=0
 geometry=None
 for group in runner.kv_cache_config.kv_cache_groups:
  spec=group.kv_cache_spec
  if not isinstance(spec,AttentionSpec):continue
  logical=spec.block_size
  for name in group.layer_names:
   for state in runner._iter_kv_cache_states(name):
    # Ascend stores separate K/V as [pages,tokens,heads,dim]. Also support
    # the ordinary combined K/V views without confusing a two-page cache.
    axis=1 if state.ndim==5 and state.shape[0]==2 else 0
    token_axis=2 if state.ndim==5 else 1
    physical=state.shape[token_axis]
    if physical<=0 or logical%physical:
     raise RuntimeError(f'Invalid logical/physical KV block sizes: {logical}/{physical}')
    ratio=logical//physical
    if min(block_ids)<0 or (max(block_ids)+1)*ratio>state.shape[axis]:
     raise RuntimeError('Logical KV block to clear is outside the physical cache')
    pages=(ids[:,None]*ratio+torch.arange(ratio,device=state.device)[None,:]).reshape(-1)
    state.index_fill_(axis,pages,0)
    touched+=1
    if geometry is None:
     geometry=dict(logical=logical,physical=physical,pages_per_block=ratio)
 # Report only after the whole call succeeds, not after the first state when a
 # later state might still fail. Metadata never reads tensor values or syncs.
 if touched:
  runner._huanyu_zero_execution=dict(logical_blocks=len(block_ids),states=touched,**geometry)
