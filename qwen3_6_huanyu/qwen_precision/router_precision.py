"""Match the actor's BF16 gate before Ascend's internal MoE routing.

Recent Ascend MoE runners calculate their own FP32 gate and bypass the gate
module's forward. Correct logits at the routed-expert boundary, where the
original BF16 expert inputs and the live, reloadable gate are available.
"""
import os
from types import MethodType

import torch


def attach(routed, gate):
    if getattr(routed, '_q36_actor_router_reference', False):
        return False
    original = routed.forward_impl
    reported = False

    def forward(self, *, hidden_states, router_logits, input_ids=None):
        nonlocal reported
        if hidden_states.dtype != torch.bfloat16 or gate.weight.dtype != torch.bfloat16:
            raise RuntimeError('Actor router reference requires BF16 inputs and gate weights')
        logits, bias = gate(hidden_states)
        if bias is not None:
            logits = logits + bias
        if logits.dtype != torch.bfloat16:
            raise RuntimeError('Actor router reference gate returned non-BF16 logits')
        if not reported:
            reported = True
            print(f'[q36-router] actual routed boundary: incoming={router_logits.dtype}, '
                  f'actor_reference={logits.dtype}, shape={tuple(logits.shape)}', flush=True)
        return original(hidden_states=hidden_states, router_logits=logits, input_ids=input_ids)

    routed.forward_impl = MethodType(forward, routed)
    routed._q36_actor_router_reference = True
    return True


def install_rollout(native):
    if os.environ.get('MINICPMO_HF_ROUTER_REFERENCE', '0') != '1':
        return 0
    language = getattr(native, 'llm', None)
    if language is None:
        language = getattr(getattr(native, 'thinker', None), 'llm', None)
    if language is None:
        raise RuntimeError('Actor router reference requires the policy language model')
    count = 0
    for layer in language.model.layers:
        mlp = layer.mlp
        runner = mlp.experts
        if runner.moe_config.ep_size != 1 or mlp.is_sequence_parallel:
            raise RuntimeError('Actor router reference currently requires EP1 and no sequence parallel MoE')
        count += attach(runner.routed_experts, mlp.gate)
    if count:
        print(f'[q36-router] actor BF16 gate reference installed: {count} MoE layers', flush=True)
    return count