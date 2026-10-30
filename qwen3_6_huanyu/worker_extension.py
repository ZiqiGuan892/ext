"""Mandatory Huanyu IPC binding, resolved by the native worker factory.

Unlike registry-side best-effort imports, a worker_extension_cls import cannot
be skipped while continuing to start that worker. Keep the base __new__, LoRA,
sleep and wake APIs; only the full Huanyu weight-update method is replaced.
"""

import inspect
import os
import socket

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension

from .cache_precision import install_huanyu_cache_precision
from .processor_compat import install_huanyu_processor_compat
from .rollout_precision import install_huanyu_rollout_precision
from .weight_reload import bind_huanyu_weight_reload
from .precision_policy import report_policy


class HuanyuColocateWorkerExtension(vLLMOmniColocateWorkerExtension):
    pass


report_policy("rollout-worker")
bind_huanyu_weight_reload(HuanyuColocateWorkerExtension)
# Some native checkouts do not execute the external-module registry gate in
# the spawned worker. The explicit worker factory must install these too,
# before it constructs the model. Installation errors propagate to startup.
install_huanyu_processor_compat()
install_huanyu_rollout_precision()
install_huanyu_cache_precision()
if os.environ.get("HUANYU_EMBEDDING_AB_DIR", "").strip():
    from .prompt_embedding_probe import install_runtime_probe
    install_runtime_probe()
print(
    f"[huanyu-worker] explicit precision extension selected; host={socket.gethostname()} "
    f"pid={os.getpid()} source={__file__} "
    f"ipc_signature={inspect.signature(vLLMOmniColocateWorkerExtension.update_weights_from_ipc)}",
    flush=True,
)
