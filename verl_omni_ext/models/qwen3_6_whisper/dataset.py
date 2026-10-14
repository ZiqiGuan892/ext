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
"""RL dataset for Qwen3.6-35B-A3B-whisper thinker training.

继承自 MiniCPM-o 5.0 的 dataset 实现，因为：
  - ``processing_minicpmo.py`` 与 MiniCPM-o 5.0 byte-identical
  - processor 类名 ``MiniCPMOProcessor`` 相同
  - chat_template 与 tokenizer 的分工与 5.0 完全相同
  - 视频处理路径也走 ``minicpmo_5_0_video.py``（MiniCPM-o 5.0 把 video 拆成
    frames-as-images 的路径对 Qwen3-VL deepstack visual encoder 同样适用，
    因为 model.visual 同时处理 image 与 video frames）

差异：
  - 预计算 cache key 包含 video path（与 MiniCPM-o 5.0 实现相同，video 走
    ``_load_precomputed(spec.path, cache_dir, "vision")``）
  - 新模型的 video 模态由 ``model.visual`` 直接支持（不再需要 frames-as-images
    兼容路径），但 MiniCPM-o 5.0 的实现已能正确处理 video（拆 frames 走 image
    path），与 vLLM-Omni 侧 ``process_videos`` 行为一致，故直接继承。

Wiring
------
Set both keys; verl-omni's ``get_dataset_class`` delegates to upstream's
loader as soon as ``custom_cls.path`` is present::

    data.custom_cls.path=pkg://verl_omni_ext.models.qwen3_6_whisper.dataset
    data.custom_cls.name=Qwen36WhisperOmniRLHFDataset
"""

import hashlib
import logging
import os

from verl.utils.dataset.rl_dataset import RLHFDataset

# 直接复用 MiniCPM-o 5.0 的 dataset 类，因为 processor 与 video 处理路径完全相同
from verl_omni_ext.models.minicpmo_5_0.dataset import (
    MiniCPMOOmniRLHFDataset as _MiniCPMOOmniRLHFDataset,
    MiniCPMOThinkerRLHFDataset as _MiniCPMOThinkerRLHFDataset,
    _EmbedList,
    _attach_precomputed,
    _load_precomputed,
    _path_hash,
)

logger = logging.getLogger(__name__)

__all__ = ["Qwen36WhisperThinkerRLHFDataset", "Qwen36WhisperOmniRLHFDataset"]


class Qwen36WhisperThinkerRLHFDataset(_MiniCPMOThinkerRLHFDataset):
    """``RLHFDataset`` that measures prompt length with the tokenizer, always.

    直接继承 MiniCPM-o 5.0 的实现，因为：
      - processor 类名相同（``MiniCPMOProcessor``）
      - chat_template 与 tokenizer 的分工相同
      - 数据集过滤到 0 行的检测逻辑相同
      - 多模态编码（PIL RGB + float32 numpy + video frames）相同
    """

    pass


class Qwen36WhisperOmniRLHFDataset(_MiniCPMOOmniRLHFDataset):
    """Omni (image + audio + video) RL dataset for Qwen3.6-whisper thinker.

    直接继承 MiniCPM-o 5.0 的 OmniMix dataset 实现。新模型的 video 模态由
    ``model.visual`` 直接支持，但 MiniCPM-o 5.0 的 frames-as-images 路径与
    vLLM-Omni 侧的 ``process_videos`` 行为一致（均把 video 拆成 N images），
    所以无需改写 ``_process_multi_modal_info``。

    预计算 cache key 与 MiniCPM-o 5.0 实现相同：image/video 走 ``vision``
    subdirectory，audio 走 ``audio`` subdirectory，key 为文件路径 SHA-256。
    """

    pass