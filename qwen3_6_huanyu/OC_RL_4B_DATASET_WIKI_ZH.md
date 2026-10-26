# OC-RL 4B 数据集简要说明

用于多模态强化学习的图文问答数据，原始版本为 `cleaned_rl_dataset_v3`，经 `omni_mix_4b.py` 转成训练所需的 Parquet 格式。以下规模按已确认的 v1 转换记录统计。

## 数据量与长度

| 数据范围 | 样本数 |
|---|---:|
| 原始数据（199 个分片、30 个来源） | 99,342 |
| 清洗后全量（28 个来源） | 90,247 |
| 全量训练集 / 验证集 | 81,222 / 9,025 |
| 全来源调试子集（每来源训练 8 条、验证 2 条） | 224 / 56 |

转换时排除了 9,073 条缺少 `answer.solution` 的记录、21 条含 MPO 图片的记录和 1 条无效问题。训练/验证按关联样本分组后约 9:1 划分，共享图片或来源 ID 的样本保留在同一侧。此前控制台显示的 **224 条属于调试子集**。

单条问题与答案长度可变；现有记录未提供 token 长度的均值、分位数和最大值。训练脚本中的长度上限属于运行配置，不能当作数据实际长度。

## 来源与种类

原始文件位于 `/opt/huawei/dataset/verl_omni_4b_data/OC-RL/cleaned_rl_dataset_v3/data/`。主要来源和任务包括：

| 种类 | 来源示例（非完整清单） |
|---|---|
| 数学、几何与学科解题 | Geo170K、multimodal_math、Chegg、mmk12 |
| 图表、表格与文档问答 | TABMWP、WTQ、DVQA、FigureQA、DocVQA_vr、ArxivQA_pr |
| 视觉问答与推理 | visual_web_instruct、options_53_vr、R-CoT、logical_spatial_shuffled |

清洗后的全量样本均为**图片＋文本问题＋参考答案**，无音频、视频样本。原始数据中的指令遵循来源 `nemotron_if_filtered` 和设计约束来源 `mdif` 因缺少 `solution` 被排除。

## Parquet 字段

原始分片只有 `problem`、`answer`、`data_source`、`images`、`reward_type` 五列；转换后的训练和验证文件统一包含以下八列：

| 字段 | 内容 |
|---|---|
| `data_source` | 原始数据来源名称，如 `Geo170K` |
| `prompt` | `role/content` 消息列表，包含 system 提示及带 `<image>` 标记的用户问题 |
| `images` | 图片列表，元素形如 `{"image": "绝对图片路径"}` |
| `videos` | 空列表 `[]`，保留统一格式 |
| `audios` | 空列表 `[]`，保留统一格式 |
| `ability` | 当前全量数据为 `image` |
| `reward_model` | `style="model"`；`ground_truth` 来自原始 `answer.solution` |
| `extra_info` | 划分、索引、问题 ID、原问题/答案、来源文件及行号、奖励类型、完整答案 JSON、图片哈希及引用索引等溯源信息 |

参考答案用于 GenRM 评分；原始 `word_list`、`weight_list` 等保存在 `extra_info.source_answer_json` 中，当前评分流程未使用这些权重。

当前启动脚本默认读取 `/opt/huawei/dataset/infra_data/data/train.parquet` 和同目录的 `validation.parquet`；迁移后的图片路径前缀为 `/opt/huawei/dataset/infra_data/data/dataset/media/`。实际读取全量还是调试子集，取决于这两个文件的内容。

统计依据：[既有转换记录](D:/verl-omni/docs/oc_rl_4b_server_runbook_zh.md:13)；字段定义：[转换脚本](D:/verl-omni/verl_omni_ext/examples/omni_mix_4b.py:356)。
