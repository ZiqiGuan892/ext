# 主仓库及其他插件训练脚本的显存默认值

本分支直接发布 `qwen3_6_huanyu/`。另外 8 个本地训练脚本不在本分支中，
对应修改保存在 `other-training-memory-defaults.patch`，路径相对于 verl-omni 项目根目录。
它将 Actor rollout 默认显存预算改为 0.6，保留 GenRM 和 AB/serve 设置。

在包含这些脚本的项目根目录使用补丁文件的实际路径，先执行
`git apply --check /path/to/other-training-memory-defaults.patch`，
检查通过后执行 `git apply /path/to/other-training-memory-defaults.patch`。
如果该工作区已经应用过本次修改，无需再次应用。
