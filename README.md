# 多方式手写数字识别 Skill

面向 MNIST 学习与实验的开源 Codex Skill：研究贝叶斯、SVM、KNN 的原理与实现，用 PCA/LDA 做降维对照，并扩展到 MLP、LeNet 风格 CNN。支持从理论文章、代码教程、实验报告和现有工程进入任务。

这是指导 AI 完成研究、实现和实验审查的技能包；创建 Skill 不等于已经训练完所有模型。默认中文讲解，采用 Python、scikit-learn、PyTorch，也可适配已有框架。

## 使用方法

Skill 位于 [`.agents/skills/mnist-multi-method/SKILL.md`](.agents/skills/mnist-multi-method/SKILL.md)，界面信息位于 [`agents/openai.yaml`](.agents/skills/mnist-multi-method/agents/openai.yaml)。在支持项目 Agent Skills 的环境中打开本项目，输入：

```text
$mnist-multi-method
请根据这篇文章，解释贝叶斯、SVM、KNN，并设计原始特征、PCA、LDA
与 MLP/CNN 的 MNIST 对照实验。先交付方案，不启动训练。
文章：<粘贴文章、提供链接或本地文件路径>
```

当前会话尚未发现新 Skill 时，可新建项目会话，或直接说：

```text
读取 .agents/skills/mnist-multi-method/SKILL.md，并按其中要求审查我提供的代码。
```

输入可包含文章/代码、已有数据路径、运行设备、时间预算及目标交付。需要实现或训练时明确提出；Skill 本身没有安装依赖或下载数据的脚本。提供的 `mnist_deep_learning_complete.zip` 是可选现有工程参考，已保留原样。

## 输入输出示例

| 输入 | 预期输出 |
|---|---|
| “解释三类经典算法，并用 PCA/LDA 优化，CPU，只做方案” | 原理与假设、共同划分、经典算法九组 + MLP/CNN 两组、预算和待运行结果表 |
| “文章先合并训练/测试再 PCA，LDA 取 50 维，请检查” | 定位数据泄漏，纠正 10 类 LDA 最多 9 维，给出训练内拟合的 Pipeline 方案 |
| “复用压缩包，补齐经典算法并比较” | 核对现有 MLP/CNN 接口、复用索引、给出缺失模块和公平对比实现；历史结果单列 |

未运行结果的示例（不是实验成绩）：

```json
{"model":"GaussianNB","representation":"pca","n_components":50,"accuracy":null,"macro_f1":null,"status":"not_run"}
```

正式实验应输出配置与划分、模型/预处理工件、结果表、混淆矩阵和各类指标；神经网络另有训练/验证曲线。单图推理保存预测与实际输入预览。文章成绩、附件历史成绩和本次实测分别标记。

## 开发与验证

使用官方 `skill-creator/scripts/init_skill.py` 初始化，未添加 Skill 运行脚本。测试用例见 [`tests/article-cases.md`](tests/article-cases.md)，执行记录见 [`tests/behavior-report.md`](tests/behavior-report.md)，实际回答保存在 [`tests/outputs/`](tests/outputs/)。测试覆盖理论、含错误教程、真实官方深度学习教程、附件项目文章及无关文章边界。

在装有官方 Skill Creator 的环境中运行验证（PowerShell）：

```powershell
python -X utf8 "$env:USERPROFILE/.codex/skills/.system/skill-creator/scripts/quick_validate.py" ".agents/skills/mnist-multi-method"
```

`-X utf8` 避免 Windows 默认 GBK 编码误读中文文件。该验证器检查 Skill 结构与元数据；行为测试和小规模代码检查另行记录，不证明完整 MNIST 实验的准确率。

## 许可

本次新增的 Skill、文档和本地测试材料采用 [MIT License](LICENSE)。用户提供的 ZIP、其中的权重/历史结果、MNIST 数据及外部链接内容不在本次 MIT 授权范围内。原始 ZIP、模型权重和本地运行输出由 `.gitignore` 排除，保留在本地；`tests/outputs/` 中的行为测试记录纳入版本管理。
