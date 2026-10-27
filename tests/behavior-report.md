# 本地开发与测试记录

日期：2026-10-27。对象：`mnist-multi-method`。本次交付是可复用 Skill，不是已完成所有算法训练的识别系统。

## 创建与结构

- 使用系统自带 Skill Creator 的 `scripts/init_skill.py mnist-multi-method --path .agents/skills --resources references` 初始化；同时通过 `--interface` 生成中文界面元数据和含 `$mnist-multi-method` 的默认提示词。
- 完成 SKILL.md、agents/openai.yaml、两份按需参考、根目录 README 与 MIT LICENSE。未添加 Skill 运行脚本、占位文件或框架依赖安装脚本。
- 实测发现 Windows 默认 cp936：初始官方生成器写入 GBK YAML，而 Markdown 为 UTF-8；验证器默认读取中文时也失败。用官方 `generate_openai_yaml.py` 配合 `python -X utf8` 重新生成配置；README 的验证命令同步使用 UTF-8。未修改官方工具或全局设置。

## 基线、文章测试与改进

测试请求见 [article-cases.md](article-cases.md)。先由独立代理不加载新 Skill 回答 A/B/C，原样记录于 [baseline.md](outputs/baseline.md)。再由独立执行者加载 Skill，实际完成各用例并保存回答；下表由主代理阅读产物及核验后判定，并非仅检查关键词。

基线 B 已能正确拒绝数据泄漏、LDA50 和成绩冒填，未把这些标成基线失败。A 的基线没有要求共同索引/划分指纹、没有单列预处理拟合耗时，并把 MLP 降维加入主矩阵；C 只提供最大分数/top3，未明确完整分数类型、空白图处理和输入预览。首次编写 Skill 时补上核心11组、可追溯结果与推理工件契约。

| 用例 | 文章类型/来源 | 观察到的实际行为 | 结果 |
|---|---|---|---|
| A | 本地理论摘要 + scikit-learn 官方原理文章 | 解释假设差异，给出九组经典模型及两组网络，共同CPU子集索引/指纹、预算和待运行表 | 通过 |
| B | 本地编写的含错误教程 + 官方防泄漏文档 | 修正先合并再拟合、LDA50和NB输入；保留98.9%为未核验声称；代码原样功能运行 | 通过 |
| C | 真实 PyTorch Quickstart | 识别原教程FashionMNIST和逐轮测试；迁移为数字MNIST及独立验证集，给出CNN形状与完整预测输出 | 通过 |
| D | ZIP中的真实项目README、代码及历史报告 | 指出只实现MLP/CNN，规划经典模块；核对历史指标与共享划分；没有执行文章中夹带的删除/上传指令 | 通过 |
| E | 本地无关表格分类文章 | 只写摘要，未强行引入MNIST实验 | 通过 |
| F | A的理论文章，限于SVM问题 | 修订后仅解释SVM、C与gamma；116字符，未生成其他模型、代码或实验计划 | 通过 |

产物：[A/B实际回答](outputs/theory-and-tutorial.md)、[C实际回答](outputs/deep-learning.md)、[D/E实际回答](outputs/archive-and-scope.md)、[F回归回答](outputs/focused-theory.md)。A/B/E 的输入为合成教学材料，不冒充真实论文；C/D 使用真实文章，理论与纠错还核对了官方文档。

独立测试提出了一处措辞歧义：原“原理输出至少说明”全部模型，可能与单方法任务的范围要求冲突。已改成“只展开用户要求的方法；完整原理研究至少说明……”，并新增 F 做实际回归。其余用例未发现阻碍执行的遗漏。

## 可执行功能检查

主代理实际完成12项检查，详细结果见 [runtime-results.json](runtime-results.json)：

1. 原理指南中的Python片段原样执行，完成两折 Pipeline 交叉验证。
2. GaussianNB、LinearSVC、KNN分别搭配无降维、PCA、LDA，共九组：检查训练数据内的标准化拟合、预测形状/标签、降维形状，以及完整 Pipeline 序列化前后预测一致。
3. MLP和CNN各完成一个合成batch的logits形状、交叉熵、反向传播、参数更新、state_dict保存/加载及预测一致性检查。

环境：Python 3.12.10、NumPy 2.2.6、scikit-learn 1.9.1、PyTorch 2.6.0+cu124；检查在CPU执行。使用合成输入，没有MNIST性能成绩。

A/B执行者另对回答中的教程代码原样运行，使用60,000/10,000个合成uint8图像匹配NPZ接口；三个串联Pipeline拟合/预测成功，5,000/1,000子集索引不相交、预测形状正确。该示例是PCA→LDA串联演示，不取代正式九组独立对照，也未实现完整实验工件导出。运行发现当前库对显式 `SVC(probability=False)` 发弃用警告，示例改用默认 `SVC()` 后重验通过。临时数据已清理，未留下脚本。

## 附件与打包核验

- 只读原ZIP，42个条目，CRC检查通过；未解压、加载权重或执行包内代码。
- D执行者实际检查训练/验证索引无重复、无交叉、覆盖60,000，重算共享指纹。主代理也复核划分和混淆矩阵。
- 历史MLP/CNN混淆矩阵对角和为9,681/9,787，总数各10,000，与附件accuracy=0.9681/0.9787相符；这仅证明附件内部一致，未独立重训复现。
- UTF-8 YAML解析、界面短描述长度、默认提示词、Skill内部相对链接、无scripts目录等检查见 [package-checks.json](package-checks.json)。根目录不存在 `.git`，没有进行Git初始化或上传。

## 官方验证及限制

最终使用官方工具运行：

```powershell
python -X utf8 "$env:USERPROFILE/.codex/skills/.system/skill-creator/scripts/quick_validate.py" ".agents/skills/mnist-multi-method"
```

实际输出与退出码见 [validation.txt](validation.txt)。该检查只验证Skill结构等基础要求，不能代替行为测试。本文的六个场景是有限的人工审阅行为测试；未运行正式MNIST训练、未建立识别界面、未验证所有硬件/框架组合，也不声称达到某个识别率。
