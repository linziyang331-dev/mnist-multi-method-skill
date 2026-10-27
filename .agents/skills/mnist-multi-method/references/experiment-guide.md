# 原理与实验指南

用于原理讲解、生成代码和实验设计；数值是候选配置，不是性能保证。官方资料核对日期：2026-10-27。调用具体 API 时结合当前环境版本检查参数。

## 贝叶斯、SVM、KNN

**贝叶斯决策**：0–1 损失下，`ŷ = argmax_k P(y=k|x) = argmax_k P(x|y=k)P(y=k)`。朴素贝叶斯进一步假设给定类别时像素条件独立：`log P(y=k) + Σ_j log P(x_j|y=k)`。GaussianNB 对各特征使用类条件高斯，记录均值、方差和先验；用对数计算防止概率乘积下溢。像素相关性明显，所以独立假设是近似；PCA 去相关也不自动保证条件独立。

GaussianNB 训练约 `O(Nd)`，预测约 `O(Kd)`/样本；常数类方差需要平滑。连续像素/带负值投影可用 GaussianNB；BernoulliNB 是显式阈值二值化的独立消融。MultinomialNB 要求非负特征，不直接接中心化/PCA/LDA 输出。若研究一般高斯贝叶斯或 LDA/QDA 分类器，应单列协方差假设，不能静默替换 GaussianNB。

**SVM**：二分类软间隔目标 `min ½||w||² + CΣξ_i`，约束 `y_i(wᵀφ(x_i)+b) ≥ 1−ξ_i, ξ_i≥0`。解释 C 的误差/间隔权衡及 RBF 核 `exp(−γ||x−x′||²)`。多分类实现需注明策略：SVC 内部 one-vs-one，LinearSVC 通常 one-vs-rest。标准化对距离/间隔很重要；核方法训练成本随样本数增长很快，LinearSVC 作为 CPU 快速基线，RBF-SVC 是单列扩展。推理成本取决于维数及支持向量数；没有统一的所有求解器训练复杂度公式。

**KNN**：找距离最近的 k 个训练样本，按等权或距离权重投票；解释 k、欧氏/曼哈顿距离、平局策略及维数灾难。几乎无参数训练，但存储约 `O(Nd)`，暴力查询约 `O(Nd)`/样本；推理计时需记录查询批大小，不能因 fit 快就断言总成本低。

## PCA 与 LDA

- PCA：训练数据中心化后用 SVD 得到正交方向，最大化投影方差；它不使用标签，也不直接优化分类边界。记录标准化策略、维数、累计解释方差、whiten 和 solver。
- 整数 PCA 维数要小于等于当前训练折样本数/特征数的限制，`arpack` 要严格小于两者最小值。例如 `n_components=0.95` 表示解释方差比例，可使用 `svd_solver="full"`，不是随机 SVD 的整数维数参数。
- LDA：定义 `S_W = Σ_k Σ_{i∈k}(x_i−μ_k)(x_i−μ_k)ᵀ`、`S_B = Σ_k n_k(μ_k−μ)(μ_k−μ)ᵀ`；通过 `S_B w = λ S_W w` 的广义特征方向解释类间/类内散度比。奇异矩阵不宜直接求逆；选择 SVD 或有依据的正则化求解。
- scikit-learn 的 `LinearDiscriminantAnalysis(solver="svd")` 可作 transform；`solver="lsqr"` 不支持 transform。shrinkage 的兼容性需查 solver；不要拼接不支持的参数。
- LDA 最多 `min(K_train−1, d)` 个方向，实际可更少；某折缺类时按该折类别数处理。MNIST 完整 10 类上限 9，不要沿用 PCA 的 50/100 维。
- PCA→LDA 属于串联路线，PCA 可能丢掉区分类别的方向；单独记录，而非宣称总优于独立 LDA。

## 经典模型实验矩阵与预算

| 模型 | 无降维 | PCA | LDA | 少量验证候选 |
|---|---|---|---|---|
| GaussianNB | 784维 | 合法 d | 合法 d≤9 | var_smoothing：1e-11、1e-9、1e-7 |
| SVM | 784维 | 合法 d | 合法 d≤9 | LinearSVC C：0.1、1、10；RBF 另列 C/γ |
| KNN | 784维 | 合法 d | 合法 d≤9 | k：3、5、7；等权/距离权重 |

PCA 候选可用 20/50/100 或 95% 方差，LDA 可用 2/5/9，按样本量与类别数裁剪。先选少量候选验证，不默认穷举所有组合。CPU 快速检查可固定分层子集；正式横比要让 11 组共享相同索引和样本量。所有候选、预算及是否达到收敛均记录；一组 ConvergenceWarning 不能默认为成功收敛。

最小可运行的折内变换示例（纯合成流程检查，不是 MNIST 实验）：

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

rng = np.random.default_rng(42)
X = rng.normal(size=(100, 784)).astype(np.float32)
y = np.tile(np.arange(10), 10)
pipe = make_pipeline(
    StandardScaler(),
    LinearDiscriminantAnalysis(n_components=9, solver="svd"),
    GaussianNB(),
)
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv)
assert scores.shape == (2,) and np.isfinite(scores).all()
print("synthetic functional check only; NOT MNIST")
```

## 经典深度学习

优先复用已有 PyTorch 工程；新实现可从以下结构开始，明确叫“LeNet 风格”，不要声称完整复刻 LeNet-5。

- MLP：Flatten → 784–256–128–10，隐藏层 ReLU，可选 Dropout(0.2)。
- CNN：Conv(1→6,k=5) → ReLU → MaxPool(2) → Conv(6→16,k=5) → ReLU → MaxPool(2) → Flatten → 256–120–84–10，隐藏全连接用 ReLU。无 padding 时空间尺寸为 `28→24→12→8→4`，展平为 `16×4×4=256`。
- 标签是 0–9 的整数 `long`，输出 `[N,10]` logits；CrossEntropyLoss 在训练中处理归一化。
- 可从 Adam、lr=0.001、batch=128、10 epochs、seed=42 起步，按验证集和实际预算调整；Windows 默认 `num_workers=0`，没有 CUDA 时用 CPU。
- 保存最小验证损失的 checkpoint，再用该 checkpoint 做最终测试。若复用固定 MNIST 归一化常数 `mean=0.1307/std=0.3081`，注明其来源；自行估计时只用训练部分。
- 记录损失、accuracy、epoch 时长、训练与验证曲线。跨 seed 均值/标准差是可选增强；不要挑最好的测试 seed 报告。

## 推理与输出

经典模型保存包括缩放/降维在内的整个 Pipeline；神经网络保存 state_dict 以及结构/归一化/标签映射。加载后使用相同输入再预测，核对预测一致性；只加载可信来源的模型文件。

单图约定：灰度 → 按背景决定反色 → 提取笔画边界 → 等比缩放到约 20×20 → 放入 28×28 → 居中 → 与训练一致的缩放/归一化。拒绝空白或无有效对比度图；保留 `preprocessed.png`。多数字、复杂背景照片是额外检测/域适配任务，不自动变成整页 OCR。

预测 JSON 字段：`model`、`predicted_digit`、`scores`、`score_type`、`classes`、`preprocessing`、`checkpoint_source`、`input_preview`。提供可用的完整 10 类分数；SVM 的 decision_function 记为 margin，不伪装概率。只有 top-k 时注明输出截断。未经校准的 softmax 或 GaussianNB 分数不要表述为实际正确概率。

结果状态可机器读为 `measured_current`、`historical_attachment`、`reported_source`、`functional_check_only`、`not_run`；每条结果必须有来源。未运行示例：

```json
{"model":"GaussianNB","representation":"pca","n_components":50,"seed":42,"dataset":"MNIST","split_sha256":null,"accuracy":null,"macro_f1":null,"status":"not_run"}
```

预处理拟合时间、分类器 fit 时间和端到端时间分开记录；推理说明 batch、预热、是否含预处理及设备同步。不同硬件或输入规模下的耗时不能直接横比。报告混淆矩阵时固定类别顺序 0–9，行是真实类、列是预测类。

## 官方依据

- [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [PCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [LDA/QDA 原理](https://scikit-learn.org/stable/modules/lda_qda.html)
- [LDA API](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [预处理与数据泄漏](https://scikit-learn.org/stable/common_pitfalls.html)
- [Torchvision MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)：原文是 FashionMNIST；用于学习训练流程，迁移时更换为数字 MNIST。
- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
