# A：实际回答

文章的方向正确，但贝叶斯后验决策不等于朴素条件独立。基线用 GaussianNB：按类别估计各像素均值、方差，预测 argmax P(y)∏P(x_j|y)；训练约 O(Nd)。SVM 以软间隔平衡误差与间隔，RBF 核可学习非线性边界；CPU 首轮采用 LinearSVC，核 SVC 另列。KNN 用欧氏近邻投票，存储及单次暴力查询约 O(Nd)。MLP 学习非线性全连接映射；CNN 用局部感受野和共享权重保持空间结构。

PCA 最大化投影方差、不用标签；LDA 最大化类间/类内散度比，只用训练标签，10 类最多9维。二者均仅在训练数据拟合；降维不保证提高准确率。

CPU 先预检，再做固定分层子集：seed=42，从官方60,000训练池划分55,000/5,000；预算内取共同5,000/1,000子集并保存索引及指纹。官方10,000测试集封存。11组共用索引；子集结果单列，不冒充全量。预计耗时须经预检确认，首轮限制两小时，未结束项标未运行。

|模型|表示|候选|
|---|---|---|
|GaussianNB、LinearSVC、KNN|原始784/PCA50/LDA9|共9组；首轮固定平滑1e-9/C=1/k=3|
|MLP、LeNet风格CNN|784/1×28×28|各1组；CPU先2轮，按验证损失选权重|

像素缩放到[0,1]；SVM/KNN可标准化，差异须记录。后续只凭验证集调参，配置锁定后统一测试。输出列：模型、表示/维数、样本数、seed、指纹、accuracy、macro-F1、预处理/训练耗时、推理批量与计时范围、状态；本次均“未运行”。另存0–9混淆矩阵、分类报告、误分类索引、神经网络曲线和完整模型。依据：[官方指南](https://scikit-learn.org/stable/modules/lda_qda.html)。

# B：实际回答

教程在“合并后拟合再划分”处泄漏测试信息；LDA50非法（10类最多9维）；投影可能有负值，改用GaussianNB；无需概率时关闭SVC概率估计。依据：[防泄漏指南](https://scikit-learn.org/stable/common_pitfalls.html)。98.9%只能列为“教程声称、未核验”，当前accuracy/macro-F1为null，状态未运行。

已有Python、NumPy、scikit-learn可用。保存下列代码为run.py，提供本地数字MNIST的mnist.npz，键x_train/y_train/x_test/y_test，分别为60,000和10,000个原始uint8图像/标签；运行`python run.py`。此为PCA→LDA串联方案，不代替独立PCA/LDA对照。CPU子集结果单列；本次未执行训练。

```python
import numpy as np
from sklearn.model_selection import train_test_split as split
from sklearn.pipeline import make_pipeline as pipe
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

d=np.load('mnist.npz',allow_pickle=False)
X=d['x_train'].reshape(60000,784).astype('float32')/255
y=d['y_train']
assert set(y)==set(range(10))
t,v=split(np.arange(len(y)),test_size=5000,stratify=y,random_state=42)
t,_=split(t,train_size=5000,stratify=y[t],random_state=42)
v,_=split(v,train_size=1000,stratify=y[v],random_state=42)
np.savez('indices.npz',train=t,validation=v)
models=[]
for c in [GaussianNB(),SVC(),KNeighborsClassifier(3)]:
    m=pipe(StandardScaler(),PCA(50,svd_solver='randomized',random_state=42),
           LDA(n_components=9,solver='svd'),c)
    m.fit(X[t],y[t])
    p=m.predict(X[v])
    a=accuracy_score(y[v],p)
    print(type(c).__name__,'validation',a,f1_score(y[v],p,average='macro'))
    models.append((a,m))
# 验证集选定模型；此后不得根据测试结果换模型或调参。
m=max(models,key=lambda z:z[0])[1]
T=d['x_test'].reshape(10000,784).astype('float32')/255
p=m.predict(T)
print('locked test',accuracy_score(d['y_test'],p),
      f1_score(d['y_test'],p,average='macro'))
```

# 执行记录（不属于A/B回答正文）

- 实际读取：`tests/article-cases.md`、`.agents/skills/mnist-multi-method/SKILL.md`、`references/experiment-guide.md`；另读using-superpowers技能，其中SUBAGENT-STOP要求受派子代理忽略该流程。
- 实际核对网页：scikit-learn官方Common pitfalls、LDA/QDA、Naive Bayes、SVM、Nearest Neighbors页面。
- 目录与依赖检查：根目录已有ZIP、README、LICENSE和技能/tests目录；未发现解包后的Python工程、依赖清单或本地MNIST数据；没有解压ZIP。检测到Python3.12.10、NumPy2.2.6、scikit-learn1.9.1、PyTorch2.6.0+cu124。
- 未进行MNIST训练、未下载数据、未安装依赖、未修改Skill。对回答内原样代码生成60000/10000张合成uint8图像执行了一次功能检查：3个Pipeline成功拟合与预测、训练/验证子集索引不相交、输出有限、预测形状正确；此检查不代表MNIST识别率。临时合成数据与索引已清理，没有额外.py文件。代码依赖用户提供符合约定的本地NPZ，尚未以MNIST执行；不提供性能数值。
- 环境差异：scikit-learn1.9.1对显式传入SVC(probability=False)也发弃用警告，改成默认SVC()后复核。首次系统临时目录访问受限，改到工作区受控临时目录完成检查。
- 范围说明：A/B中文正文各不超过800字；B为满足可运行要求另附完整代码。简短教程片段未实现完整工件导出、计时与单图推理，正式实验应按Skill契约补齐。
- Skill观察：未发现影响本次纠错的实质遗漏。原理输出“至少说明”六类模型的写法可能使仅问单一方法时过度展开；与“只交付本次要求”的边界可再明确。
