# 未加载 Skill 的基线回答

独立测试代理未读取新 Skill，回答 A/B/C 三个用例。以下保存实际输出，供比较，不代表作者推荐的最终方案。

## A

原理梳理：比较贝叶斯后验决策、SVM间隔、KNN投票，并补充MLP非线性映射与CNN局部卷积；说明PCA保方差、LDA利用类别分离。
固定种子，官方训练集分训练/验证，测试集最后评估。CPU先用分层抽样6000张训练、1000张验证。标准化、PCA、LDA仅在训练集拟合。GaussianNB、线性SVM、KNN、MLP比较原始784维、PCA50维、LDA9维；CNN使用原始28×28图像作参照。MLP/CNN先跑2轮检查，再按验证表现早停。
结果表：模型｜表示｜维数｜训练数｜轮数｜验证准确率｜测试准确率｜宏F1｜训练秒数｜单张推理毫秒。未运行项留空。

## B

教程需修正：保留官方测试集，先拆训练/验证，再拟合预处理；10类LDA最多9维。代码骨架：
`Pipeline([('scale',StandardScaler()),('reduce',PCA(50)),('clf',GaussianNB())])`
以原始特征、PCA50、LDA9为三个分支；分类器替换为`SVC(probability=False)`、`KNeighborsClassifier(3)`、`GaussianNB()`。标准化/PCA会产生负值，不能直接使用MultinomialNB；必须保留它时另设非负像素分支。各流水线只对训练集调用fit，验证后锁定设置，测试一次；固定种子并记录耗时。98.9%仅列“教程声称、未核验”，本次实测准确率留空，运行后填写。

## C

数据改用`datasets.MNIST`，输入统一为`[N,1,28,28]`并归一化。MLP：Flatten→784–128–64–10；经典LeNet式CNN：Conv(1,6,5)→池化→Conv(6,16,5)→池化→Flatten→256–120–84–10，隐藏层用ReLU。
接口：`build_model(name)`、`train_epoch(model,loader,opt)`、`evaluate(model,loader)`、`predict_image(path,checkpoint)`。
固定种子，官方训练集分55000/5000，测试集封存；交叉熵接收logits，CPU先跑2轮检查，按验证损失保存最佳模型，最终测试一次。推理执行相同预处理及eval/no_grad，返回预测数字、最大softmax概率和top3。附件2轮结果标为未核验，不填入实测表。
