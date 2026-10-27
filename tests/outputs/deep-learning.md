# C：迁移方案

已阅读[官方Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)；未训练、安装或下载数据。

**原文**：数据章节用FashionMNIST、batch=64；模型章节为Flatten→784–512–512–10、ReLU；优化章节用交叉熵、SGD(lr=0.001)，5轮且每轮查看测试集；保存/加载章节用state_dict重建网络。无CNN或独立验证集，成绩不属于数字MNIST。

**迁移设计**：仅有ZIP，以下为建议。改用MNIST，类别0–9；只读本地数据，缺失报错。60,000训练池分层划分55,000/5,000，seed=42，保存索引/指纹；10,000测试集封存。图像float32、[0,1]，标签long，共用划分与预处理。

接口：`loaders(config)→train,val,test,索引`；`build_model(name)→Module`；`train/evaluate(loader,model)→loss,accuracy`；`predict(image,checkpoint)→JSON`。

MLP：输入[N,1,28,28]→Flatten[N,784]→256→128→10，隐藏层ReLU。LeNet风格CNN：Conv(1→6,5)→ReLU→Pool(2)→Conv(6→16,5)→ReLU→Pool(2)，无padding，空间28→24→12→8→4；展平256→120→84→10，隐藏全连接ReLU。两者输出[N,10] logits，交叉熵前不softmax。

CPU：Adam、lr=0.001、batch=128、最多10轮、workers=0；训练shuffle。train()、清梯度/反传/更新；验证eval()+no_grad，损失按样本加权。最低验证损失保存权重及结构、预处理、类别、来源；调参/早停仅看验证，锁定后加载最佳权重测测试集。记录版本、设备、耗时、曲线和分类指标。检查形状、划分无交叉、存取预测一致。

单图：灰度、按需反色、裁笔画、等比缩至约20×20并居中到28×28、同训练缩放；拒绝空白，保存预览。eval()+no_grad后argmax给数字；softmax给10类分数，非校准正确率。JSON：model、predicted_digit、scores、score_type、classes、preprocessing、checkpoint_source、input_preview。

附件“CPU已跑2轮”缺日志，属未核验声称。两模型status=not_run，accuracy/macro_f1=null，无实际预测。
