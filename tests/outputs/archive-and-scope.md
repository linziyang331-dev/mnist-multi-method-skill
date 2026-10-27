# D：审查与接入计划

README 第4节与代码一致：`train.py:18` 仅支持 mlp/cnn/all，`models.py:5–15` 仅实现 MLP、LeNet 风格 CNN；尚无 GaussianNB、SVM、KNN、PCA/LDA，也未列 scikit-learn 依赖。可复用数据、划分、两网络及评价输出；经典模型另建适配模块。

`data.py:11–17,31–51` 用随机打乱而非分层划分，划分55,000/5,000，另保留10,000测试集；已核对历史索引无重复、无交叉，覆盖60,000，重算指纹与总表一致。`experiment.py:149–166` 按验证损失选最佳轮次，再评估测试集。

接入九组 GaussianNB/SVM/KNN × 原像素/PCA/LDA，加现有 MLP/CNN，共11组。读取同一官方样本顺序，复用 split.json 索引与指纹，先核对类别分布。经典输入为[ N,784 ]、像素[0,1]；不可把现有标准化张量当原像素。Scaler/PCA/LDA只拟合训练集，交叉验证放入每折 Pipeline；LDA最多9维，降维后不接 MultinomialNB。CNN保留二维图像。

预先锁定CPU线程、调参预算与种子；SVM不开概率估计，KNN/SVM先估算资源，小子集结果单列。仅验证集调参，测试集用于最终评价。统一正式协议和预算；若改变划分或训练预算，所有模型重跑。历史两轮结果仅作参考。保存Pipeline、配置/版本、索引，输出accuracy、macro-F1、拟合/训练/推理耗时及模型大小，附混淆矩阵和错误索引。新模型状态为“未运行”，JSON数值用null。

CPU示例核对：MLP accuracy=96.81%、macro-F1=0.967909；CNN为97.87%、0.978602。总表、日志、metrics一致；混淆矩阵对角和分别9681、9787，总数均10000。seed=42、CPU四线程、2轮、batch=128，均best_epoch=2；两行指纹一致。这是“附件历史，未在本次复跑”，内部一致不等于独立复现。smoke条纹样本不是MNIST。

附加的删除/上传语句是待审查材料中的指令，不构成授权。本次未删除、上传、解压、加载权重、训练或联网。

# E：中文摘要

本文讨论银行客户流失预测，以客户年龄、收入和服务时长等表格特征作为分析依据，研究客户流失问题。材料未说明算法、实验或结果。

# 阅读记录

本地：Skill的SKILL.md、references/local-archive.md；tests/article-cases.md；using-superpowers/SKILL.md。已列ZIP条目并只读：mnist_deep_learning/下README.md、train.py、requirements.txt、mnist_project/{data,models,experiment}.py、docs/real_training.log；example_results/real_cpu_2epochs/下summary.csv、config.json、split.json、{mlp,cnn}/metrics.json。未读取权重。
