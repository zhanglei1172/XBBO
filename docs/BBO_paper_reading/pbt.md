# Population Based Training of Neural Networks

网络权重和超参同时更新，最终搜索的结果是超参的题调整过程。

- exploit：随机选择一些top模型的weight进行继承、超参进行继承
- explore：对继承的超参进行一定扰动


1. 初始化population
2. 对population中每个个体进行训练一定epoch（partial training）
3. 在达到一定epoch时对表现最差top一定比例的个体进行exploit与explore，形成新的population
4. 判断是否满足终止条件，不满足再回到第二步中