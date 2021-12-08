# Regularized Evolution for Image Classifier Architecture Search

针对NAS问题，主要思路是：killing the oldest model in the population

1. 初始化population
2. 对population中每个个体进行完整训练、评估
3. 锦标赛选择parents
4. 创建children（通过交叉、变异）
5. 从population中移除最旧的个体，保持population数量固定
6. 判断是否满足终止条件，不满足再回到第二步中