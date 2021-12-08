# NSGA-II

> **Dominate**: An individual is said to dominate another if the objective functions of it is no worse than theother and at least in one of its objective functions it is better than the other

## 主要过程

- 选择、交叉、变异（使用二元锦标赛选择）
- non-dominant sort：第一个 front 完全是 non-dominant set，第二个 front 只被第一个 front dominant，依次下去。
- fitness values：每个 front 被赋予 rank value。即第一个 front 的 fitness 是 1，第二个 front 是 2
- crowding distance：描述一个 individual 和邻居有多近，越大代表这个 population 中的多样性越好
- 筛选，保持最优的 N 个 individual 留下（精英策略）

### crowed distance

![](../attachments/2021-08-17-10-53-10.png)

只在同一个 Pareto rank 内比较才有意义
