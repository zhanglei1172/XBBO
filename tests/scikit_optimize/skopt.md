# Comparison

## Note

在开始对比测试前，需安装[scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)

## Synthetic Function Optimization

- 测试函数：Branin (2d)
- 测试优化器：Bayesian Optimization
- 初始设计：Sobol design
- Budget: 200
- Repeat num: 10

| Method          | Minimum         | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| --------------- | --------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| skopt(gp_minimize)     | 0.398+/-0.000   | 0.398        | 24.4                | 2.69072 | 22                     |
| skopt(forest_minimize) | 0.417+/-0.033   | 0.398        | 118.1               | 36.4786 | 31                     |
| skopt(gbrt_minimize)   | 0.550+/-0.196   | 0.399        | 132.0               | 61.236  | 11                     |
| skopt(dummy_minimize)  | 0.662+/-0.167   | 0.483        | 86.1                | 54.769  | 19                     |
| XBBO(bo-gp)      | 0.398 +/- 0.000 | 0.398        | 42.0                | 5.0398  | 30                     |

