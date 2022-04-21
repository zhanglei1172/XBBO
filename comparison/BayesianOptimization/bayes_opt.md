# Comparison

## Note

在开始对比测试前，需安装[Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)


- 测试函数：Branin (2d)
- Budget: 200
- Repeat num: 10

| Method        | Minimum       | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| ------------- | ------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| bayes_opt(bo) | 0.398+/-0.000 | 0.398        | 97.0                | 50.367             | 39                     |
| XBBO(bo-gp)   | 0.398+/-0.000 | 0.398        | 42.0                | 5.0398             | 30                     |
