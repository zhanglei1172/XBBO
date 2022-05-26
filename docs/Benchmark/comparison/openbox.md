# Comparison with open-box

## Note

在开始对比测试前，需安装[open-box](https://github.com/thomas-young-2013/open-box)

## Synthetic Function Optimization

- 测试函数：Branin (2d)
- 测试优化器：Bayesian Optimization
- 初始设计：Sobol design
- Budget: 200
- Repeat num: 10

| Method        | Minimum       | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| ------------- | ------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| openbox(auto) | 0.398+/-0.000 | 0.398        | 144.0               | 35.426             | 90                     |
| XBBO(bo-gp)   | 0.398+/-0.000 | 0.398        | 42.0                | 5.0398             | 30                     |

