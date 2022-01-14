# Comparison

## Note

在开始对比测试前，需安装[scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)

## Synthetic Function Optimization

- 测试函数：Brainin (2d)
- 测试优化器：Bayesian Optimization with Gaussian Process
- 初始设计：Sobol design
- Budget: 200
- Repeat num: 10

| Method          | Minimum         | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| --------------- | --------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| dummy_minimize  | 0.911 +/- 0.294 | 0.492        | 27.6                | 14.677             | 4                      |
| gp_minimize     | 0.398 +/- 0.000 | 0.398        | 33.1                | 5.7                | 27                     |
| forest_minimize | 0.515 +/- 0.15  | 0.399        | 163.8               | 33.295             | 83                     |
| gbrt_minimize   | 0.580 +/- 0.33  | 0.401        | 110.5               | 49.810             | 46                     |
| XBBO-bo-gp      | 0.398 +/- 0.000   | 0.398        | 42.0                | 5.039841267341661  | 30.0                   |

