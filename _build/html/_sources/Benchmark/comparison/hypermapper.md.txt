# Comparison with hypermapper

## Note

在开始对比测试前，需安装[hypermapper](https://github.com/luinardi/hypermapper)

## Synthetic Function Optimization

- 测试函数：Branin (2d)
- Budget: 200
- Repeat num: 10

| Method           | Minimum       | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| ---------------- | ------------- | ------------ | ------------------- | ------------------ | ---------------------- |
|hypermapper(bo)   | 0.423+/-0.018 | 0.403       | 100.6            | 59.011            | 5                     |
| hypermapper(evolution)|0.504+/-0.084| 0.403        | 85.6                | 63.756             | 4                     |
| XBBO(bo-gp)   | 0.398+/-0.000 | 0.398        | 42.0                | 5.0398             | 30                     |