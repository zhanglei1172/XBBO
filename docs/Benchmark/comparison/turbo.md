# Comparison with TuRBO

## Note

在开始对比测试前，需安装[TuRBO](https://github.com/uber-research/TuRBO)


- 测试函数：Branin (2d)
- Budget: 200
- Repeat num: 10

| Method           | Minimum       | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min |
| ---------------- | ------------- | ------------ | ------------------- | ------------------ | ---------------------- |
| TuRBO(turbo-1) | 0.399+/-0.001 |    0.398     |        141.3        |       42.711       |           50           |
| TuRBO(turbo-2) | 0.399+/-0.001 |    0.398     |        130.1        |       47.365       |           34           |
| XBBO(turbo-1) | 0.398+/-0.000 |    0.398     |        110.3        |       46.596       |           46           |
| XBBO(turbo-2) | 0.398+/-0.000 |    0.398     |        130.7        |       48.57        |           68           |