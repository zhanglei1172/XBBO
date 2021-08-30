# bbomark

---

## 更新

### TST-R (2021年8月29日)

测试运行：

```bash
PYTHONPATH='./' python ./bbomark/search_algorithm/transfer_tst_optimizer.py
```

Result:

![](./out/TST-R.png)

![](./out/TST-R-correct.png)

Output:


```

----------
iter 1: 
0.810625
rank:  162
rank_best:  162
----------
iter 2: 
0.847886
rank:  5
rank_best:  5
----------
iter 3: 
0.845144
rank:  40
rank_best:  5
----------
iter 4: 
0.763538
rank:  250
rank_best:  5
----------
iter 5: 
0.770806
rank:  231
rank_best:  5
----------
iter 6: 
0.839083
rank:  95
rank_best:  5
----------
iter 7: 
0.844099
rank:  52
rank_best:  5
----------
iter 8: 
0.780428
rank:  168
rank_best:  5
----------
iter 9: 
0.842666
rank:  67
rank_best:  5
----------
iter 10: 
0.849217
rank:  1
rank_best:  1
----------
iter 11: 
0.821169
rank:  150
rank_best:  1
----------
iter 12: 
0.845532
rank:  33
rank_best:  1
----------
iter 13: 
0.846453
rank:  25
rank_best:  1
----------
iter 14: 
0.843587
rank:  59
rank_best:  1
----------
iter 15: 
0.847681
rank:  9
rank_best:  1
----------
iter 16: 
0.834169
rank:  126
rank_best:  1
----------
iter 17: 
0.847272
rank:  12
rank_best:  1
----------
iter 18: 
0.845429
rank:  36
rank_best:  1
----------
iter 19: 
0.847784
rank:  7
rank_best:  1
----------
iter 20: 
0.847272
rank:  12
rank_best:  1
----------
iter 21: 
0.846658
rank:  21
rank_best:  1
----------
iter 22: 
0.828744
rank:  138
rank_best:  1
----------
iter 23: 
0.84717
rank:  16
rank_best:  1
----------
iter 24: 
0.838264
rank:  103
rank_best:  1
----------
iter 25: 
0.837343
rank:  110
rank_best:  1
----------
iter 26: 
0.847989
rank:  4
rank_best:  1
----------
iter 27: 
0.847272
rank:  12
rank_best:  1
----------
iter 28: 
0.847886
rank:  5
rank_best:  1
----------
iter 29: 
0.846555
rank:  22
rank_best:  1
----------
iter 30: 
0.846555
rank:  22
rank_best:  1

```

---

### Anneal （2021年8月25日）
   
在 branin 黑盒函数测试

![](./out/bbomark-anneal-noise.png)
best_loss: 0.36538257770050925

![](./out/bbomark-anneal.png)
best_loss: 0.398102262920645

![](./out/hyperopt-anneal.png)
best_loss: 0.4024387250287731

---

---

所有参数配置使用yaml格式，见`/cfgs/*.yaml`


## 测试

[实验记录点这里！](./实验记录.md)

跑benchmark实验运行示例：

```bash
PYTHONPATH='./' python ./example/demo.py
```

对实验结果分析：

```bash
PYTHONPATH='./' python ./example/analysis.py
```

[comment]: <> (![]&#40;./out/demo_res.png&#41;)

![](./out/ana_res1628421033.21489.png)

## TODO

- [ ] Transfer
  - [x] TST-R
  - [ ] TAF
  - [ ] TAF(RGPE)
  - [ ] RMoGP
  - [ ] RGPE(mean)
  
- [ ] Optimizer
  - [x] BORE optimizer
  - [x] Anneal
  - [ ] DE
  - [ ] CMA
  - [ ] NSGA
  - [ ] Regularized EA
  - [ ] PBT
  
- [ ] condition config
- [ ] Parallelizing
- [x] Result visualize
- [x] Reproductable ( Random state )
- [ ] log verbose
- [ ] suggest duplicate detection

