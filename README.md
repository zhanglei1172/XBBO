# bbomark

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

- [x] Transfer
- [ ] Parallelizing
- [x] Result visualize
- [ ] Reproductable ( Random state )
- [x] Extra data and model
- [ ] log verbose
- [x] BORE optimizer