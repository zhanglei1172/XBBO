# CMA-ES

## ES

1. 采样
2. 更新均值
3. 更新协方差

形式： (μ/ρ +, λ)-ES

- μ：population
- $\rho$表示从 population 中选择的父代数
- λ：子代数（offspring）
- ‘+’代表 ρ + λ 混合起来进行适者生存；‘，’代表只在 λ 中进行适者生存

### 正态分布

选择正态分布的理由：

1. 正态分布变量的求和仍然服从正态分布
2. 正态分布易于生成各项同性的样本，rotation invariant
3. 最大熵（最少的先验）

## 采样

$$
\boldsymbol{x}_{k}^{(g+1)} \sim \boldsymbol{m}^{(g)}+\sigma^{(g)} \mathcal{N}\left(\mathbf{0}, \boldsymbol{C}^{(g)}\right)
$$

## 选择和重组（移动均值）

$$
\begin{gathered}
m^{(g+1)}=\sum_{i=1}^{\mu} w_{i} \boldsymbol{x}_{i: \lambda}^{(g+1)} \\
\sum_{i=1}^{\mu} w_{i}=1, \quad w_{1} \geq w_{2} \geq \cdots \geq w_{\mu}>0
\end{gathered}
$$

- $\lambda$是 population 数量
- ($i:\lambda$) 表示 $\lambda$ 中排名（rank）$i$的索引
- 均值更新即为选择较好的一部分（_truncation selection_）父代的加权平均

Generalized:

$$
\boldsymbol{m}^{(g+1)}=\boldsymbol{m}^{(g)}+c_{\mathrm{m}} \sum_{i=1}^{\mu} w_{i}\left(\boldsymbol{x}_{i: \lambda}^{(g+1)}-\boldsymbol{m}^{(g)}\right)
$$

$c_m$是学习率，当 $c_m$ 取 1 时就是均值加权形式，一般形式对 noise functions 更有优势

## Covariance matrix

$$
\boldsymbol{C}_{\mu}^{(g+1)}=\sum_{i=1}^{\mu} w_{i}\left(\boldsymbol{x}_{i: \lambda}^{(g+1)}-\boldsymbol{m}^{(g)}\right)\left(\boldsymbol{x}_{i: \lambda}^{(g+1)}-\boldsymbol{m}^{(g)}\right)^{\top}
$$

$$
C_{\mathrm{EMNA}_{\text {global }}}^{(g+1)}=\frac{1}{\mu} \sum_{i=1}^{\mu}\left(\boldsymbol{x}_{i: \lambda}^{(g+1)}-\boldsymbol{m}^{(g+1)}\right)\left(\boldsymbol{x}_{i: \lambda}^{(g+1)}-m^{(g+1)}\right)^{\top}
$$

$\boldsymbol{C}_{\mu}^{(g+1)}$ 关注于 step 的方差，即 $(\boldsymbol{x}_{i}^{(g+1)}-\boldsymbol{m}^{(g)})$；而 EMNA（ Estimation of Multivariate Normal Algorithm） 关注的是样本点的方差。

![](../attachments/2021-08-15-09-32-28.png)

1. 从图上看，可以得出：$\boldsymbol{C}_{\mu}^{(g+1)}$ 总是会在 gradient 方向增加方差(梯度方向会得到更多的探索)，而 EMNA 则会在 gradient 方向减少方差。所以 EMNA 容易造成**过早收敛**

2. 考虑到**_reliable_**,结合前几代方差的信息。

### rank $\mu$ update

using all selected steps from a _single generation_

$$
\begin{aligned}
\boldsymbol{C}^{(g+1)} &=\left(1-c_{\mu} \sum w_{i}\right) \boldsymbol{C}^{(g)}+c_{\mu} \sum_{i=1}^{\lambda} w_{i} \boldsymbol{y}_{i: \lambda}^{(g+1)} \boldsymbol{y}_{i: \lambda}^{(g+1)^{\top}} \\
&=\boldsymbol{C}^{(g)^{1 / 2}}\left(\mathbf{I}+c_{\mu} \sum_{i=1}^{\lambda} w_{i}\left(\boldsymbol{z}_{i: \lambda}^{(g+1)} \boldsymbol{z}_{i: \lambda}^{(g+1)^{\top}}-\mathbf{I}\right)\right) \boldsymbol{C}^{(g)^{1 / 2}}
\end{aligned}
$$

$x$ 变换到 $y$,得到各向同性的性质

### rank 1 update

in the generation sequence using a _single selected step_

$$
\boldsymbol{C}^{(g+1)}=\left(1-c_{1}\right) \boldsymbol{C}^{(g)}+c_{1} \boldsymbol{y}_{g+1} \boldsymbol{y}_{g+1}^{\top}
$$

这步可以增加 successful step 的似然

因为$\boldsymbol{y}$的正负号与方差更新无关，所以引入*path*的概念。

$$
\boldsymbol{p}_{\mathrm{c}}^{(g+1)}=\left(1-c_{\mathrm{c}}\right) \boldsymbol{p}_{\mathrm{c}}^{(g)}+\sqrt{c_{\mathrm{c}}\left(2-c_{\mathrm{c}}\right) \mu_{\mathrm{eff}}} \frac{\boldsymbol{m}^{(g+1)}-\boldsymbol{m}^{(g)}}{\sigma^{(g)}}
$$

### combine

$$
\begin{array}{r}
\boldsymbol{C}^{(g+1)}=(\underbrace{1-c_{1}-c_{\mu} \sum w_{j}}_{\text {can be close or equal to } 0}) \boldsymbol{C}^{(g)} \\
+c_{1} \underbrace{\boldsymbol{p}_{\mathrm{c}}^{(g+1)} \boldsymbol{p}_{\mathrm{c}}^{(g+1)}}_{\text {rank-one update }}+c_{\mu} \underbrace{\sum_{i=1}^{\lambda} w_{i} \boldsymbol{y}_{i: \lambda}^{(g+1)}\left(\boldsymbol{y}_{i: \lambda}^{(g+1)}\right)^{\top}}_{\text {rank- } \mu \text { update }}
\end{array}
$$

## Step-size control（$\sigma$）

![](../attachments/2021-08-18-17-45-55.png)

- 1/5-th success rule
  - 应用于“+” selection
  - 超过 20%的新 solutions 被接受就增加$\sigma$，否则减少
  - $$
    \sigma \leftarrow \sigma \times \exp \left(\frac{1}{N} \times \frac{p_{s}-p_{\text {target }}}{1-p_{\text {target }}}\right)
    $$
- $\sigma$-self-adaption
  - 应用于‘，’ selection
  - 引入 mutation
- path length control（Cumulative Step-szie Adaptrion）CSA
  - CMA-ES 的默认 step-szie

### CSA

![](../attachments/2021-08-14-18-44-26.png)

$$
p_{\sigma}^{(g+1)}=\left(1-c_{\sigma}\right) \boldsymbol{p}_{\sigma}^{(g)}+\sqrt{c_{\sigma}\left(2-c_{\sigma}\right) \mu_{\mathrm{eff}}} \boldsymbol{C}^{(g)^{-\frac{1}{2}}} \frac{\boldsymbol{m}^{(g+1)}-\boldsymbol{m}^{(g)}}{\sigma^{(g)}}
$$

$$
\sigma \leftarrow \sigma \times \exp \left(\frac{c_{\sigma}}{d_{\sigma}}\left(\frac{\left\|p_{\sigma}\right\|}{\mathrm{E}\|\mathcal{N}(\mathbf{0}, \mathbf{I})\|}-1\right)\right)
$$
