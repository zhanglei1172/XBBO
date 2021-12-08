# CEM(Cross Entropy Method)

考虑估计期望的问题：

$$
\ell=\mathbb{E}_{\mathbf{u}}[H(\mathbf{X})]=\int H(\mathbf{x}) f(\mathbf{x} ; \mathbf{u}) \mathrm{d} \mathbf{x}
$$

可以用 importance sampling 方法

$$
\hat{\ell}=\frac{1}{N} \sum_{i=1}^{N} H\left(\mathbf{X}_{i}\right) \frac{f\left(\mathbf{X}_{i} ; \mathbf{u}\right)}{g\left(\mathbf{X}_{i}\right)}
$$

最优的$g^*$:

$$
g^{*}(\mathbf{x})=H(\mathbf{x}) f(\mathbf{x} ; \mathbf{u}) / \ell
$$

但$\ell$是未知的，算法就是通过最小化$g$与$g^*$的 KL 散度来选择$g$

## 算法

1. $t=1$ ,选择初始向量$v^{(0)}$
2. 从$ f\left(\mathbf{x} ; \mathbf{v}^{(t-1)}\right)$ 中随机采样 $\mathbf{X}_{1}, \ldots, \mathbf{X}_{N} $
3. $\mathbf{v}^{(t)}=\underset{\mathbf{u}}{\operatorname{argmax}} \frac{1}{N} \sum_{i=1}^{N} H\left(\mathbf{X}_{i}\right) \frac{f\left(\mathbf{X}_{i} ; \mathbf{u}\right)}{f\left(\mathbf{X}_{i} ; \mathbf{v}^{(t-1)}\right)} \log f\left(\mathbf{X}_{i} ; \mathbf{v}^{(t-1)}\right)$
4. 若未收敛$t=t+1$,返回`2`，收敛则返回

## 例子

原始的用途是估计期望，可以用来优化

如求$S(x)=\mathrm{e}^{-(x-2)^{2}}+0.8 \mathrm{e}^{-(x+2)^{2}}$最大

估计$\mathbb{P}_{\theta}(S(X) \geq \gamma)$，$\gamma$是一个分位数点，逐步更新$g$与$\gamma$
（重要性采样中，$H$和 $f$都比较大的部分$g^*$应该也越大）

```
while t < maxits and σ2 > ε do
    // Obtain N samples from current sampling distribution
    X := SampleGaussian(μ, σ2, N)
    // Evaluate objective function at sampled points
    S := exp(−(X − 2) ^ 2) + 0.8 exp(−(X + 2) ^ 2)
    // Sort X by objective function values in descending order
    X := sort(X, S)
    // Update parameters of sampling distribution
    μ := mean(X(1:Ne))
    σ2 := var(X(1:Ne))
    t := t + 1
```
