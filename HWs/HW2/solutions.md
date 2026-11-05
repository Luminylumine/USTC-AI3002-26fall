# HW2

**姓名：** 林思睿　　**学号：** PB24000301　　**学校：** 中国科学技术大学  
**课程：** 人工智能与机器学习基础 (AI3002 26fall)

---

## 1 神经网络的前向传播与反向传播

网络结构包含输入层（2 个神经元）、隐藏层（2 个神经元，sigmoid 激活）与输出层（标量线性输出）：

$$
\mathbf{z}=\mathbf{W}^{\top}\mathbf{x}+\mathbf{b},\qquad \mathbf{h}=\sigma(\mathbf{z}),\qquad \hat{y}=\mathbf{v}^{\top}\mathbf{h},\qquad L=\tfrac12(\hat{y}-y)^2
$$

激活函数为 $\sigma(t)=1/(1+e^{-t})$，对向量逐元素作用。给定参数与输入数据：

$$
\mathbf{x}=\begin{bmatrix}1\\-1\end{bmatrix},\quad \mathbf{W}=\begin{bmatrix}1&0\\0&0\end{bmatrix},\quad \mathbf{b}=\begin{bmatrix}-1\\0\end{bmatrix},\quad \mathbf{v}=\begin{bmatrix}1\\-1\end{bmatrix},\quad y=1
$$

### (a) 计算 $\mathbf{z},\mathbf{h},\hat{y},L$

我按照前向传播的计算图分步代入计算：

**第一步：计算隐藏层线性输入 $\mathbf{z}$。**
转置矩阵 $\mathbf{W}^{\top} = \begin{bmatrix}1&0\\0&0\end{bmatrix}$，先算矩阵向量乘积：

$$
\mathbf{W}^{\top}\mathbf{x} = \begin{bmatrix}1&0\\0&0\end{bmatrix}\begin{bmatrix}1\\-1\end{bmatrix} = \begin{bmatrix}1\times 1 + 0\times(-1)\\0\times 1 + 0\times(-1)\end{bmatrix} = \begin{bmatrix}1\\0\end{bmatrix}
$$

加上偏置向量 $\mathbf{b}$：

$$
\mathbf{z} = \mathbf{W}^{\top}\mathbf{x} + \mathbf{b} = \begin{bmatrix}1\\0\end{bmatrix} + \begin{bmatrix}-1\\0\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}
$$

**第二步：计算激活值 $\mathbf{h}$。**
因为 $\sigma(0) = \frac{1}{1 + e^0} = \frac{1}{2} = 0.5$，逐元素激活得到：

$$
\mathbf{h} = \sigma(\mathbf{z}) = \begin{bmatrix}\sigma(0)\\\sigma(0)\end{bmatrix} = \begin{bmatrix}1/2\\1/2\end{bmatrix}
$$

**第三步：计算网络输出 $\hat{y}$。**

$$
\hat{y} = \mathbf{v}^{\top}\mathbf{h} = \begin{bmatrix}1 & -1\end{bmatrix}\begin{bmatrix}1/2\\1/2\end{bmatrix} = 1\times\frac12 + (-1)\times\frac12 = 0
$$

**第四步：计算标量损失 $L$。**
目标值 $y=1$，均方误差损失为：

$$
L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0 - 1)^2 = \frac{1}{2} = 0.5
$$

$$
\boxed{\mathbf{z} = \begin{bmatrix}0\\0\end{bmatrix},\qquad \mathbf{h} = \begin{bmatrix}1/2\\1/2\end{bmatrix},\qquad \hat{y} = 0,\qquad L = \frac{1}{2}}
$$

---

### (b) 链式法则推导与梯度计算

先建立反向传播的通用微分链式法则关系，再代入题设数值。

#### 1. 一般传播关系推导

损失标量对预测输出 $\hat{y}$ 的导数为：

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y
$$

- **对输出层权重 $\mathbf{v}$ 的梯度：**
  由 $\hat{y} = \mathbf{v}^{\top}\mathbf{h} = \sum_{k} v_k h_k$，对第 $k$ 个分量有 $\frac{\partial \hat{y}}{\partial v_k} = h_k$。根据链式法则：

  $$
  \frac{\partial L}{\partial v_k} = \frac{\partial L}{\partial \hat{y}}\frac{\partial \hat{y}}{\partial v_k} = (\hat{y} - y) h_k
  $$

  写成向量形式即为：

  $$
  \frac{\partial L}{\partial \mathbf{v}} = (\hat{y} - y)\mathbf{h}
  $$

- **对隐藏层线性输入 $\mathbf{z}$ 的误差向量 $\boldsymbol{\delta}$：**
  首先损失对激活状态 $\mathbf{h}$ 的梯度为 $\frac{\partial L}{\partial \mathbf{h}} = (\hat{y}-y)\mathbf{v}$。
  因为 $h_k = \sigma(z_k)$，其导数为 $\frac{\partial h_k}{\partial z_k} = \sigma'(z_k) = \sigma(z_k)(1-\sigma(z_k))$，不同分量之间无耦合（$\frac{\partial h_k}{\partial z_j} = 0, j\neq k$）。
  记误差向量 $\boldsymbol{\delta} = \frac{\partial L}{\partial \mathbf{z}} \in \mathbb{R}^2$：

  $$
  \delta_k = \frac{\partial L}{\partial z_k} = \frac{\partial L}{\partial h_k}\frac{\partial h_k}{\partial z_k} = (\hat{y}-y)v_k \sigma'(z_k)
  $$

  写成向量逐元素积（Hadamard 积）或对角阵形式：

  $$
  \boldsymbol{\delta} = \frac{\partial L}{\partial \mathbf{z}} = (\hat{y}-y)\mathbf{v} \odot \sigma'(\mathbf{z}) = \operatorname{diag}(\sigma'(\mathbf{z}))\mathbf{v}(\hat{y}-y)
  $$

- **对隐藏层偏置 $\mathbf{b}$ 的梯度：**
  由于 $z_k = \sum_{j} W_{jk} x_j + b_k$，直接求导得 $\frac{\partial z_k}{\partial b_k} = 1$，且 $\frac{\partial z_j}{\partial b_k} = 0\ (j\neq k)$。因此：

  $$
  \frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}} = \boldsymbol{\delta} = (\hat{y}-y)\mathbf{v}\odot\sigma'(\mathbf{z})
  $$

- **对权重矩阵 $\mathbf{W}$ 的梯度：**
  提示指出 $(\partial L/\partial \mathbf{W})_{ij} = \frac{\partial L}{\partial W_{ij}}$。
  注意到 $z_k = \sum_{l} W_{lk}x_l + b_k$，故 $\frac{\partial z_k}{\partial W_{ij}} = \begin{cases}x_i, & k=j\\ 0, & k\neq j\end{cases}$。由多元链式法则：

  $$
  \frac{\partial L}{\partial W_{ij}} = \sum_{k} \frac{\partial L}{\partial z_k}\frac{\partial z_k}{\partial W_{ij}} = \frac{\partial L}{\partial z_j} x_i = x_i \delta_j
  $$

  根据外积定义，矩阵形式恰为列向量 $\mathbf{x}$ 与行向量 $\boldsymbol{\delta}^{\top}$ 的外积：

  $$
  \frac{\partial L}{\partial \mathbf{W}} = \mathbf{x}\boldsymbol{\delta}^{\top} = \mathbf{x}\big((\hat{y}-y)\mathbf{v}\odot\sigma'(\mathbf{z})\big)^{\top}
  $$

#### 2. 数值代入计算

将 (a) 问结果 $\hat{y}=0, y=1 \implies \hat{y}-y = -1$ 代入。
在 $\mathbf{z} = \begin{bmatrix}0\\0\end{bmatrix}$ 处：

$$
\sigma'(0) = \sigma(0)(1-\sigma(0)) = \frac12\left(1-\frac12\right) = \frac14 = 0.25
$$

故 $\sigma'(\mathbf{z}) = \begin{bmatrix}1/4\\1/4\end{bmatrix}$。

1. **计算 $\frac{\partial L}{\partial \mathbf{v}}$：**

   $$
   \frac{\partial L}{\partial \mathbf{v}} = (\hat{y}-y)\mathbf{h} = (-1)\begin{bmatrix}1/2\\1/2\end{bmatrix} = \begin{bmatrix}-1/2\\-1/2\end{bmatrix}
   $$

2. **计算误差向量 $\boldsymbol{\delta}$ 与 $\frac{\partial L}{\partial \mathbf{b}}$：**

   $$
   \boldsymbol{\delta} = (-1)\begin{bmatrix}1\\-1\end{bmatrix}\odot\begin{bmatrix}1/4\\1/4\end{bmatrix} = \begin{bmatrix}-1/4\\1/4\end{bmatrix}
   $$

   因此：

   $$
   \frac{\partial L}{\partial \mathbf{b}} = \boldsymbol{\delta} = \begin{bmatrix}-1/4\\1/4\end{bmatrix}
   $$

3. **计算 $\frac{\partial L}{\partial \mathbf{W}}$：**
   代入 $\mathbf{x} = \begin{bmatrix}1\\-1\end{bmatrix}$ 与 $\boldsymbol{\delta}^{\top} = \begin{bmatrix}-1/4 & 1/4\end{bmatrix}$：

   $$
   \frac{\partial L}{\partial \mathbf{W}} = \begin{bmatrix}1\\-1\end{bmatrix}\begin{bmatrix}-1/4 & 1/4\end{bmatrix} = \begin{bmatrix}1\times(-1/4) & 1\times(1/4)\\(-1)\times(-1/4) & (-1)\times(1/4)\end{bmatrix} = \begin{bmatrix}-1/4 & 1/4\\1/4 & -1/4\end{bmatrix}
   $$

$$
\boxed{\frac{\partial L}{\partial \mathbf{v}} = \begin{bmatrix}-1/2\\-1/2\end{bmatrix},\qquad \frac{\partial L}{\partial \mathbf{b}} = \begin{bmatrix}-1/4\\1/4\end{bmatrix},\qquad \frac{\partial L}{\partial \mathbf{W}} = \begin{bmatrix}-1/4 & 1/4\\1/4 & -1/4\end{bmatrix}}
$$

*(数值已在 `verify.py` 中通过 PyTorch Autograd 及中心有限差分双向严格校验)*

---

### (c)(i) 证明激活导数连乘上界

**证明：**

第一步，考察单个实数 $t\in\mathbb{R}$ 处的 sigmoid 导数：

$$
\sigma'(t) = \frac{d}{dt}\left(\frac{1}{1+e^{-t}}\right) = \frac{e^{-t}}{(1+e^{-t})^2} = \sigma(t)\big(1-\sigma(t)\big)
$$

第二步，因为 $e^{-t} > 0$，所以 $\sigma(t) = \frac{1}{1+e^{-t}} \in (0, 1)$。
令 $u = \sigma(t) \in (0, 1)$，则导数可表示为关于 $u$ 的二次多项式 $g(u) = u(1-u)$。
对 $g(u)$ 配方：

$$
g(u) = -u^2 + u = -\left(u - \frac{1}{2}\right)^2 + \frac{1}{4}
$$

当且仅当 $u = \frac{1}{2}$（对应 $t = 0$）时取得唯一最大值 $\frac{1}{4}$。
同时，由于 $u\in(0, 1)$，故 $u(1-u) > 0$。因此对任意实数 $t\in\mathbb{R}$ 恒有：

$$
0 < \sigma'(t) \le \frac{1}{4}
$$

第三步，对任意正整数 $m$ 以及任意实数 $z_1, \dots, z_m \in \mathbb{R}$，连乘项各项均为正，因此：

$$
\left|\prod_{\ell=1}^m \sigma'(z_\ell)\right| = \prod_{\ell=1}^m \sigma'(z_\ell) \le \prod_{\ell=1}^m \frac{1}{4} = \left(\frac{1}{4}\right)^m = 4^{-m}
$$

当且仅当 $z_1 = z_2 = \dots = z_m = 0$ 时等号成立。证毕。$\blacksquare$

---

### (c)(ii) 第一层权重梯度的范数上界与 $s$ 的范围

**证明：**

**第一步：写出多层网络第一层权重的反向传播链式关系。**
在 $m$ 层网络中，第 $\ell$ 层的线性输入为 $\mathbf{z}^{(\ell)} = \mathbf{W}_\ell^{\top}\mathbf{h}^{(\ell-1)} + \mathbf{b}_\ell$（其中 $\mathbf{h}^{(0)}=\mathbf{x}$），激活值为 $\mathbf{h}^{(\ell)} = \sigma(\mathbf{z}^{(\ell)})$。
记各层激活导数对角矩阵为：

$$
\boldsymbol{\Sigma}_\ell = \operatorname{diag}\big(\sigma'(\mathbf{z}^{(\ell)})\big)
$$

顶层输出为 $\hat{y} = \mathbf{v}^{\top}\mathbf{h}^{(m)}$，损失标量关于第 $m$ 层线性输入的误差向量为：

$$
\boldsymbol{\delta}^{(m)} = \frac{\partial L}{\partial \mathbf{z}^{(m)}} = \boldsymbol{\Sigma}_m \frac{\partial L}{\partial \mathbf{h}^{(m)}} = \boldsymbol{\Sigma}_m (\hat{y}-y)\mathbf{v}
$$

对于中间层 $\ell = m, m-1, \dots, 2$：

$$
\frac{\partial L}{\partial \mathbf{h}^{(\ell-1)}} = \mathbf{W}_\ell \frac{\partial L}{\partial \mathbf{z}^{(\ell)}} = \mathbf{W}_\ell \boldsymbol{\delta}^{(\ell)}
$$

$$
\boldsymbol{\delta}^{(\ell-1)} = \frac{\partial L}{\partial \mathbf{z}^{(\ell-1)}} = \boldsymbol{\Sigma}_{\ell-1} \frac{\partial L}{\partial \mathbf{h}^{(\ell-1)}} = \boldsymbol{\Sigma}_{\ell-1}\mathbf{W}_\ell \boldsymbol{\delta}^{(\ell)}
$$

沿此递推关系自顶向下展开直至第一层：

$$
\boldsymbol{\delta}^{(1)} = \boldsymbol{\Sigma}_1 \mathbf{W}_2 \boldsymbol{\Sigma}_2 \mathbf{W}_3 \cdots \boldsymbol{\Sigma}_{m-1} \mathbf{W}_m \boldsymbol{\Sigma}_m (\hat{y}-y)\mathbf{v}
$$

由于第一层线性变换为 $\mathbf{z}^{(1)} = \mathbf{W}_1^{\top}\mathbf{x} + \mathbf{b}_1$，由 (b) 问推导知第一层权重的梯度为外积：

$$
\frac{\partial L}{\partial \mathbf{W}_1} = \mathbf{x}(\boldsymbol{\delta}^{(1)})^{\top}
$$

**第二步：利用矩阵算子范数放缩。**
利用题目给定的外积范数性质 $\|\mathbf{a}\mathbf{b}^{\top}\|_2 \le \|\mathbf{a}\|_2 \|\mathbf{b}\|_2$：

$$
\left\|\frac{\partial L}{\partial \mathbf{W}_1}\right\|_2 \le \|\mathbf{x}\|_2 \|\boldsymbol{\delta}^{(1)}\|_2
$$

由相容矩阵范数的次乘性 $\|\mathbf{A}\mathbf{u}\|_2 \le \|\mathbf{A}\|_2 \|\mathbf{u}\|_2$ 及 $\|\mathbf{A}\mathbf{B}\|_2 \le \|\mathbf{A}\|_2 \|\mathbf{B}\|_2$：

$$
\|\boldsymbol{\delta}^{(1)}\|_2 \le \left(\prod_{\ell=1}^m \|\boldsymbol{\Sigma}_\ell\|_2\right) \left(\prod_{\ell=2}^m \|\mathbf{W}_\ell\|_2\right) \|(\hat{y}-y)\mathbf{v}\|_2
$$

**第三步：计算各因子的上界。**
1. 对角矩阵 $\boldsymbol{\Sigma}_\ell$ 的谱范数为其对角元绝对值的最大值：
   由 (c)(i) 问结论，对任意 $z$ 均有 $0 < \sigma'(z) \le \frac{1}{4}$，因此：

   $$
   \|\boldsymbol{\Sigma}_\ell\|_2 = \max_i \sigma'(z_{\ell, i}) \le \frac{1}{4}
   $$

   从第 1 层到第 $m$ 层共计 $m$ 个激活导数矩阵，其谱范数乘积满足：

   $$
   \prod_{\ell=1}^m \|\boldsymbol{\Sigma}_\ell\|_2 \le \left(\frac{1}{4}\right)^m
   $$

2. 权重矩阵 $\mathbf{W}_2, \dots, \mathbf{W}_m$ 共计 $m-1$ 个，每个满足 $\|\mathbf{W}_\ell\|_2 \le s$，故其乘积满足：

   $$
   \prod_{\ell=2}^m \|\mathbf{W}_\ell\|_2 \le s^{m-1}
   $$

3. 题设输出误差满足 $\|(\hat{y}-y)\mathbf{v}\|_2 \le G$。

将上述三项上界代回 $\|\boldsymbol{\delta}^{(1)}\|_2$：

$$
\|\boldsymbol{\delta}^{(1)}\|_2 \le \left(\frac{1}{4}\right)^m s^{m-1} G = \frac{G}{4}\left(\frac{s}{4}\right)^{m-1}
$$

再乘以 $\|\mathbf{x}\|_2$，即证得：

$$
\left\|\frac{\partial L}{\partial \mathbf{W}_1}\right\|_2 \le \frac{G\|\mathbf{x}\|_2}{4}\left(\frac{s}{4}\right)^{m-1}
$$

**第四步：求使该上界在 $m\to\infty$ 时趋于零的 $s$ 的范围。**
该上界中依赖层数 $m$ 的唯一步进因子是指数序列 $\left(\frac{s}{4}\right)^{m-1}$，系数 $\frac{G\|\mathbf{x}\|_2}{4} > 0$ 与 $m$ 无关。
当且仅当底数绝对值严格小于 1 时，几何级数项趋于 0：

$$
\left|\frac{s}{4}\right| < 1 \iff -4 < s < 4
$$

题目已给定常数 $s > 0$，因此 $s$ 的取值范围为：

$$
\boxed{0 < s < 4 \quad (\text{即 } s \in (0, 4))}
$$

**直观解释：** 反向传播每穿越一个隐藏层，线性权重连接最多将误差信号放大 $s$ 倍，而 sigmoid 导数最多为 $\frac{1}{4}$。二者相乘构成每层的收缩因子 $\frac{s}{4}$。当 $s < 4$ 时，收缩因子严格小于 1，梯度上界随层数呈几何级数衰减，导致浅层梯度消失。

---

### (c)(iii) 表达能力限制还是优化困难？

**结论：主要属于优化困难（Optimization Difficulty）。**

**理由如下：**

1. **从表达能力（模型容量）看：**
   根据神经网络万能近似定理（Universal Approximation Theorem），具有非线性激活函数（包含 sigmoid）的前馈神经网络在神经元数量足够时，可以以任意精度逼近紧集上的连续函数。因此网络的函数容量并不受限，理论上存在可拟合目标函数的参数配置。
2. **从优化过程看：**
   如 (c)(ii) 所证，sigmoid 激活函数导数上界为 $1/4$，导致反向传播到浅层时梯度范数上界按 $(s/4)^{m-1}$ 指数衰减。基于梯度的一阶优化算法在浅层权重上的更新量趋近于零，浅层特征提取难以根据输出端误差进行有效更新。
3. **总结：**
   网络容量足以逼近目标函数，但一阶梯度优化算法因浅层梯度衰减难以有效更新参数。因此，梯度消失属于**优化困难**，而非表达能力受限。

---

## 2 K-Means：目标下降与解的质量

样本 $x_1, \dots, x_n \in \mathbb{R}^d$，$K$ 个聚类中心 $\mu_1, \dots, \mu_K \in \mathbb{R}^d$，$z_i \in \{1,\dots,K\}$。目标函数：

$$
J(z, \mu) = \sum_{i=1}^n \|x_i - \mu_{z_i}\|^2 = \sum_{k=1}^K \sum_{i\in C_k} \|x_i - \mu_k\|^2
$$

### (a) 证明均值最小化簇内平方误差和

**证明：**

设 $C_k$ 为任意非空簇，$|C_k| \ge 1$，样本均值为 $\bar{x}_k = \frac{1}{|C_k|}\sum_{i\in C_k} x_i$。
对于任意选取的中心 $u \in \mathbb{R}^d$，我们引入恒等变换，将样本点与 $u$ 的差向量分解为“相对于均值的差”与“均值相对于 $u$ 的差”：

$$
x_i - u = (x_i - \bar{x}_k) + (\bar{x}_k - u)
$$

对其欧氏范数平方展开：

$$
\|x_i - u\|^2 = \big\|(x_i - \bar{x}_k) + (\bar{x}_k - u)\big\|^2 = \|x_i - \bar{x}_k\|^2 + 2(x_i - \bar{x}_k)^{\top}(\bar{x}_k - u) + \|\bar{x}_k - u\|^2
$$

对该簇内所有样本 $i \in C_k$ 求和：

$$
\sum_{i\in C_k} \|x_i - u\|^2 = \sum_{i\in C_k} \|x_i - \bar{x}_k\|^2 + 2\left(\sum_{i\in C_k}(x_i - \bar{x}_k)\right)^{\top}(\bar{x}_k - u) + \sum_{i\in C_k} \|\bar{x}_k - u\|^2
$$

考察中间的交叉乘积项，由均值的定义：

$$
\sum_{i\in C_k} (x_i - \bar{x}_k) = \sum_{i\in C_k} x_i - |C_k|\bar{x}_k = |C_k|\bar{x}_k - |C_k|\bar{x}_k = \mathbf{0}
$$

因此交叉项恒等于零。最后项与求和指标 $i$ 无关，求和后为 $|C_k|\|\bar{x}_k - u\|^2$。我们得到恒等分解式：

$$
\sum_{i\in C_k} \|x_i - u\|^2 = \sum_{i\in C_k} \|x_i - \bar{x}_k\|^2 + |C_k|\|\bar{x}_k - u\|^2
$$

由于 $|C_k| \ge 1$ 且欧氏距离平方恒满足 $\|\bar{x}_k - u\|^2 \ge 0$，第二项非负，故：

$$
\sum_{i\in C_k} \|x_i - \bar{x}_k\|^2 \le \sum_{i\in C_k} \|x_i - u\|^2
$$

等号成立当且仅当 $\|\bar{x}_k - u\|^2 = 0 \iff u = \bar{x}_k$。证毕。$\blacksquare$

---

### (b) 证明目标值单调下降与收敛

**证明：**

一次完整迭代分为**分配步（Assignment Step）**与**更新步（Update Step）**：

**第一步：分配步（固定中心 $\mu^{(t)}$，更新分配 $z^{(t+1)}$）。**
算法将每个样本分配到离其最近的中心（并列时选标号最小者）：

$$
z_i^{(t+1)} = \arg\min_{k\in\{1,\dots,K\}} \|x_i - \mu_k^{(t)}\|^2
$$

由最小值的定义，对每个样本 $i \in \{1,\dots,n\}$，新分配下的距离平方必然不大于原分配下的距离平方：

$$
\|x_i - \mu_{z_i^{(t+1)}}^{(t)}\|^2 \le \|x_i - \mu_{z_i^{(t)}}^{(t)}\|^2
$$

对全部 $n$ 个样本求和，得：

$$
J(z^{(t+1)}, \mu^{(t)}) = \sum_{i=1}^n \|x_i - \mu_{z_i^{(t+1)}}^{(t)}\|^2 \le \sum_{i=1}^n \|x_i - \mu_{z_i^{(t)}}^{(t)}\|^2 = J(z^{(t)}, \mu^{(t)})
$$

**第二步：更新步（固定分配 $z^{(t+1)}$，更新中心 $\mu^{(t+1)}$）。**
此时各簇的样本集确定为 $C_k^{(t+1)} = \{i : z_i^{(t+1)} = k\}$。
- 若 $C_k^{(t+1)} \neq \emptyset$，中心更新为该簇的均值 $\mu_k^{(t+1)} = \bar{x}_k^{(t+1)}$。根据 (a) 问的结论，对旧中心 $u = \mu_k^{(t)}$ 成立：

  $$
  \sum_{i\in C_k^{(t+1)}} \|x_i - \mu_k^{(t+1)}\|^2 \le \sum_{i\in C_k^{(t+1)}} \|x_i - \mu_k^{(t)}\|^2
  $$

- 若 $C_k^{(t+1)} = \emptyset$，按规则中心保持不变 $\mu_k^{(t+1)} = \mu_k^{(t)}$，对应项为空集求和（为 0），不等式 $\le$ 同样成立。

对所有 $K$ 个簇求和：

$$
J(z^{(t+1)}, \mu^{(t+1)}) = \sum_{k=1}^K \sum_{i\in C_k^{(t+1)}} \|x_i - \mu_k^{(t+1)}\|^2 \le \sum_{k=1}^K \sum_{i\in C_k^{(t+1)}} \|x_i - \mu_k^{(t)}\|^2 = J(z^{(t+1)}, \mu^{(t)})
$$

**第三步：组合得到单步单调非增性。**
串联分配步与更新步的不等式：

$$
J(z^{(t+1)}, \mu^{(t+1)}) \le J(z^{(t+1)}, \mu^{(t)}) \le J(z^{(t)}, \mu^{(t)})
$$

**第四步：目标值序列收敛的理由。**
1. **单调性**：标量目标值序列 $\{J(z^{(t)}, \mu^{(t)})\}_{t=1}^\infty$ 单调非增；
2. **有下界**：对于任意配置，目标函数为各距离平方之和，恒有 $J(z, \mu) \ge 0$，即 0 是其天然下界；
3. **单调有界收敛原理**：根据实分析中的单调有界定理（Monotone Convergence Theorem），任何单调递减且有下界的实数序列必然收敛到某一有限实数极限 $J^* \ge 0$。

*(补充说明：在有限样本集上，所有可能的划分方案总数至多为 $K^n$ 种。若每次划分更新都使目标值严格下降，算法不会重复访问同一划分，因此在有限步内收敛至不动点。)*

---

### (c) 验证两种分簇均为不动点及其启示

给定 4 个数据点：

$$
x_1 = (-2, -1),\quad x_2 = (-2, 1),\quad x_3 = (2, -1),\quad x_4 = (2, 1)
$$

我逐条对状态 A 与状态 B 执行“分配 → 均值更新”的完整迭代验证。

#### 1. 验证状态 A（左右分簇）是不动点

初始中心为：$\mu_1 = (-2, 0)$，$\mu_2 = (2, 0)$。

- **分配步：** 计算 4 个样本到两中心的距离平方 $\|x - \mu\|^2$：
  - $x_1 = (-2, -1)$：$\|x_1 - \mu_1\|^2 = 0^2 + (-1)^2 = 1$；$\|x_1 - \mu_2\|^2 = (-4)^2 + (-1)^2 = 17$。$1 < 17 \implies z_1 = 1$。
  - $x_2 = (-2, 1)$：$\|x_2 - \mu_1\|^2 = 0^2 + 1^2 = 1$；$\|x_2 - \mu_2\|^2 = (-4)^2 + 1^2 = 17$。$1 < 17 \implies z_2 = 1$。
  - $x_3 = (2, -1)$：$\|x_3 - \mu_1\|^2 = 4^2 + (-1)^2 = 17$；$\|x_3 - \mu_2\|^2 = 0^2 + (-1)^2 = 1$。$1 < 17 \implies z_3 = 2$。
  - $x_4 = (2, 1)$：$\|x_4 - \mu_1\|^2 = 4^2 + 1^2 = 17$；$\|x_4 - \mu_2\|^2 = 0^2 + 1^2 = 1$。$1 < 17 \implies z_4 = 2$。

  分簇结果为 $C_1 = \{x_1, x_2\}$（左侧点集），$C_2 = \{x_3, x_4\}$（右侧点集）。
- **更新步：** 计算各簇新均值：
  - $\mu_1^{\text{new}} = \frac{x_1 + x_2}{2} = \frac{(-2, -1) + (-2, 1)}{2} = (-2, 0) = \mu_1$；
  - $\mu_2^{\text{new}} = \frac{x_3 + x_4}{2} = \frac{(2, -1) + (2, 1)}{2} = (2, 0) = \mu_2$。

新中心与原中心完全重合，分簇方案亦不变，目标值为 $J_A = 1+1+1+1 = 4$。**状态 A 为不动点。**

#### 2. 验证状态 B（上下分簇）是不动点

初始中心为：$\mu_1 = (0, -1)$，$\mu_2 = (0, 1)$。

- **分配步：** 计算 4 个样本到两中心的距离平方：
  - $x_1 = (-2, -1)$：$\|x_1 - \mu_1\|^2 = (-2)^2 + 0^2 = 4$；$\|x_1 - \mu_2\|^2 = (-2)^2 + (-2)^2 = 8$。$4 < 8 \implies z_1 = 1$。
  - $x_2 = (-2, 1)$：$\|x_2 - \mu_1\|^2 = (-2)^2 + 2^2 = 8$；$\|x_2 - \mu_2\|^2 = (-2)^2 + 0^2 = 4$。$4 < 8 \implies z_2 = 2$。
  - $x_3 = (2, -1)$：$\|x_3 - \mu_1\|^2 = 2^2 + 0^2 = 4$；$\|x_3 - \mu_2\|^2 = 2^2 + (-2)^2 = 8$。$4 < 8 \implies z_3 = 1$。
  - $x_4 = (2, 1)$：$\|x_4 - \mu_1\|^2 = 2^2 + 2^2 = 8$；$\|x_4 - \mu_2\|^2 = 2^2 + 0^2 = 4$。$4 < 8 \implies z_4 = 2$。

  分簇结果为 $C_1 = \{x_1, x_3\}$（下侧点集），$C_2 = \{x_2, x_4\}$（上侧点集）。
- **更新步：** 计算各簇新均值：
  - $\mu_1^{\text{new}} = \frac{x_1 + x_3}{2} = \frac{(-2, -1) + (2, -1)}{2} = (0, -1) = \mu_1$；
  - $\mu_2^{\text{new}} = \frac{x_2 + x_4}{2} = \frac{(-2, 1) + (2, 1)}{2} = (0, 1) = \mu_2$。

新中心与原中心完全重合，分簇方案亦不变，目标值为 $J_B = 4+4+4+4 = 16$。**状态 B 为不动点。**

#### 3. 对初始化影响的讨论

1. **存在多个局部极小**：这 4 个样本在横向跨度为 4，纵向跨度为 2。全局最优解为左右切分的状态 A（$J_A=4$）；而状态 B 同样满足不动点条件，其目标值为 $J_B = 16$（为状态 A 的 4 倍）。
2. **对初始化的敏感性**：K-Means 目标函数非凸，存在多个局部极小不动点。若初始中心落在状态 B 对应的区域，算法迭代一步后即收敛至状态 B，无法自行转移至全局更优的状态 A。
3. **初始化策略的必要性**：这说明标准 K-Means 的收敛结果显著依赖初始中心。为了避免陷入次优解，通常需要进行多次随机重启，或使用基于概率分散的初始化方法（如 K-Means++）。

---

### (d) 单调性 $J_{K+1}^* \le J_K^*$、$J_n^* = 0$ 与簇数选择

**证明：**

1. **证明 $J_{K+1}^* \le J_K^*$：**
   定义全局最优训练目标为：

   $$
   J_K^* = \min_{\mu_1, \dots, \mu_K} \sum_{i=1}^n \min_{k\in\{1,\dots,K\}} \|x_i - \mu_k\|^2
   $$

   设某组达到该最小值的最优中心为 $(\mu_1^*, \dots, \mu_K^*)$。
   因为允许中心重复或未使用，我们构造一组含有 $K+1$ 个中心的候选解：前 $K$ 个中心取原最优中心，第 $K+1$ 个中心直接复制其中的任意一个（例如取 $\mu_{K+1} = \mu_K^*$）：

   $$
   \tilde{\boldsymbol{\mu}} = (\mu_1^*, \dots, \mu_K^*, \mu_K^*)
   $$

   对任意样本 $x_i$，由于新增加的中心与已有中心重合，样本到这 $K+1$ 个中心的最小距离与到原 $K$ 个中心的最小距离恒等：

   $$
   \min_{k\in\{1,\dots,K+1\}} \|x_i - \tilde{\mu}_k\|^2 = \min\left(\min_{k\in\{1,\dots,K\}} \|x_i - \mu_k^*\|^2,\ \|x_i - \mu_K^*\|^2\right) = \min_{k\in\{1,\dots,K\}} \|x_i - \mu_k^*\|^2
   $$

   因此这组候选中心达到的目标函数值恰为 $J_K^*$：

   $$
   \sum_{i=1}^n \min_{k\in\{1,\dots,K+1\}} \|x_i - \tilde{\mu}_k\|^2 = \sum_{i=1}^n \min_{k\in\{1,\dots,K\}} \|x_i - \mu_k^*\|^2 = J_K^*
   $$

   根据 $J_{K+1}^*$ 是所有 $K+1$ 个中心的全局下确界/最小值，必然有：

   $$
   J_{K+1}^* \le \sum_{i=1}^n \min_{k\in\{1,\dots,K+1\}} \|x_i - \tilde{\mu}_k\|^2 = J_K^*
   $$

   即 $J_{K+1}^* \le J_K^*$。

2. **证明 $J_n^* = 0$：**
   当 $K = n$ 时，构造候选中心为每一个数据样本本身：$\mu_i = x_i\ (i=1,\dots,n)$。
   对任意样本 $x_i$，其自身就是一个中心，到该中心的距离为 $\|x_i - \mu_i\|^2 = \|x_i - x_i\|^2 = 0$。
   因此：

   $$
   0 \le J_n^* \le \sum_{i=1}^n \min_{k\in\{1,\dots,n\}} \|x_i - \mu_k\|^2 \le \sum_{i=1}^n \|x_i - \mu_i\|^2 = 0 \implies J_n^* = 0
   $$

3. **为什么不能仅按最小训练目标选择簇数 $K$？**
   上述两条性质说明：最优训练目标 $J_K^*$ 是关于簇数 $K$ 的**单调递减函数**，并且在 $K = n$ 时平凡地达到全局绝对最小值 0。
   如果单纯以“最小化训练损失 $J_K$”作为模型选择准则，该准则必然且唯一地选择最大允许的簇数 $K = n$（即每个数据点各自成立一个独立的簇）。
   - **丧失聚类意义**：聚类的核心目标是数据压缩、模式发现与信息提炼。将每个样本作为一个簇完全没有对数据进行任何归纳，聚类退化为机械记忆；
   - **模型严重过拟合**：增加 $K$ 等同于增加模型的自由参数与容量，训练损失单调下降是模型复杂度增加的自然数学推论，并不代表更优的真实泛化结构。
   因此，确定簇数 $K$ 必须在“训练误差”与“模型复杂度”之间进行惩罚权衡，例如使用肘部法则（Elbow Method，观测边际收益递减的拐点）、轮廓系数（Silhouette Analysis）、Gap Statistic 或信息准则（AIC/BIC）。

---

## 3 K-Means++：距离、样本数与离群点

已选中心集合为 $C$，样本到最近中心的距离记为 $D(x_i) = \min_{c\in C}\|x_i - c\|$，$D^2$ 抽样概率为：

$$
P(x_i) = \frac{D(x_i)^2}{\sum_j D(x_j)^2}
$$

### (a) 区域概率下界与大样本稀释效应

#### 1. 写出下一中心落入 $B$ 的精确概率

下一中心落入区域 $B$ 意味着抽选到样本点 $x \in B$。根据互不相交集合的概率可加性：

$$
\boxed{P(B) = \sum_{x\in B} P(x) = \frac{\sum_{x\in B} D(x)^2}{\sum_{x\in A} D(x)^2 + \sum_{x\in B} D(x)^2}}
$$

#### 2. 证明下界 $P(B) \ge \frac{sR^2}{mr^2 + sR^2}$

**证明：**
记两区域的距离平方和分别为 $S_A = \sum_{x\in A} D(x)^2$ 和 $S_B = \sum_{x\in B} D(x)^2$。
因为分母为正且 $S_B > 0$，可将精确概率化简为：

$$
P(B) = \frac{S_B}{S_A + S_B} = \frac{1}{1 + \frac{S_A}{S_B}}
$$

由题设条件：
- 对所有 $x\in A$，$D(x)^2 \le r^2$，且 $A$ 中含有 $m$ 个样本，故：

  $$
  S_A = \sum_{x\in A} D(x)^2 \le \sum_{x\in A} r^2 = m r^2
  $$

- 对所有 $x\in B$，$D(x)^2 \ge R^2$，且 $B$ 中含有 $s$ 个样本，故：

  $$
  S_B = \sum_{x\in B} D(x)^2 \ge \sum_{x\in B} R^2 = s R^2 > 0
  $$

两式相比得到比值的上界：

$$
\frac{S_A}{S_B} \le \frac{m r^2}{s R^2}
$$

由于函数 $f(t) = \frac{1}{1 + t}$ 在 $t \ge 0$ 上严格单调递减，代入比值上界可得 $P(B)$ 的下界：

$$
P(B) = \frac{1}{1 + \frac{S_A}{S_B}} \ge \frac{1}{1 + \frac{m r^2}{s R^2}} = \frac{s R^2}{m r^2 + s R^2}
$$

证毕。$\blacksquare$

#### 3. 当 $s\ll m$ 时，仅有 $R\gg r$ 能否保证 $P(B)$ 接近 1？

**结论：不能。**

**理由如下：**

我们将下界分子分母同除以 $sR^2$：

$$
P(B) \ge \frac{1}{1 + \frac{m}{s}\left(\frac{r}{R}\right)^2}
$$

要保证 $P(B)$ 接近 1，必须使分母中附加项远小于 1，即需要满足：

$$
\frac{m}{s}\left(\frac{r}{R}\right)^2 \ll 1 \iff \left(\frac{R}{r}\right)^2 \gg \frac{m}{s} \iff \frac{R}{r} \gg \sqrt{\frac{m}{s}}
$$

若仅仅满足 $R \gg r$（例如取距离比 $R/r = 10$，即 $(R/r)^2 = 100$），但区域样本数差距更为悬殊（例如 $s=1, m=10^4$，即 $m/s = 10^4$），则：

$$
\frac{m}{s}\left(\frac{r}{R}\right)^2 = 10^4 \times \frac{1}{100} = 100 \gg 1
$$

此时下界衰减为：

$$
P(B) \ge \frac{1}{1 + 100} = \frac{1}{101} \approx 0.0099 \ll 1
$$

这是样本规模对距离权重的累积效应：尽管区域 $B$ 中的单个样本到现有中心的距离大于区域 $A$（$R\gg r$），但当区域 $A$ 的样本量 $m$ 远大于 $s$ 时，区域 $A$ 内样本距离平方的总和 $\sum_{x\in A} D(x)^2$ 仍可能大于区域 $B$ 的总和 $\sum_{x\in B} D(x)^2$，使得下一次抽样落入区域 $B$ 的概率仍可能偏低。
因此，仅有 $R\gg r$ 不能保证 $P(B) \approx 1$；必须满足距离比值的平方能够抵消样本量的悬殊比例（即 $(R/r)^2 \gg m/s$），才能确保 $P(B)$ 接近 1。

---

### (b) 一维离群点抽样概率推导与分析

已选中心集合为 $C=\{0\}$，未选样本共 $2m+1$ 个：
- $m$ 个位于坐标 1，距离平方为 $D(1)^2 = |1 - 0|^2 = 1$；
- $m$ 个位于坐标 2，距离平方为 $D(2)^2 = |2 - 0|^2 = 4$；
- 1 个位于坐标 $R > 2$（离群点），距离平方为 $D(R)^2 = |R - 0|^2 = R^2$。

全部未选样本的距离平方和为：

$$
\sum_j D(x_j)^2 = m\times 1^2 + m\times 2^2 + 1\times R^2 = m + 4m + R^2 = 5m + R^2
$$

#### (b)(i) 求离群点概率并与均匀抽样比较

1. **$D^2$ 抽样选中离群点的概率：**

   $$
   \boxed{P_{\text{D2}}(R) = \frac{R^2}{5m + R^2}}
   $$

2. **在未选样本中均匀抽样的概率为：**

   $$
   P_{\text{uniform}} = \frac{1}{2m + 1}
   $$

3. **作差比较两者的概率大小：**

   $$
   P_{\text{D2}}(R) - P_{\text{uniform}} = \frac{R^2}{5m + R^2} - \frac{1}{2m + 1} = \frac{R^2(2m + 1) - (5m + R^2)}{(5m + R^2)(2m + 1)}
   $$

   分子化简展开：

   $$
   2m R^2 + R^2 - 5m - R^2 = 2m R^2 - 5m = m(2R^2 - 5)
   $$

   题目给定 $m$ 为正整数（$m \ge 1$）且 $R > 2$。因此：

   $$
   R > 2 \implies R^2 > 4 \implies 2R^2 - 5 > 2\times 4 - 5 = 3 > 0
   $$

   由于分母 $(5m + R^2)(2m + 1) > 0$，分子 $m(2R^2 - 5) \ge 3m > 0$，故两者之差恒为正：

   $$
   P_{\text{D2}}(R) - P_{\text{uniform}} = \frac{m(2R^2 - 5)}{(5m + R^2)(2m + 1)} > 0
   $$

   **结论：对任意正整数 $m$ 和任意 $R > 2$，$D^2$ 抽样选中离群点的概率严格大于均匀抽样的概率。**

**数值算例核对：**
取 $m=2, R=4$（未选样本共 $2\times 2 + 1 = 5$ 个）：
- 均匀抽样概率：$P_{\text{uniform}} = \frac{1}{5} = 0.2000$；
- $D^2$ 抽样概率：$P_{\text{D2}} = \frac{4^2}{5\times 2 + 4^2} = \frac{16}{10 + 16} = \frac{16}{26} = \frac{8}{13} \approx 0.615385$；
- 两者相差：$\frac{8}{13} - \frac{1}{5} = \frac{27}{65} \approx 0.415385 > 0$。$D^2$ 抽样将该样本的抽样概率从 $0.2000$ 提升至约 $0.6154$。

#### (b)(ii) 证明 $R^2 \ge 5m$ 时离群点被选概率至少为 $1/2$

**证明：**

将 $P_{\text{D2}}(R)$ 的分子分母同除以 $R^2$：

$$
P_{\text{D2}}(R) = \frac{1}{1 + \frac{5m}{R^2}}
$$

当条件 $R^2 \ge 5m$（$R^2 > 0$）满足时，两边同除以 $R^2$ 得：

$$
\frac{5m}{R^2} \le 1
$$

代入分母不等式：

$$
P_{\text{D2}}(R) = \frac{1}{1 + \frac{5m}{R^2}} \ge \frac{1}{1 + 1} = \frac{1}{2}
$$

当且仅当 $R^2 = 5m$ 时取等号。证毕。$\blacksquare$

---

## 4 GMM 与 EM

设样本 $x_1,\dots,x_n\in\mathbb{R}^d$，$z_i\in\{1,\dots,K\}$ 为隐变量，多元高斯密度函数为：

$$
\mathcal{N}(x\mid\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}\sqrt{\det\Sigma}}\exp\Big[-\frac12(x-\mu)^{\top}\Sigma^{-1}(x-\mu)\Big]
$$

模型参数 $\theta=\{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$，$\pi_k>0$，$\sum_{k=1}^K\pi_k=1$，$\Sigma_k \succ 0$。

### (a) 责任度推导与硬分配对比

#### 1. 责任度 $\gamma_{ik}^{(t)}$ 的推导

根据条件概率定义与 Bayes 公式，在当前参数 $\theta^{(t)}$ 下，样本 $x_i$ 属于第 $k$ 个成分的后验概率为：

$$
\gamma_{ik}^{(t)} = p(z_i = k \mid x_i, \theta^{(t)}) = \frac{p(x_i, z_i = k \mid \theta^{(t)})}{p(x_i \mid \theta^{(t)})}
$$

联合概率密度由先验与条件高斯相乘给出：

$$
p(x_i, z_i = k \mid \theta^{(t)}) = p(z_i = k \mid \theta^{(t)}) p(x_i \mid z_i = k, \theta^{(t)}) = \pi_k^{(t)} \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})
$$

边缘概率密度由全概率公式对离散隐变量 $z_i$ 求和给出：

$$
p(x_i \mid \theta^{(t)}) = \sum_{j=1}^K p(x_i, z_i = j \mid \theta^{(t)}) = \sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})
$$

两式相比即得责任度闭式解：

$$
\boxed{\gamma_{ik}^{(t)} = \frac{\pi_k^{(t)} \mathcal{N}(x_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x_i \mid \mu_j^{(t)}, \Sigma_j^{(t)})}}
$$

#### 2. 与 K-Means 硬分配的区别

1. **软分配（Soft Assignment） vs. 硬分配（Hard Assignment）：**
   - K-Means 采取 0-1 的硬切分：每个样本 $x_i$ 确定性地只归属于唯一的某一个簇，其指示变量 $z_{ik} \in \{0, 1\}$；
   - GMM 采取概率软分配：责任度 $\gamma_{ik}^{(t)} \in (0, 1)$ 是一个属于连续区间的置信度（满足 $\sum_k \gamma_{ik}^{(t)} = 1$），定量刻画了该样本由第 $k$ 个成分生成的后验隶属程度，允许样本以不同概率同时属于多个簇。
2. **考虑的几何与统计因素不同：**
   - K-Means 仅按各向同性的欧氏距离比较，隐式假设各个簇具有相同的球形先验方差；
   - GMM 的责任度综合权衡了三种因素：① 几何中心距离 $(x_i-\mu_k)$；② 簇先验规模 $\pi_k$；③ 协方差矩阵 $\Sigma_k$ 所蕴含的椭球轴向拉伸与相关性旋转。即便样本距某个中心在欧氏距离上更近，若该中心方差极小或先验很低，其后验责任度仍可能低于距离稍远但分布更宽的成分。

---

### (b) M 步参数更新公式推导

E 步后固定责任度 $\gamma_{ik}^{(t)}$，记有效样本数 $N_k^{(t)} = \sum_{i=1}^n \gamma_{ik}^{(t)}$。M 步最大化期望完全数据对数似然：

$$
Q(\theta\mid\theta^{(t)}) = \sum_{i=1}^n\sum_{k=1}^K \gamma_{ik}^{(t)} \log p(x_i, z_i=k\mid\theta)
$$

将联合概率对数展开：

$$
\log p(x_i, z_i=k\mid\theta) = \log\pi_k - \frac{d}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma_k - \frac{1}{2}(x_i - \mu_k)^{\top}\Sigma_k^{-1}(x_i - \mu_k)
$$

代入 $Q$ 并按参数完全解耦分离：

$$
Q(\theta\mid\theta^{(t)}) = \sum_{k=1}^K N_k^{(t)}\log\pi_k - \frac{1}{2}\sum_{k=1}^K\left[ N_k^{(t)}\log\det\Sigma_k + \sum_{i=1}^n \gamma_{ik}^{(t)}(x_i - \mu_k)^{\top}\Sigma_k^{-1}(x_i - \mu_k) \right] + \text{const}
$$

#### 1. 推导混合权重 $\pi_k^{(t+1)}$

仅提取含 $\pi_k$ 的项，目标为在约束 $\sum_{k=1}^K\pi_k = 1$ 下最大化 $\sum_{k=1}^K N_k^{(t)}\log\pi_k$。
引入 Lagrange 乘子 $\lambda$：

$$
\mathcal{L}(\pi, \lambda) = \sum_{k=1}^K N_k^{(t)}\log\pi_k - \lambda\left(\sum_{k=1}^K \pi_k - 1\right)
$$

对 $\pi_k$ 求偏导并令为 0：

$$
\frac{\partial \mathcal{L}}{\partial \pi_k} = \frac{N_k^{(t)}}{\pi_k} - \lambda = 0 \implies \pi_k = \frac{N_k^{(t)}}{\lambda}
$$

对所有 $k$ 求和利用归一化条件：

$$
1 = \sum_{k=1}^K \pi_k = \frac{1}{\lambda}\sum_{k=1}^K N_k^{(t)} = \frac{1}{\lambda}\sum_{i=1}^n \underbrace{\sum_{k=1}^K \gamma_{ik}^{(t)}}_{=1} = \frac{n}{\lambda} \implies \lambda = n
$$

代回得：

$$
\boxed{\pi_k^{(t+1)} = \frac{N_k^{(t)}}{n}}
$$

#### 2. 推导均值向量 $\mu_k^{(t+1)}$

仅提取含 $\mu_k$ 的二次型项：

$$
Q_k(\mu_k) = -\frac{1}{2}\sum_{i=1}^n \gamma_{ik}^{(t)} (x_i - \mu_k)^{\top}\Sigma_k^{-1}(x_i - \mu_k)
$$

对向量 $\mu_k$ 求梯度（利用 $\nabla_{\mu} (x-\mu)^{\top}\Sigma^{-1}(x-\mu) = -2\Sigma^{-1}(x-\mu)$）：

$$
\nabla_{\mu_k} Q_k = \sum_{i=1}^n \gamma_{ik}^{(t)} \Sigma_k^{-1}(x_i - \mu_k) = \Sigma_k^{-1}\left( \sum_{i=1}^n \gamma_{ik}^{(t)} x_i - \mu_k \sum_{i=1}^n \gamma_{ik}^{(t)} \right) = \Sigma_k^{-1}\left( \sum_{i=1}^n \gamma_{ik}^{(t)} x_i - N_k^{(t)}\mu_k \right)
$$

因为协方差矩阵正定，$\Sigma_k^{-1}$ 非奇异，令梯度为零向量：

$$
\sum_{i=1}^n \gamma_{ik}^{(t)} x_i - N_k^{(t)}\mu_k = \mathbf{0} \implies \boxed{\mu_k^{(t+1)} = \frac{1}{N_k^{(t)}}\sum_{i=1}^n \gamma_{ik}^{(t)} x_i}
$$

#### 3. 推导一般协方差矩阵 $\Sigma_k^{(t+1)}$

固定均值为更新值 $\mu_k^{(t+1)}$。记精度矩阵为对称正定矩阵 $A = \Sigma_k^{-1}$，则：

$$
\log\det\Sigma_k = -\log\det(\Sigma_k^{-1}) = -\log\det A
$$

利用提示给出的恒等式 $u^{\top}Au = \operatorname{tr}(Auu^{\top})$，令 $u_i = x_i - \mu_k^{(t+1)}$：

$$
\sum_{i=1}^n \gamma_{ik}^{(t)} (x_i - \mu_k^{(t+1)})^{\top} A (x_i - \mu_k^{(t+1)}) = \sum_{i=1}^n \gamma_{ik}^{(t)} \operatorname{tr}\big(A u_i u_i^{\top}\big) = \operatorname{tr}\left( A \sum_{i=1}^n \gamma_{ik}^{(t)} u_i u_i^{\top} \right) = \operatorname{tr}(A S_k)
$$

其中记对称加权散度阵为：

$$
S_k = \sum_{i=1}^n \gamma_{ik}^{(t)} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^{\top}
$$

将上式代入关于 $A$ 的目标函数：

$$
Q_k(A) = -\frac{1}{2}N_k^{(t)}(-\log\det A) - \frac{1}{2}\operatorname{tr}(A S_k) = \frac{1}{2}N_k^{(t)}\log\det A - \frac{1}{2}\operatorname{tr}(A S_k)
$$

利用提示给出的矩阵梯度公式 $\nabla_A \log\det A = A^{-1}$ 以及 $\nabla_A \operatorname{tr}(AB) = B$（此处 $B=S_k$ 为对称阵）：

$$
\nabla_A Q_k(A) = \frac{1}{2}N_k^{(t)}A^{-1} - \frac{1}{2}S_k
$$

令梯度为零矩阵：

$$
\frac{1}{2}N_k^{(t)}A^{-1} - \frac{1}{2}S_k = \mathbf{0} \implies A^{-1} = \frac{1}{N_k^{(t)}}S_k
$$

因为 $\Sigma_k = A^{-1}$，故协方差更新公式为：

$$
\boxed{\Sigma_k^{(t+1)} = \frac{1}{N_k^{(t)}}\sum_{i=1}^n \gamma_{ik}^{(t)} (x_i - \mu_k^{(t+1)})(x_i - \mu_k^{(t+1)})^{\top}}
$$

---

### (c) 证据下界（ELBO）与 Jensen 不等式

#### 1. 证明 $\ell(\theta) \ge \mathcal{L}(q, \theta)$

**证明：**
对每个样本 $x_i$，其观测对数似然可恒等改写为：

$$
\log p(x_i\mid\theta) = \log\sum_{k=1}^K p(x_i, z_i=k\mid\theta) = \log\sum_{k=1}^K q_i(k)\frac{p(x_i, z_i=k\mid\theta)}{q_i(k)}
$$

因为 $q_i(k) > 0$ 且 $\sum_{k=1}^K q_i(k) = 1$，求和项实质为离散分布 $q_i$ 下的数学期望：

$$
\log p(x_i\mid\theta) = \log \mathbb{E}_{q_i}\left[ \frac{p(x_i, z_i\mid\theta)}{q_i(z_i)} \right]
$$

由于 $f(x) = \log x$ 在 $(0, \infty)$ 上二阶导 $f''(x) = -1/x^2 < 0$，是严格凹函数。由 Jensen 不等式 $\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$：

$$
\log\sum_{k=1}^K q_i(k)\frac{p(x_i, z_i=k\mid\theta)}{q_i(k)} \ge \sum_{k=1}^K q_i(k)\log\frac{p(x_i, z_i=k\mid\theta)}{q_i(k)}
$$

对所有 $n$ 个独立样本求和：

$$
\ell(\theta) = \sum_{i=1}^n \log p(x_i\mid\theta) \ge \sum_{i=1}^n\sum_{k=1}^K q_i(k)\log\frac{p(x_i, z_i=k\mid\theta)}{q_i(k)} = \mathcal{L}(q, \theta)
$$

不等式证毕。

#### 2. 证明取等当且仅当 $q_i(k) = p(z_i = k \mid x_i, \theta)$ 对所有 $i, k$ 成立

**证明：**
我们利用条件概率公式 $p(x_i, z_i=k\mid\theta) = p(z_i=k\mid x_i,\theta) p(x_i\mid\theta)$ 将对数商展开：

$$
\log\frac{p(x_i, z_i=k\mid\theta)}{q_i(k)} = \log p(x_i\mid\theta) + \log\frac{p(z_i=k\mid x_i,\theta)}{q_i(k)} = \log p(x_i\mid\theta) - \log\frac{q_i(k)}{p(z_i=k\mid x_i,\theta)}
$$

代入单样本的求和式：

$$
\sum_{k=1}^K q_i(k)\log\frac{p(x_i, z_i=k\mid\theta)}{q_i(k)} = \log p(x_i\mid\theta)\underbrace{\sum_{k=1}^K q_i(k)}_{=1} - \sum_{k=1}^K q_i(k)\log\frac{q_i(k)}{p(z_i=k\mid x_i,\theta)}
$$

右边第二项正是分布 $q_i$ 与真实后验 $p(z_i\mid x_i,\theta)$ 之间的 Kullback-Leibler (KL) 散度：

$$
= \log p(x_i\mid\theta) - D_{\text{KL}}\big( q_i \,\|\, p(z_i\mid x_i,\theta) \big)
$$

对全部样本求和，得到等式关系：

$$
\ell(\theta) - \mathcal{L}(q, \theta) = \sum_{i=1}^n D_{\text{KL}}\big( q_i \,\|\, p(z_i\mid x_i,\theta) \big)
$$

根据 KL 散度的核心性质（亦由对数函数严格凹性决定）：
1. 对任意分布 $q_i$，恒有 $D_{\text{KL}}\big( q_i \,\|\, p(z_i\mid x_i,\theta) \big) \ge 0$；
2. 取等号 $D_{\text{KL}} = 0$ 当且仅当 $q_i$ 与 $p(z_i\mid x_i,\theta)$ 处处相同，即 $q_i(k) = p(z_i=k\mid x_i,\theta)\ (\forall k)$。

由于每一项非负，总和为 0 当且仅当每一项均为 0：

$$
\ell(\theta) = \mathcal{L}(q, \theta) \iff q_i(k) = p(z_i=k\mid x_i,\theta)\quad (\forall i=1,\dots,n,\ k=1,\dots,K)
$$

充要条件证毕。$\blacksquare$

---

### (d) 证明 EM 算法对数似然单调不减

**证明：**

在第 $t$ 次迭代中：

1. **E 步紧贴下界：**
   算法令 $q_i^{(t)}(k) = \gamma_{ik}^{(t)} = p(z_i = k \mid x_i, \theta^{(t)})$。
   根据 (c) 问结论，当辅助分布与当前后验精确相同时，KL 散度为零，下界在当前参数 $\theta^{(t)}$ 处精确等于真实对数似然：

   $$
   \mathcal{L}(q^{(t)}, \theta^{(t)}) = \ell(\theta^{(t)})
   $$

2. **M 步提升下界：**
   将证据下界改写为期望对数似然与熵之和：

   $$
   \mathcal{L}(q^{(t)}, \theta) = \sum_{i, k} q_i^{(t)}(k)\log p(x_i, z_i=k\mid\theta) - \sum_{i, k} q_i^{(t)}(k)\log q_i^{(t)}(k) = Q(\theta\mid\theta^{(t)}) + \sum_{i=1}^n H(q_i^{(t)})
   $$

   注意分布 $q^{(t)}$ 在 M 步是固定的，故熵项 $\sum_i H(q_i^{(t)})$ 与参数 $\theta$ 无关。
   因此，M 步最大化 $Q(\theta\mid\theta^{(t)})$ 完全等价于在固定 $q^{(t)}$ 下关于 $\theta$ 最大化下界 $\mathcal{L}(q^{(t)}, \theta)$。
   因为 $\theta^{(t+1)}$ 是该下界的最优解，自然满足：

   $$
   \mathcal{L}(q^{(t)}, \theta^{(t+1)}) \ge \mathcal{L}(q^{(t)}, \theta^{(t)})
   $$

3. **下界性质保证新参数似然不低于下界：**
   由 (c) 问下界定理对任意参数成立，特别地对新参数 $\theta^{(t+1)}$ 有：

   $$
   \ell(\theta^{(t+1)}) \ge \mathcal{L}(q^{(t)}, \theta^{(t+1)})
   $$

4. **串联单调不等式链：**
   综合上述三步：

   $$
   \ell(\theta^{(t+1)}) \ge \mathcal{L}(q^{(t)}, \theta^{(t+1)}) \ge \mathcal{L}(q^{(t)}, \theta^{(t)}) = \ell(\theta^{(t)})
   $$

由此严格证明：

$$
\ell(\theta^{(t+1)}) \ge \ell(\theta^{(t)})
$$

证毕。$\blacksquare$

---

### (e) $\Sigma_k = \sigma^2 I$ 时的极限行为分析

#### 1. 唯一最近中心下的责任度极限证明

当 $\Sigma_k = \sigma^2 I$ 时，高斯分布退化为各向同性高斯，密度为：

$$
\mathcal{N}(x_i \mid \mu_k, \sigma^2 I) = \frac{1}{(2\pi\sigma^2)^{d/2}}\exp\left( -\frac{\|x_i - \mu_k\|^2}{2\sigma^2} \right)
$$

将其代入责任度定义，系数 $\frac{1}{(2\pi\sigma^2)^{d/2}}$ 上下约去：

$$
\gamma_{ik} = \frac{\pi_k \exp\left( -\frac{\|x_i - \mu_k\|^2}{2\sigma^2} \right)}{\sum_{j=1}^K \pi_j \exp\left( -\frac{\|x_i - \mu_j\|^2}{2\sigma^2} \right)}
$$

分子分母同除以最大指数因子 $\exp\left( -\frac{\|x_i - \mu_{k_*}\|^2}{2\sigma^2} \right)$：

$$
\gamma_{ik} = \frac{\pi_k \exp\left( -\frac{\|x_i - \mu_k\|^2 - \|x_i - \mu_{k_*}\|^2}{2\sigma^2} \right)}{\pi_{k_*} + \sum_{j \neq k_*} \pi_j \exp\left( -\frac{\|x_i - \mu_j\|^2 - \|x_i - \mu_{k_*}\|^2}{2\sigma^2} \right)}
$$

已知 $k_*$ 是 $x_i$ 的**唯一**最近中心，对任意 $j \neq k_*$，距离差恒严格大于零：

$$
\Delta_{ij} \triangleq \|x_i - \mu_j\|^2 - \|x_i - \mu_{k_*}\|^2 > 0
$$

当 $\sigma^2 \to 0^+$ 时，$\frac{\Delta_{ij}}{2\sigma^2} \to +\infty$，对应的负指数项以指数级速度迅速衰减到 0：

$$
\lim_{\sigma^2 \to 0^+} \exp\left( -\frac{\Delta_{ij}}{2\sigma^2} \right) = 0
$$

由于各先验权重 $\pi_j > 0$ 为正实数，分母的极限为：

$$
\lim_{\sigma^2 \to 0^+} \left[ \pi_{k_*} + \sum_{j \neq k_*} \pi_j \exp\left( -\frac{\Delta_{ij}}{2\sigma^2} \right) \right] = \pi_{k_*} + 0 = \pi_{k_*} > 0
$$

- 对于唯一最近中心 $k = k_*$，分子的指数项为 $\exp(0) = 1$，故：

  $$
  \lim_{\sigma^2 \to 0^+} \gamma_{ik_*} = \frac{\pi_{k_*}\cdot 1}{\pi_{k_*}} = 1
  $$

- 对于其余中心 $k \neq k_*$，分子中的指数项趋于 0，故：

  $$
  \lim_{\sigma^2 \to 0^+} \gamma_{ik} = \frac{\pi_k\cdot 0}{\pi_{k_*}} = 0
  $$

证毕。$\blacksquare$

#### 2. 均值更新公式的极限

若所有样本的最近中心均唯一，记 $z_i^* = \arg\min_k \|x_i - \mu_k\|^2$ 为样本 $x_i$ 的硬分配目标簇。
由上一问结论，当 $\sigma^2 \to 0$ 时，责任度收敛为 Kronecker $\delta$ 符号指示函数：

$$
\gamma_{ik} \to \mathbb{I}(z_i^* = k) = \begin{cases}1, & k = z_i^*\\ 0, & k \neq z_i^*\end{cases}
$$

记第 $k$ 簇的样本集为 $C_k = \{i : z_i^* = k\}$。题设各中心均得到样本，故 $|C_k| \ge 1$。
分别考察有效样本数与加权和的极限：

$$
N_k = \sum_{i=1}^n \gamma_{ik} \to \sum_{i\in C_k} 1 = |C_k|
$$

$$
\sum_{i=1}^n \gamma_{ik} x_i \to \sum_{i\in C_k} x_i
$$

因此均值更新公式趋于：

$$
\lim_{\sigma^2 \to 0} \mu_k^{(t+1)} = \lim_{\sigma^2 \to 0} \frac{\sum_{i=1}^n \gamma_{ik} x_i}{N_k} = \frac{1}{|C_k|}\sum_{i\in C_k} x_i = \bar{x}_k
$$

**结论：均值更新趋于该簇样本均值 $\bar{x}_k$，与 K-Means 更新步的质心更新一致。**
这说明 K-Means 算法恰好是高斯混合模型在方差趋于 0（零温度极限）下的退化特例。

#### 3. 最近中心并列时，责任度是否必然趋于某一个中心的概率 1？

**结论：不必然。** 责任度通常趋于并列中心之间的先验权重比例，并不趋于 1。

**理由与反例说明：**

设样本 $x_i$ 到两个最近中心 $\mu_1$ 与 $\mu_2$ 的欧氏距离严格并列相等，且小于到其他中心的距离：

$$
\|x_i - \mu_1\|^2 = \|x_i - \mu_2\|^2 = d_{\min}^2 < \|x_i - \mu_k\|^2\quad (\forall k \ge 3)
$$

同理，分子分母同除以 $\exp\left(-\frac{d_{\min}^2}{2\sigma^2}\right)$。对于 $k \ge 3$ 的项，其距离差大于 0，指数项趋于 0，但前两个并列项的指数均为 $\exp(0) = 1$：

$$
\gamma_{i1} = \frac{\pi_1}{\pi_1 + \pi_2 + \sum_{k \ge 3} \pi_k \exp\left( -\frac{\Delta_k}{2\sigma^2} \right)} \xrightarrow{\sigma^2 \to 0} \frac{\pi_1}{\pi_1 + \pi_2}
$$

$$
\gamma_{i2} \xrightarrow{\sigma^2 \to 0} \frac{\pi_2}{\pi_1 + \pi_2}
$$

以及对 $k \ge 3$ 有 $\gamma_{ik} \to 0$。

**反例实测（在 `verify.py` 中已严格复核）：**
取中心 $\mu_1 = (0, 0), \mu_2 = (2, 0)$，样本点取在对称中点 $x_i = (1, 0)$。
此时样本到两中心的距离严格相等（$\|x_i-\mu_1\|^2 = 1 = \|x_i-\mu_2\|^2$）：
1. 若混合权重等权 $\pi_1 = \pi_2 = 0.5$，当 $\sigma^2 \to 0$ 时，后验责任度极限为：

   $$
   \gamma_{i1} \to \frac{0.5}{0.5 + 0.5} = 0.5,\qquad \gamma_{i2} \to \frac{0.5}{0.5 + 0.5} = 0.5
   $$

   责任度在两个并列中心之间各占一半（$0.5$ 与 $0.5$），不趋于 1。
2. 即使混合权重不等（例如 $\pi_1 = 0.3, \pi_2 = 0.7$），极限也是确定的先验比例：

   $$
   \gamma_{i1} \to 0.3,\qquad \gamma_{i2} \to 0.7
   $$

   二者均严格介于 0 和 1 之间。

因此，原说法**错误**。
