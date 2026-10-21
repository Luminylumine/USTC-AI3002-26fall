# HW1 

---

## 1 平方损失的偏差–方差分解

固定输入 $\boldsymbol{x}_0$，测试标签 $y = f_*(\boldsymbol{x}_0) + \epsilon$，$\mathbb{E}[\epsilon]=0$，$\mathbb{E}[\epsilon^2]=\sigma^2$，
$Z = f_{\hat{\boldsymbol{w}}_S}(\boldsymbol{x}_0) = \hat{\boldsymbol{w}}_S\cdot\phi(\boldsymbol{x}_0)$，$m = \mathbb{E}_S[Z]$，$\epsilon \perp S$。

### (a) 偏差–方差分解的补全

**第一步：把误差拆成噪声与模型两部分。**
$$
\mathbb{E}_{S,\epsilon}\big[(y-Z)^2\big]
= \mathbb{E}_{S,\epsilon}\big[(f_*(\boldsymbol{x}_0) + \epsilon - Z)^2\big]
= \mathbb{E}_{S,\epsilon}\big[\big(\epsilon + (f_*(\boldsymbol{x}_0) - Z)\big)^2\big]
$$

展开平方：

$$
= \underbrace{\mathbb{E}[\epsilon^2]}_{\text{噪声项}}
+ \underbrace{2\,\mathbb{E}\big[\epsilon\,(f_*(\boldsymbol{x}_0)-Z)\big]}_{\text{交叉项}}
+ \mathbb{E}\big[(f_*(\boldsymbol{x}_0)-Z)^2\big]
$$

**第二步：交叉项为零。** $f_*(\boldsymbol{x}_0)$ 与 $Z$ 都只依赖于训练集 $S$（不依赖 $\epsilon$），故 $(f_*(\boldsymbol{x}_0)-Z)$ 是 $S$ 的函数；由 $\epsilon \perp S$ 可分离期望，再用 $\mathbb{E}[\epsilon]=0$：
$$
\mathbb{E}\big[\epsilon(f_*(\boldsymbol{x}_0)-Z)\big]
= \mathbb{E}[\epsilon]\cdot\mathbb{E}\big[f_*(\boldsymbol{x}_0)-Z\big]
= 0\cdot(\cdots) = 0.
$$

于是 $\mathbb{E}[\epsilon^2]=\sigma^2$ 给出**不可约噪声** $\sigma^2$。

**第三步：把剩余项拆成偏差与方差。** 令 $\delta = Z - m$，则 $\mathbb{E}_S[\delta]=0$，$\mathbb{E}_S[\delta^2]=\operatorname{Var}_S(Z)$，且
$$
f_*(\boldsymbol{x}_0) - Z = (f_*(\boldsymbol{x}_0)-m) - \delta .
$$

展开（显然有 $(f_*(\boldsymbol{x}_0)-m)$ 是常数、$\mathbb{E}[\delta]=0$ 使交叉项消失）：

$$
\mathbb{E}\big[(f_*-Z)^2\big]
= (f_*(\boldsymbol{x}_0)-m)^2 - 2(f_*(\boldsymbol{x}_0)-m)\underbrace{\mathbb{E}[\delta]}_{=0} + \mathbb{E}[\delta^2]
= \big(m-f_*(\boldsymbol{x}_0)\big)^2 + \operatorname{Var}_S(Z).
$$

**合并：**

$$
{\ \mathbb{E}_{S,\epsilon}\big[(y-Z)^2\big] = \sigma^2 + \big(m-f_*(\boldsymbol{x}_0)\big)^2 + \operatorname{Var}_S(Z)\ }
$$

三项依次为：不可约噪声（irreducible noise）、平方偏差（bias²）、预测方差（variance）。

### (b) 三个说法的判断

**1. 错误。** 该条件只保证平方偏差项 $(m-f_*(\boldsymbol{x}_0))^2=0$，但期望测试误差仍为 $\sigma^2 + \operatorname{Var}_S(Z)$。只要标签有噪声（$\sigma^2>0$）或不同训练集给出的预测有波动（$\operatorname{Var}_S(Z)>0$），误差就严格大于零；只有两者同时为零时才为零。

**2. 正确。** 由分解，期望测试误差 $= \sigma^2 + \text{平方偏差} + \text{方差}$。$\sigma^2$ 不变时，总变化量 $= \Delta(\text{平方偏差}) + \Delta(\text{方差})$。题设“平方偏差的增加小于预测方差的减少”即 $\Delta(\text{平方偏差}) < -\Delta(\text{方差})$，故总变化量 $<0$，期望测试误差下降。

**3. 错误。** 训练误差只度量模型对训练集的拟合程度；训练误差极低往往意味着高方差（过拟合），而泛化误差由偏差²、方差与噪声共同决定，与训练误差之间不存在单调关系（例如无正则的高容量模型可做到训练误差为 0 而验证误差很大）。

---

## 2 平方损失下的多输出回归

### (a) 单输出损失与可分性

第 $j$ 个输出的均方误差：

$$
L_j(\boldsymbol{w}_j) = \frac{1}{N}\sum_{i=1}^{N}\big(\boldsymbol{w}_j\cdot\phi(\boldsymbol{x}_i) - y_{ij}\big)^2 ,\qquad j=1,\dots,n,
$$

于是总损失**恰好是各单输出损失之和**：

$$
L(\boldsymbol{W}) = \frac{1}{N}\sum_{i=1}^N\sum_{j=1}^n\big(\boldsymbol{w}_j\cdot\phi(\boldsymbol{x}_i)-y_{ij}\big)^2
= \sum_{j=1}^{n} L_j(\boldsymbol{w}_j).
$$

**只改变 $\boldsymbol{w}_k$、其余固定时，只有 $L_k$ 可能变化。** 理由：$L_j\ (j\neq k)$ 的表达式中只出现参数列 $\boldsymbol{w}_j$ 与数据 $(\boldsymbol{x}_i,y_{ij})$，完全不含 $\boldsymbol{w}_k$，因此

$$
\frac{\partial L_j}{\partial \boldsymbol{w}_k} = \boldsymbol{0}\quad (j\neq k),
$$

$L_j$ 的取值与 $\boldsymbol{w}_k$ 无关；而 $L_k$ 显含 $\boldsymbol{w}_k$，一般会随 $\boldsymbol{w}_k$ 改变（除非 $\boldsymbol{w}_k$ 恰在 $L_k$ 的驻点处移动方向与梯度正交）。因此 $L(\boldsymbol{W})$ 的变化量就等于 $L_k$ 的变化量。

### (b) 分别最优 ⇒ 联合最优（等价性证明）

设各单输出问题存在最优解，记 $\boldsymbol{w}_j^\star \in \arg\min_{\boldsymbol{w}_j} L_j(\boldsymbol{w}_j)$，$\boldsymbol{W}^\star=[\boldsymbol{w}_1^\star,\dots,\boldsymbol{w}_n^\star]$。

**①（组合解是联合最优）** 对任意可行的 $\boldsymbol{W}=[\boldsymbol{w}_1,\dots,\boldsymbol{w}_n]$，由每个 $\boldsymbol{w}_j^\star$ 的最优性 $L_j(\boldsymbol{w}_j)\ge L_j(\boldsymbol{w}_j^\star)$，得

$$
L(\boldsymbol{W}) = \sum_{j=1}^n L_j(\boldsymbol{w}_j) \;\ge\; \sum_{j=1}^n L_j(\boldsymbol{w}_j^\star) = L(\boldsymbol{W}^\star).
$$

故 $\boldsymbol{W}^\star$ 是 $L$ 的全局最小点。

**②（联合最优的每一列分别最优）** 设 $\hat{\boldsymbol{W}}=[\hat{\boldsymbol{w}}_1,\dots,\hat{\boldsymbol{w}}_n]$ 是 $L$ 的任一全局最小点。反设存在某 $k$ 使 $\hat{\boldsymbol{w}}_k$ 不是 $L_k$ 的最优解，即存在 $\boldsymbol{v}$ 使 $L_k(\boldsymbol{v}) < L_k(\hat{\boldsymbol{w}}_k)$。构造 $\tilde{\boldsymbol{W}}$：第 $k$ 列换成 $\boldsymbol{v}$，其余列不变，则

$$
L(\tilde{\boldsymbol{W}}) = \sum_{j\neq k}L_j(\hat{\boldsymbol{w}}_j) + L_k(\boldsymbol{v})
< \sum_{j=1}^n L_j(\hat{\boldsymbol{w}}_j) = L(\hat{\boldsymbol{W}}),
$$

与 $\hat{\boldsymbol{W}}$ 是全局最小点矛盾。故每列 $\hat{\boldsymbol{w}}_k \in \arg\min L_k$。

由 ①②，"分别最小化后组合" 与 "联合最小化" 得到的是同一集合的解，二者等价。（若最优解不唯一，任取各 $L_j$ 的最优解组合仍是联合最优解。）

**不需要假设不同输出在统计上独立。** 上述等价性只依赖目标函数的**代数可分性**（$L=\sum_j L_j$ 且第 $j$ 项只含第 $j$ 列参数），这是损失函数的结构性质，与 $(y_{i1},\dots,y_{in})$ 的联合分布无关。各输出之间可以任意相关——相关性只影响各 $\boldsymbol{w}_j^\star$ 本身的取值与统计性质，但因为损失中不存在跨输出的交叉项，参数之间不会互相耦合。"统计独立" 似然意义上的分解才会用到独立性假设（如朴素贝叶斯），本题不是那种情形。

### (c) 增加约束 $\boldsymbol{w}_1=\boldsymbol{w}_2$ 后

**不能**再直接组合两个单输出问题的最优解。

设两问题各自的无约束最优解唯一且不同：$\boldsymbol{w}_1^\star \neq \boldsymbol{w}_2^\star$。在约束 $\boldsymbol{w}_1=\boldsymbol{w}_2=\boldsymbol{w}$ 下，目标变成**同一个变量**上的和

$$
\min_{\boldsymbol{w}}\ \big[L_1(\boldsymbol{w}) + L_2(\boldsymbol{w})\big].
$$

- **不可能同时达到两个无约束最小值。** 由唯一性，$L_1(\boldsymbol{w}) \ge L_1(\boldsymbol{w}_1^\star)$ 等号仅当 $\boldsymbol{w}=\boldsymbol{w}_1^\star$；$L_2(\boldsymbol{w}) \ge L_2(\boldsymbol{w}_2^\star)$ 等号仅当 $\boldsymbol{w}=\boldsymbol{w}_2^\star$。因 $\boldsymbol{w}_1^\star\neq\boldsymbol{w}_2^\star$，不存在同时取到两者的 $\boldsymbol{w}$，故对一切可行 $\boldsymbol{w}$，

$$
L_1(\boldsymbol{w}) + L_2(\boldsymbol{w}) \;>\; L_1(\boldsymbol{w}_1^\star) + L_2(\boldsymbol{w}_2^\star),
$$

即总损失严格大于两个无约束最小值之和，两个单输出的最小值**无法同时达到**。

- **(b) 中失效的条件。** (b) 的证明建立在"各输出的参数向量可以**独立**选择"上，即可行集是直积 $\mathcal{W}=\mathcal{W}_1\times\cdots\times\mathcal{W}_n$、目标是按块可分的和。约束 $\boldsymbol{w}_1=\boldsymbol{w}_2$ 把两个原本独立的块耦合在一起，可行集变为对角集 $\{(\boldsymbol{w},\boldsymbol{w})\}$，**不再是直积**，目标也**不再按块可分**——(b) 中"替换某一列不影响其他项"的论证（①②两步）不再成立。

---

## 3 线性回归：特征表示、模型训练与正则化

常数特征排在首位，故

$$
\phi(\boldsymbol{x}) = (1,\ x_1,\ x_2)^{\mathsf{T}} \in \mathbb{R}^3 .
$$

### (a) 设计矩阵、目标向量与可行性

按样本顺序 $(1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1)$：

$$
\Phi =
\begin{pmatrix}
1 & 1 & 0\\
1 & 0 & 1\\
1 & 1 & 1\\
1 & -1 & 0\\
1 & 0 & -1\\
1 & -1 & -1
\end{pmatrix}\in\mathbb{R}^{6\times 3},
\qquad
\boldsymbol{y} =
\begin{pmatrix}
2\\0\\3\\-4\\-2\\-5
\end{pmatrix}\in\mathbb{R}^{6}.
$$

**存在**这样的权重。逐条列方程（$\boldsymbol{w}=(w_0,w_1,w_2)^{\mathsf{T}}$，$w_0$ 为截距）：

$$
\begin{aligned}
&\text{① } w_0+w_1=2 &&\text{② } w_0+w_2=0 &&\text{③ } w_0+w_1+w_2=3\\
&\text{④ } w_0-w_1=-4 &&\text{⑤ } w_0-w_2=-2 &&\text{⑥ } w_0-w_1-w_2=-5
\end{aligned}
$$

①+④：$2w_0 = -2 \Rightarrow w_0=-1$；代回①得 $w_1=3$；由②得 $w_2=1$。于是候选

$$
\boldsymbol{w}_A = (-1,\ 3,\ 1)^{\mathsf{T}} .
$$

**代回验证**：③ $-1+3+1=3$ ✓；⑤ $-1-1=-2$ ✓；⑥ $-1-3-1=-5$ ✓（①②④ 由构造满足）。六条全部成立，即 $\Phi\boldsymbol{w}_A=\boldsymbol{y}$，训练 MSE 为 0。（等价地说：$\Phi$ 列满秩为 3，且 $\boldsymbol{y}\in\operatorname{col}(\Phi)$，方程组相容。）

### (b) 目标函数的矩阵形式、二次型与梯度

$$
J_\lambda(\boldsymbol{w}) = \frac{1}{N}\|\Phi\boldsymbol{w}-\boldsymbol{y}\|_2^2 + \lambda\|P\boldsymbol{w}\|_2^2,
\qquad N=6,\ P=\operatorname{diag}(0,1,1),
$$

其中 $\|P\boldsymbol{w}\|^2 = w_1^2 + w_2^2$（分量从 0 计，即不惩罚截距）。

展开：

$$
J_\lambda(\boldsymbol{w})
= \frac{1}{6}\big(\boldsymbol{w}^{\mathsf{T}}\Phi^{\mathsf{T}}\Phi\,\boldsymbol{w} - 2\boldsymbol{w}^{\mathsf{T}}\Phi^{\mathsf{T}}\boldsymbol{y} + \boldsymbol{y}^{\mathsf{T}}\boldsymbol{y}\big) + \lambda\,\boldsymbol{w}^{\mathsf{T}}P\boldsymbol{w}
$$

$$
= \boldsymbol{w}^{\mathsf{T}}\underbrace{\Big(\tfrac{1}{6}\Phi^{\mathsf{T}}\Phi + \lambda P\Big)}_{\boldsymbol{A}=\boldsymbol{A}^{\mathsf{T}}}\boldsymbol{w}
- 2\underbrace{\Big(\tfrac{1}{6}\Phi^{\mathsf{T}}\boldsymbol{y}\Big)}_{\boldsymbol{b}}^{\!\mathsf{T}}\boldsymbol{w}
+ \underbrace{\tfrac{1}{6}\boldsymbol{y}^{\mathsf{T}}\boldsymbol{y}}_{c}.
$$

具体地，

$$
\Phi^{\mathsf{T}}\Phi = \begin{pmatrix}6&0&0\\0&4&2\\0&2&4\end{pmatrix},\qquad
\Phi^{\mathsf{T}}\boldsymbol{y} = \begin{pmatrix}-6\\14\\10\end{pmatrix},\qquad
\boldsymbol{y}^{\mathsf{T}}\boldsymbol{y} = 4+0+9+16+4+25 = 58 .
$$

**梯度**（$\nabla(\boldsymbol{w}^{\mathsf{T}}\boldsymbol{A}\boldsymbol{w}) = 2\boldsymbol{A}\boldsymbol{w}$ 对称 $\boldsymbol{A}$，$\nabla(\boldsymbol{w}^{\mathsf{T}}\boldsymbol{A}\boldsymbol{w})$ 与 $\nabla(\boldsymbol{g}^{\mathsf{T}}\boldsymbol{w})=\boldsymbol{g}$）：

$$
\nabla_{\boldsymbol{w}} J_\lambda(\boldsymbol{w})
= \frac{2}{6}\Phi^{\mathsf{T}}(\Phi\boldsymbol{w}-\boldsymbol{y}) + 2\lambda P\boldsymbol{w}
= \frac{1}{3}\Phi^{\mathsf{T}}(\Phi\boldsymbol{w}-\boldsymbol{y}) + 2\lambda P\boldsymbol{w}.
$$

即 $\nabla J_\lambda = \frac{1}{3}\Phi^{\mathsf{T}}\Phi\,\boldsymbol{w} - \frac{1}{3}\Phi^{\mathsf{T}}\boldsymbol{y} + 2\lambda P\boldsymbol{w}$，分量形式：

$$
\frac{\partial J_\lambda}{\partial w_0} = \frac{1}{3}\big[(\Phi^{\mathsf{T}}\Phi\boldsymbol{w})_0 - (\Phi^{\mathsf{T}}\boldsymbol{y})_0\big],\qquad
\frac{\partial J_\lambda}{\partial w_j} = \frac{1}{3}\big[(\Phi^{\mathsf{T}}\Phi\boldsymbol{w})_j - (\Phi^{\mathsf{T}}\boldsymbol{y})_j\big] + 2\lambda w_j,\ j=1,2 .
$$

### (c) 一次全批量梯度下降（$\lambda=1/3$，$\boldsymbol{w}^{(0)}=(0,1,-1)^{\mathsf{T}}$，$\eta=1/4$）

**残差向量** $\boldsymbol{r} = \Phi\boldsymbol{w}^{(0)}-\boldsymbol{y}$：

| 样本 | $\phi(\boldsymbol{x})$ | $\phi\cdot\boldsymbol{w}^{(0)}$ | $y$ | $r_i$ |
| --- | --- | --- | --- | --- |
| 1 | (1,1,0) | $0+1+0=1$ | 2 | $-1$ |
| 2 | (1,0,1) | $0+0-1=-1$ | 0 | $-1$ |
| 3 | (1,1,1) | $0+1-1=0$ | 3 | $-3$ |
| 4 | (1,-1,0) | $0-1+0=-1$ | $-4$ | $3$ |
| 5 | (1,0,-1) | $0+0+1=1$ | $-2$ | $3$ |
| 6 | (1,-1,-1) | $0-1+1=0$ | $-5$ | $5$ |

$$
\boldsymbol{r} = (-1,\,-1,\,-3,\,3,\,3,\,5)^{\mathsf{T}} .
$$

**梯度：** 先算数据项

$$
\Phi^{\mathsf{T}}\boldsymbol{r} =
\begin{pmatrix}
\sum_i r_i\\ \sum_i x_{i1}r_i\\ \sum_i x_{i2}r_i
\end{pmatrix}
= \begin{pmatrix}
-1-1-3+3+3+5\\
(-1)(1)+0+( -3)(1)+(3)(-1)+0+(5)(-1)\\
0+(-1)(1)+(-3)(1)+0+(3)(-1)+(5)(-1)
\end{pmatrix}
= \begin{pmatrix}6\\-12\\-12\end{pmatrix},
$$

（第二行：$-1+0-3-3+0-5=-12$；第三行：$0-1-3-0-3-5=-12$。）

$$
\nabla L(\boldsymbol{w}^{(0)}) = \frac{1}{3}\Phi^{\mathsf{T}}\boldsymbol{r} = (2,\,-4,\,-4)^{\mathsf{T}} .
$$

正则项梯度（$P\boldsymbol{w}^{(0)} = (0,1,-1)$）：

$$
\nabla\big(\lambda\|P\boldsymbol{w}\|^2\big) = 2\lambda P\boldsymbol{w}^{(0)} = \tfrac{2}{3}(0,1,-1) = (0,\ \tfrac{2}{3},\ -\tfrac{2}{3}) .
$$

$$
\boxed{\ \nabla J_{1/3}(\boldsymbol{w}^{(0)}) = (2,\ -4+\tfrac{2}{3},\ -4-\tfrac{2}{3}) = \big(2,\ -\tfrac{10}{3},\ -\tfrac{14}{3}\big)^{\mathsf{T}}\ }
$$

**同步更新**（所有分量用同一个旧值 $\boldsymbol{w}^{(0)}$ 计算后一次性更新）：

$$
\boldsymbol{w}^{(1)} = \boldsymbol{w}^{(0)} - \eta\nabla J
= \begin{pmatrix}0\\1\\-1\end{pmatrix} - \frac14\begin{pmatrix}2\\-10/3\\-14/3\end{pmatrix}
= \begin{pmatrix}-\tfrac12\\ 1+\tfrac{5}{6}\\ -1+\tfrac{7}{6}\end{pmatrix}
= \begin{pmatrix}-\tfrac12\\ \tfrac{11}{6}\\ \tfrac16\end{pmatrix}.
$$

**目标函数值比较：**

- 更新前：$L(\boldsymbol{w}^{(0)}) = \frac16\|\boldsymbol{r}\|^2 = \frac{1+1+9+9+9+25}{6} = \frac{54}{6} = 9$；
  $\lambda\|P\boldsymbol{w}^{(0)}\|^2 = \frac13(1^2+(-1)^2) = \frac23$；
  $$J_{1/3}(\boldsymbol{w}^{(0)}) = 9 + \tfrac23 = \tfrac{29}{3} \approx 9.6667 .$$
- 更新后：新残差 $= (-\tfrac23,\ -\tfrac13,\ -\tfrac32,\ \tfrac53,\ \tfrac43,\ \tfrac52)^{\mathsf{T}}$，平方和

$$
\tfrac49+\tfrac19+\tfrac94+\tfrac{25}{9}+\tfrac{16}{9}+\tfrac{25}{4}
= \underbrace{\tfrac{46}{9}}_{\text{九分之}} + \underbrace{\tfrac{17}{2}}_{\text{四分之}}
= \tfrac{245}{18},
$$

$$
L(\boldsymbol{w}^{(1)}) = \frac16\cdot\frac{245}{18} = \frac{245}{108},\qquad
\lambda\|P\boldsymbol{w}^{(1)}\|^2 = \frac13\Big(\big(\tfrac{11}{6}\big)^2+\big(\tfrac16\big)^2\Big) = \frac13\cdot\frac{122}{36} = \frac{122}{108},
$$

$$
J_{1/3}(\boldsymbol{w}^{(1)}) = \frac{245+122}{108} = \frac{367}{108} \approx 3.3981 .
$$

因为 $\frac{367}{108}\approx 3.398 < \frac{29}{3}\approx 9.667$，**这次更新使目标函数显著下降**。

### (d) 两个候选模型的比较

**(i) 训练 MSE**

- $\boldsymbol{w}_A=(-1,3,1)$：由 (a) 精确拟合 6 条样本，$\text{MSE}_{\text{train}}(\boldsymbol{w}_A) = 0$。
- $\boldsymbol{w}_B=(-1,2,1)$：逐条残差

| 样本 | 预测 | $y$ | 残差 |
| --- | --- | --- | --- |
| 1 | $-1+2=1$ | 2 | $-1$ |
| 2 | $-1+1=0$ | 0 | 0 |
| 3 | $-1+2+1=2$ | 3 | $-1$ |
| 4 | $-1-2=-3$ | $-4$ | $1$ |
| 5 | $-1-1=-2$ | $-2$ | 0 |
| 6 | $-1-2-1=-4$ | $-5$ | $1$ |

平方和 $=1+0+1+1+0+1=4$，$\text{MSE}_{\text{train}}(\boldsymbol{w}_B) = \tfrac{4}{6} = \tfrac23 \approx 0.6667$。

**(ii) 正则项范数**

$$
\|P\boldsymbol{w}_A\|^2 = 3^2+1^2 = 10,\qquad
\|P\boldsymbol{w}_B\|^2 = 2^2+1^2 = 5 .
$$

（顺带：$J_{1/3}(\boldsymbol{w}_A)=0+\frac{10}{3}=\frac{10}{3}$，$J_{1/3}(\boldsymbol{w}_B)=\frac23+\frac53=\frac73$，在 $\lambda=1/3$ 的目标下 $\boldsymbol{w}_B$ 更优。）

**(iii) 验证集 MSE**（$V=\{((0,0),1),((1,0),2),((0,1),1)\}$）

| 验证点 | $\phi$ | $\boldsymbol{w}_A$ 预测 | $y$ | 误差² | $\boldsymbol{w}_B$ 预测 | $y$ | 误差² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| (0,0) | (1,0,0) | $-1$ | 1 | $(-2)^2=4$ | $-1$ | 1 | $4$ |
| (1,0) | (1,1,0) | $-1+3=2$ | 2 | 0 | $-1+2=1$ | 2 | 1 |
| (0,1) | (1,0,1) | $-1+1=0$ | 1 | 1 | $-1+1=0$ | 1 | 1 |

$$
\text{MSE}_{\text{val}}(\boldsymbol{w}_A) = \frac{4+0+1}{3} = \frac53 \approx 1.667,\qquad
\text{MSE}_{\text{val}}(\boldsymbol{w}_B) = \frac{4+1+1}{3} = 2 .
$$

**选择：$\boldsymbol{w}_A$**（验证 MSE $5/3 < 2$）。虽然它训练 MSE 为 0（看似"过拟合"），但在本验证集上预测更准；$\boldsymbol{w}_B$ 虽然正则惩罚更小，训练误差和验证误差却都更大。

**验证 MSE 中不应加入正则惩罚, 因为验证 MSE 度量的是模型在**未见数据上的预测误差**，即纯数据拟合项 $\frac{1}{|V|}\sum_{v\in V}(f_{\boldsymbol{w}}(x_v)-y_v)^2$，这才是我们真正关心的泛化风险的估计。而 $\lambda\|P\boldsymbol{w}\|^2$ 是**训练阶段**为控制模型复杂度而引入的惩罚/代理项，它不参与预测（推理时只是对权重打个"账"），与预测误差量纲、含义都不同：若把它加进验证指标，等于把同一个复杂度惩罚计两次（其影响已通过 $\boldsymbol{w}$ 对预测的作用体现），而且两个模型的 $\lambda$ 可能不同（$\boldsymbol{w}_A$ 是 $\lambda=0$ 的候选），加进去会破坏比较的公平性。模型选择应基于纯预测误差（估计泛化损失），正则只在训练时起作用。

### (e) $\lambda$ 增大时训练 MSE 与正则范数的单调性

记 $J_i(\boldsymbol{w}) = L(\boldsymbol{w}) + \lambda_i\|P\boldsymbol{w}\|^2$，$\boldsymbol{w}^{[i]} = \arg\min J_i$，$i=1,2$，$0\le\lambda_1<\lambda_2$。

由各自的**最优性**（$\boldsymbol{w}^{[1]}$ 在 $J_1$ 下最优、$\boldsymbol{w}^{[2]}$ 在 $J_2$ 下最优），把两个解放到**同一个目标函数**下比较：

$$
\text{(A)}\quad L(\boldsymbol{w}^{[1]}) + \lambda_1\|P\boldsymbol{w}^{[1]}\|^2 \le L(\boldsymbol{w}^{[2]}) + \lambda_1\|P\boldsymbol{w}^{[2]}\|^2,
$$

$$
\text{(B)}\quad L(\boldsymbol{w}^{[2]}) + \lambda_2\|P\boldsymbol{w}^{[2]}\|^2 \le L(\boldsymbol{w}^{[1]}) + \lambda_2\|P\boldsymbol{w}^{[1]}\|^2.
$$

**(A)+(B)**，两边的 $L(\boldsymbol{w}^{[1]})$、$L(\boldsymbol{w}^{[2]})$ 抵消：

$$
\lambda_1\|P\boldsymbol{w}^{[1]}\|^2 + \lambda_2\|P\boldsymbol{w}^{[2]}\|^2
\le \lambda_1\|P\boldsymbol{w}^{[2]}\|^2 + \lambda_2\|P\boldsymbol{w}^{[1]}\|^2,
$$

移项：

$$
(\lambda_2-\lambda_1)\,\|P\boldsymbol{w}^{[2]}\|^2 \le (\lambda_2-\lambda_1)\,\|P\boldsymbol{w}^{[1]}\|^2 .
$$

因 $\lambda_2-\lambda_1>0$，两边同除即得

$$
\boxed{\ \|P\boldsymbol{w}^{[2]}\|^2 \le \|P\boldsymbol{w}^{[1]}\|^2\ }.
$$

再由 **(A)**：

$$
L(\boldsymbol{w}^{[1]}) - L(\boldsymbol{w}^{[2]})
\le \lambda_1\big(\|P\boldsymbol{w}^{[2]}\|^2 - \|P\boldsymbol{w}^{[1]}\|^2\big) \le 0,
$$

最后一步用到 $\lambda_1\ge 0$ 与刚证的范数不等式。故

$$
\boxed{\ L(\boldsymbol{w}^{[2]}) \ge L(\boldsymbol{w}^{[1]})\ }.
$$

即：**增大 $\lambda$ 使正则项范数不增、训练 MSE 不降**（训练拟合只会变差或持平）。

**增大 $\lambda$ 不一定能降低验证 MSE。** 上述结论只涉及训练集上的两个量。验证 MSE 由"拟合误差 + 复杂度/估计误差"共同决定：适度增大 $\lambda$ 可减小方差、改善泛化；但 $\lambda$ 过大时模型欠拟合、偏差增大，训练 MSE 已在上升，验证 MSE 也会随之上升（典型的 U 形曲线）。而且 (e) 的结论只给出训练 MSE 的**下界方向**，对验证集没有任何保证——两者分布不同、样本不同，故不存在"$\lambda$ 越大验证 MSE 越小"的单调性。

### (f) L2 正则与权重衰减的关系

对 $J_\lambda(\boldsymbol{w}) = L(\boldsymbol{w}) + \lambda\|P\boldsymbol{w}\|_2^2$ 作普通梯度下降：

$$
\boldsymbol{w}^{(t+1)} = \boldsymbol{w}^{(t)} - \eta\big(\nabla L(\boldsymbol{w}^{(t)}) + 2\lambda P\boldsymbol{w}^{(t)}\big)
= \big(I - 2\eta\lambda P\big)\boldsymbol{w}^{(t)} - \eta\nabla L(\boldsymbol{w}^{(t)}).
$$

而

$$
I - 2\eta\lambda P = I - 2\eta\lambda\operatorname{diag}(0,1,1) = \operatorname{diag}\big(1,\ 1-2\eta\lambda,\ 1-2\eta\lambda\big).
$$

与题给 $D_c=\operatorname{diag}(1,c,c)$ 对比，得

$$
\boxed{\ c = 1 - 2\eta\lambda = 1 - 2\cdot\tfrac14\cdot\tfrac13 = 1 - \tfrac16 = \tfrac56\ }.
$$

**为什么 $D_c$ 的首个对角元是 1 而不是 $c$：** 因为 $P=\operatorname{diag}(0,1,1)$，截距分量**不参与正则化**，其惩罚梯度的对应分量为 $2\lambda\cdot 0 = 0$，即

$$
\big[\nabla(\lambda\|P\boldsymbol{w}\|^2)\big]_0 = 0 .
$$

要让权重衰减的更新严格等价于对 $J_\lambda$ 作梯度下降，截距就必须保持原值不被乘以衰减因子（系数 $1-2\eta\lambda\cdot 0 = 1$）。若首元也取 $c=5/6$，截距会被额外收缩，等价于额外对截距加了一个（错误的）正则项，更新就不再是对 $J_\lambda$ 的梯度下降。

---

## 4 逻辑回归：概率建模与分类置信度

数据（每组 8 条观测，似然按观测逐条相乘）：

| $x$ | A：$y=0$ | A：$y=1$ | B：$y=0$ | B：$y=1$ |
| --- | --- | --- | --- | --- |
| $-1$ | 3 | 1 | 4 | 0 |
| $1$ | 1 | 3 | 0 | 4 |

单条观测的似然贡献：$\sigma(tx_i)^{y_i}\big(1-\sigma(tx_i)\big)^{1-y_i}$。

### (a) 似然与平均负对数似然

**数据集 A：** 逐格写出（$x=-1$ 处 3 个 $y=0$、1 个 $y=1$；$x=1$ 处 1 个 $y=0$、3 个 $y=1$）：

$$
\mathcal{L}_A(t) = \big[1-\sigma(-t)\big]^3\big[\sigma(-t)\big]^1\cdot\big[1-\sigma(t)\big]^1\big[\sigma(t)\big]^3 .
$$

利用 $1-\sigma(z)=\sigma(-z)$（故 $1-\sigma(-t)=\sigma(t)$，$1-\sigma(t)=\sigma(-t)$），四个因子归并：

$$
\mathcal{L}_A(t) = \sigma(t)^3\cdot\sigma(-t)\cdot\sigma(-t)\cdot\sigma(t)^3 = \big[\sigma(t)\big]^6\big[\sigma(-t)\big]^2 .
$$

写成指数形式 $\sigma(t)=\dfrac{e^t}{1+e^t}$、$\sigma(-t)=\dfrac{1}{1+e^t}$：

$$
\mathcal{L}_A(t) = \frac{e^{6t}}{(1+e^t)^6}\cdot\frac{1}{(1+e^t)^2} = \frac{e^{6t}}{(1+e^t)^{8}},
$$

$$
J_A(t) = -\frac{1}{8}\ln\mathcal{L}_A(t) = -\frac{1}{8}\big[6t - 8\ln(1+e^t)\big],
$$

$$
\boxed{\ J_A(t) = \ln\big(1+e^{t}\big) - \tfrac34\,t\ }
$$

（等价写法 $J_A(t)=\ln\big(e^{-3t/4}+e^{t/4}\big)$；$t=0$ 时 $J_A(0)=\ln 2$。）

**数据集 B：** $x=-1$ 处 4 条全是 $y=0$，贡献 $(1-\sigma(-t))^4 = \sigma(t)^4$；$x=1$ 处 4 条全是 $y=1$，贡献 $\sigma(t)^4$。

$$
\mathcal{L}_B(t) = \sigma(t)^4\cdot\sigma(t)^4 = \sigma(t)^8,
\qquad
J_B(t) = -\frac18\ln\sigma(t)^8 = -\ln\sigma(t),
$$

$$
\boxed{\ J_B(t) = -\ln\sigma(t) = \ln\big(1+e^{-t}\big).\ }
$$

**为什么最大化似然等价于最小化该损失：** $\ln(\cdot)$ 是严格单调增函数，故 $\arg\max_t \mathcal{L}(t) = \arg\max_t \ln\mathcal{L}(t) = \arg\min_t\big(-\ln\mathcal{L}(t)\big)$；乘以正常数 $\frac1N$ 只是取平均（尺度变换），不改变最优点位置。因此最大化似然与最小化平均负对数似然给出相同的最优参数。

### (b) 最优解与可达性

**数据集 A：** 由闭式 $J_A(t) = \ln(1+e^t) - \tfrac34 t$，

$$
J_A'(t) = \sigma(t) - \frac34 = 0
\iff \frac{e^{t}}{1+e^{t}} = \frac34
\iff 4e^t = 3+3e^t
\iff e^t = 3
\iff t = \ln 3,
$$

$$
J_A''(t) = \sigma(t)\big(1-\sigma(t)\big) > 0 \quad \forall t\in\mathbb{R}.
$$

$J_A$ 处处严格凸（二阶导恒正），故驻点是**唯一**的全局最优解：

$$
\boxed{\ \boldsymbol{w}^\star = (\ln 3) \approx (1.0986),\qquad \sigma(t^\star)=\tfrac34\ }
$$

（直观：模型在 $x=-1$、$x=1$ 处输出 $\sigma(-t)$ 与 $\sigma(t)$，二者天然互补；A 中经验频率也恰好互补——$x=-1$ 处 $y=1$ 的频率为 $1/4$、$x=1$ 处为 $3/4$，于是两格给出同一个方程 $\sigma(t)=3/4$，解存在、有限且唯一，似然最大值 $<1$。）

**数据集 B：**

$$
J_B'(t) = -\frac{e^{-t}}{1+e^{-t}} = -\sigma(-t) < 0\quad\forall t,
$$

$J_B$ 严格单调递减，且

$$
\inf_{t\in\mathbb{R}} J_B(t) = \lim_{t\to+\infty}\ln(1+e^{-t}) = \ln 1 = 0 .
$$

**不存在有限参数达到该下确界**：对任意有限 $t$，$1+e^{-t}>1 \Rightarrow J_B(t)>0$；下确界只在 $t\to+\infty$ 的极限处取得。

**原因（结合标签分布）：** 数据集 A 在**每个 $x$ 上都同时出现两种标签**（$x=-1$：3 个 0、1 个 1；$x=1$：1 个 0、3 个 1），即给定特征后标签并不确定、存在"噪声"；似然被这些不一致的观测压在严格小于 1 的水平上，且模型在两点的预测 $\sigma(\pm t)$ 互补、经验频率 $1/4$ 与 $3/4$ 也互补，两格相容，故最优解**有限且唯一**（$t^\star=\ln3$，此时似然 $<1$）。数据集 B 则是**完全线性可分、无标签噪声**（$x=-1\Rightarrow y=0$，$x=1\Rightarrow y=1$，全部 8 条一致）：经验频率要求 $\sigma(-t)=0$ 且 $\sigma(t)=1$，这只能在 $t\to+\infty$ 时逼近——取 $t\to+\infty$ 时每个样本被赋予真实标签的概率 $\to 1$、似然 $\to 1$（损失 $\to 0$），但任何有限 $t$ 的似然都严格小于 1，故不存在有限的最优解。这就是"可分数据 + 无正则 ⇒ MLE 不存在（权重发散）"的经典现象。

### (c) 放大权重 $t \to ct$（$t>0$，$c>1$）

**决策边界不变。** 分类规则为 $\sigma(ct\cdot x)\ge \tfrac12 \iff ct\cdot x \ge 0$。因 $ct>0$，等价于 $x\ge 0$，即决策边界仍为 $x=0$；放大前的规则 $tx\ge0 \iff x\ge 0$ 完全相同。故边界与 $c$ 无关（只要 $c>0$）。

**分类错误率（数据集 B）：** 放大前 $t>0$ 已对全部 8 条分对：$x=-1\Rightarrow \sigma(-t)<\frac12\Rightarrow \hat y=0$ ✓，$x=1\Rightarrow\sigma(t)>\frac12\Rightarrow\hat y=1$ ✓，错误率 $0$；放大后边界未变，预测类别完全相同，**错误率仍为 0，不变**。

**真实标签的预测概率：** 对 B 中每条样本，真实标签概率均为 $\sigma(t)$（$x=1,y=1$：$\sigma(t)$；$x=-1,y=0$：$1-\sigma(-t)=\sigma(t)$），故平均真标签概率 $=\sigma(t)$。放大后变为 $\sigma(ct)$；由 $c>1,\ t>0$ 与 $\sigma$ 严格递增，$\sigma(ct)>\sigma(t)$，**概率上升、趋近 1**。

**平均负对数似然：**

$$
J_B(ct) = \ln\big(1+e^{-ct}\big) < \ln\big(1+e^{-t}\big) = J_B(t)
$$

（因 $e^{-ct}<e^{-t}$），故**平均负对数似然下降（变好）**，且随 $c\to\infty$ 单调趋于 0。

**三者变化不同的原因：** 0-1 错误率只依赖**决策边界的符号**（margin 的正负），对正的尺度变换完全不变；真实标签概率是 margin 的单调增函数，放大 margin 即提升置信度；而负对数似然进一步**惩罚"不够自信"**，它随 margin 增大而持续下降、没有下界点（下确界 0 不可达），因此它会驱使权重无限增大——这正是 B 上 MLE 不存在的另一面。三者分别只看符号、看置信度水平、看置信度的对数损失，敏感度依次增强。

### (d) 两个说法的判断与正则的作用

**1. 错误。** 训练分类错误率为零只说明 0-1 损失已达最小（符号全对），但 $J_B(t)=\ln(1+e^{-t})$ 仍随 $t$ 增大而**严格下降**（下确界 0 不可达），因此负对数似然还能通过继续增大 $t$ 来降低。0-1 错误率最小与 NLL 最小是两回事：前者只要 margin 过零点，后者要求 margin 尽可能大。

**2. 错误。** 训练样本真标签概率 $\to1$ 只反映**训练集**上的置信度（此时 $t$ 极大、模型极度自信）；同分布的测试数据含新的、尤其靠近边界的样本，大权重会让模型对少数错误预测也给出接近 1 的高置信度，单条错误的巨大 NLL 惩罚（$-\ln(\text{很小的概率})$）会使测试平均 NLL **上升**。训练置信度趋近 1 保证的是训练 NLL 下降，不保证测试 NLL 下降（过拟合 / 置信度未校准）。

**为什么 $\lambda\| \boldsymbol{w}\|_2^2 = \lambda t^2$（$\lambda>0$）能抑制权重无限增大：** 加入正则后目标为 $J_B(t)+\lambda t^2$。当 $|t|\to\infty$ 时，$J_B(t)\to 0$（有界）而 $\lambda t^2\to+\infty$，故目标在两端都趋于 $+\infty$；目标连续，因此必在**某个有限 $t$** 处取到最小值，权重被限制在有限范围内，不再无限增大。（正则项对"靠极大权重换取训练置信度"的行为按 $t^2$ 收费，使无限增大在代价上得不偿失。）

---


