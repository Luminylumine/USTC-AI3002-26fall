"""Lab2 第二题：补全五个算法函数和三个评价函数。

所有函数都使用 PyTorch Tensor，运行设备由全局 ``device`` 控制；
``device`` 由框架设置为 ``torch.device("cpu")`` 或 ``torch.device("cuda:0")``。
不要在函数中写 CPU/GPU 两套分支，也不要把张量转为 NumPy 做计算。
不能调用现成聚类求解器、初始化器或评价函数；scikit-learn 仅用于独立核验。
浮点类型由配置 dtype 或 --dtype 选择，默认 float32，也支持 float64。
新浮点张量应继承 X.dtype 和 X.device；算法返回的 labels 使用 torch.int32 或 torch.int64，
评价函数接收的 labels 为 torch.int64。平均对数似然与评价指标可返回有限 Python float
或同设备零维 Tensor。rng 为框架已设置种子的同设备 torch.Generator，随机抽样时传入
generator=rng。不要修改输入。框架负责转换、迭代、重启、评价抽样、汇总和保存文件。
评价函数只接收特征与预测分区，不接收真实标签。
"""
import hashlib
import torch
from utils.runtime import device

# Compatibility polyfill for Python < 3.11 where hashlib.file_digest is not available
if not hasattr(hashlib, "file_digest"):
    def _file_digest(fileobj, digest, /, *, _bufsize=262144):
        h = hashlib.new(digest) if isinstance(digest, str) else digest()
        while True:
            chunk = fileobj.read(_bufsize)
            if not chunk:
                break
            h.update(chunk)
        return h

    hashlib.file_digest = _file_digest



# KMeans：初始化、分配与中心更新。
def init_centers_kmeanspp(X, n_clusters, rng):
    """返回 (K,D) 初始中心。

    X:(N,D)，rng:torch.Generator。第一个中心均匀抽取；其余
    按到最近已有中心的平方距离加权抽取。所有距离为0时，从尚未选过的
    样本行中均匀选取。不要修改 X；不要构造 (N,N) 距离矩阵；抽样权重保持为 Tensor。
    可使用 torch.randint / torch.multinomial，传入 generator=rng；新张量放在 device 上。
    """
    N, D = X.shape
    centers = X.new_empty((n_clusters, D))
    selected = []

    # 第一个中心均匀抽取
    c0 = int(torch.randint(0, N, (1,), generator=rng, device=X.device).item())
    selected.append(c0)
    centers[0] = X[c0]

    # 初始化到已有中心的最小平方距离
    diff0 = X - centers[0]
    min_sq_dist = (diff0 * diff0).sum(dim=1)

    for k in range(1, n_clusters):
        min_sq_dist = torch.clamp(min_sq_dist, min=0.0)
        total_dist = min_sq_dist.sum()
        if total_dist <= 0:
            # 距离全为 0 时从尚未选过的样本行中均匀抽取
            selected_set = set(selected)
            unselected = [i for i in range(N) if i not in selected_set]
            if len(unselected) == 0:
                idx = int(torch.randint(0, N, (1,), generator=rng, device=X.device).item())
            else:
                pick = int(torch.randint(0, len(unselected), (1,), generator=rng, device=X.device).item())
                idx = unselected[pick]
        else:
            idx = int(torch.multinomial(min_sq_dist, 1, generator=rng).item())

        selected.append(idx)
        centers[k] = X[idx]
        diff_k = X - centers[k]
        new_sq_dist = (diff_k * diff_k).sum(dim=1)
        min_sq_dist = torch.minimum(min_sq_dist, new_sq_dist)

    return centers


def assign_clusters(X, centers):
    """返回 labels:(N,) 整数、min_sq_dist:(N,)。

    按最近中心的平方欧氏距离分配，并列时选择较小编号。
    允许构造 (N,K) 距离矩阵，避免 (N,K,D) 大数组；将浮点消去造成的
    负平方距离截到0。计算式：||x||²+||mu||²-2*x@mu。
    """
    x_sq = (X * X).sum(dim=1, keepdim=True)
    mu_sq = (centers * centers).sum(dim=1, keepdim=True).t()
    dot = X @ centers.t()
    sq_dist = x_sq + mu_sq - 2.0 * dot
    sq_dist = torch.clamp(sq_dist, min=0.0)
    min_sq_dist, labels = torch.min(sq_dist, dim=1)
    return labels, min_sq_dist


def update_centers(X, labels, old_centers):
    """返回更新后的 (K,D) 中心，不修改输入。

    非空簇取均值；空簇保留旧中心。空簇策略不保证声明K等于实际K，
    最终实际非空簇数由框架单独检查。
    """
    K, D = old_centers.shape
    new_centers = old_centers.clone()
    counts = torch.bincount(labels.to(torch.int64), minlength=K)
    mask = counts > 0
    if mask.any():
        sum_x = torch.zeros((K, D), dtype=X.dtype, device=X.device)
        sum_x.index_add_(0, labels.to(torch.int64), X)
        new_centers[mask] = sum_x[mask] / counts[mask].unsqueeze(1).to(dtype=X.dtype)
    return new_centers


# GMM：对角协方差下的 E 步与 M 步。
def gmm_estep(X, weights, means, variances):
    """对角协方差 GMM 的 E 步。

    weights:(K,), means/variances:(K,D)，权重与方差为正。
    返回 resp:(N,K) 及每样本平均观测对数似然 mean_log_likelihood。
    责任度每行和为1；在对数空间计算高斯密度与混合权重，使用
    log-sum-exp 归一化，不能先算高维密度再取log。
    """
    N, D = X.shape
    K = weights.shape[0]
    log_prob = X.new_empty((N, K))
    const = -0.5 * D * torch.log(X.new_tensor(2.0 * torch.pi))
    log_weights = torch.log(weights)
    log_vars_sum = torch.log(variances).sum(dim=1)

    for k in range(K):
        diff = X - means[k]
        sq_dist_weighted = (diff * diff / variances[k]).sum(dim=1)
        log_prob[:, k] = log_weights[k] + const - 0.5 * log_vars_sum[k] - 0.5 * sq_dist_weighted

    mean_log_likelihood = torch.logsumexp(log_prob, dim=1)
    resp = torch.exp(log_prob - mean_log_likelihood.unsqueeze(1))
    return resp, mean_log_likelihood.mean()


def gmm_mstep(X, resp, reg_covar):
    """返回 weights:(K,), means:(K,D), variances:(K,D)。

    N_k=sum_i resp[i,k]；更新权重、加权均值与对角加权方差。
    方差先消除舍入误差导致的负值，再加 reg_covar（正数）。
    任一 N_k <= 1e-12 时 raise ValueError，框架会记录本次初始化失败。
    不要把失败重启算作收敛；不要修改输入。
    """
    N, D = X.shape
    K = resp.shape[1]
    N_k = resp.sum(dim=0)
    if bool((N_k <= 1e-12).any()):
        raise ValueError("Degenerate GMM component: effective count N_k <= 1e-12")

    weights = N_k / N
    means = (resp.t() @ X) / N_k.unsqueeze(1)

    variances = X.new_empty((K, D))
    for k in range(K):
        diff = X - means[k]
        weighted_sq_diff = (resp[:, k:k + 1] * (diff * diff)).sum(dim=0)
        var_k = weighted_sq_diff / N_k[k]
        variances[k] = torch.clamp(var_k, min=0.0) + reg_covar

    return weights, means, variances


# 评价：框架提供原始特征与预测分区，以下函数负责计算指标。
def silhouette_score(X, labels):
    """返回平均 Silhouette，使用欧氏距离（不是平方距离）。

    框架仅在 2 <= 实际簇数 < N 时调用并接受本指标；全为单点簇时指标未定义。
    a_i：同簇其他样本的平均距离；b_i：到其他每个簇的平均距离的最小值。
    s_i=(b_i-a_i)/max(a_i,b_i)，最后对样本平均。
    单样本簇的 s_i 为0；a_i=b_i=0 时也为0。
    X 已经是固定评价子集，不在本函数中再抽样。
    可以构造 (N,N) 距离矩阵，但不要构造 (N,N,D) 中间数组。
    """
    unique_clusters, inverse_indices = torch.unique(labels, return_inverse=True)
    C = len(unique_clusters)
    N = X.shape[0]

    x_sq = (X * X).sum(dim=1, keepdim=True)
    sq_dist = torch.clamp(x_sq + x_sq.t() - 2.0 * (X @ X.t()), min=0.0)
    dist = torch.sqrt(sq_dist)
    dist.fill_diagonal_(0.0)

    M = torch.zeros((N, C), dtype=X.dtype, device=X.device)
    M.scatter_(1, inverse_indices.unsqueeze(1), 1.0)
    counts = M.sum(dim=0)

    S = dist @ M
    mean_dist = S / counts.unsqueeze(0)

    a = S.gather(1, inverse_indices.unsqueeze(1)).squeeze(1)
    n_c = counts[inverse_indices]
    a = torch.where(n_c > 1, a / (n_c - 1), torch.zeros_like(a))

    mean_dist_other = mean_dist.clone()
    mean_dist_other.scatter_(1, inverse_indices.unsqueeze(1), float('inf'))
    b, _ = torch.min(mean_dist_other, dim=1)

    denom = torch.maximum(a, b)
    s = torch.where((n_c == 1) | (denom == 0.0), torch.zeros_like(a), (b - a) / denom)
    return s.mean()


def calinski_harabasz_score(X, labels):
    """返回 CH，框架传入全量原始特征。

    框架仅在 2 <= 实际簇数 < N 时调用并接受本指标。
    B=sum_q n_q*||mu_q-mu||²；W=sum_q sum_{i in q} ||x_i-mu_q||²。
    CH=(B/(K-1))/(W/(N-K))。K 为实际非空簇数。
    为兼容原有指标口径，W 恰为0时返回1.0。
    """
    unique_clusters, inverse_indices = torch.unique(labels, return_inverse=True)
    K = len(unique_clusters)
    N, D = X.shape

    mu = X.mean(dim=0)
    counts = torch.bincount(inverse_indices, minlength=K).to(dtype=X.dtype)
    sum_x = torch.zeros((K, D), dtype=X.dtype, device=X.device)
    sum_x.index_add_(0, inverse_indices, X)
    mu_q = sum_x / counts.unsqueeze(1)

    diff_mu = mu_q - mu
    B = (counts * (diff_mu * diff_mu).sum(dim=1)).sum()

    diff_x = X - mu_q[inverse_indices]
    W = (diff_x * diff_x).sum()

    if W == 0.0:
        return X.new_tensor(1.0)

    score = (B / (K - 1)) / (W / (N - K))
    return score


def davies_bouldin_score(X, labels):
    """返回 DB，框架传入全量原始特征。

    S_q 为簇内到中心的平均欧氏距离，M_qr 为两中心的欧氏距离。
    DB=mean_q max_{r!=q}(S_q+S_r)/M_qr。
    近零判断固定使用 atol=1e-8、rtol=0，float32/float64 使用同一阈值：
    所有 S_q <= 1e-8 或所有 M_qr <= 1e-8 时返回0；
    其余情况下中心距离恰为0的对不参与最大值（距离置为无穷）。
    正的中心距离仍参与计算，不按近零阈值逐对截断。
    """
    unique_clusters, inverse_indices = torch.unique(labels, return_inverse=True)
    K = len(unique_clusters)
    N, D = X.shape

    counts = torch.bincount(inverse_indices, minlength=K).to(dtype=X.dtype)
    sum_x = torch.zeros((K, D), dtype=X.dtype, device=X.device)
    sum_x.index_add_(0, inverse_indices, X)
    centers = sum_x / counts.unsqueeze(1)

    diff = X - centers[inverse_indices]
    point_dists = torch.linalg.vector_norm(diff, dim=1)
    sum_dists = torch.zeros(K, dtype=X.dtype, device=X.device)
    sum_dists.index_add_(0, inverse_indices, point_dists)
    S = sum_dists / counts

    diff_centers = centers.unsqueeze(1) - centers.unsqueeze(0)
    M = torch.linalg.vector_norm(diff_centers, dim=2)

    if bool((S <= 1e-8).all()):
        return X.new_tensor(0.0)

    eye_mask = torch.eye(K, dtype=torch.bool, device=X.device)
    off_diag_M = M[~eye_mask]
    if bool((off_diag_M <= 1e-8).all()):
        return X.new_tensor(0.0)

    S_sum = S.unsqueeze(1) + S.unsqueeze(0)
    valid_pair = (~eye_mask) & (M > 0.0)
    ratios = torch.where(valid_pair, S_sum / M, torch.full_like(S_sum, float('-inf')))
    max_ratios, _ = torch.max(ratios, dim=1)
    max_ratios = torch.where(torch.isneginf(max_ratios), torch.zeros_like(max_ratios), max_ratios)
    return max_ratios.mean()
