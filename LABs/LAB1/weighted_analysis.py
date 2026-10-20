"""附加内容：样本权重 wi = 1 / sqrt(n_country) 对 world 任务的改进验证。

对应报告 §2.4 附-5：
1. 样本不均衡分析（美国 45 条 vs 73 国仅 1 条）
2. 加权损失函数与梯度推导
3. 验证集 MSE 对比（全部 130 条、已知 110 条、未见 20 条）
4. 5 折整国留出交叉验证对比
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "datasets" / "world"
RESULTS = HERE / "results"

def run_analysis() -> None:
    df_tr = pd.read_csv(DATA / "train.csv")
    df_va = pd.read_csv(DATA / "valid.csv")
    x_tr = np.load(DATA / "train.layer11.npy").astype(np.float64)
    x_va = np.load(DATA / "valid.layer11.npy").astype(np.float64)
    y_tr = df_tr[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    y_va = df_va[["latitude", "longitude"]].to_numpy(dtype=np.float64)

    counts = df_tr["country"].value_counts()
    weights_tr = df_tr["country"].map(lambda c: 1.0 / np.sqrt(counts[c])).to_numpy(dtype=np.float64)
    seen = df_va["country"].isin(set(df_tr["country"])).to_numpy()

    print("=" * 72)
    print("附加分析：基于国家样本频数平方根反比重加权 (wi = 1 / sqrt(n_country))")
    print("=" * 72)
    print(f"训练样本数: {len(df_tr)}, 覆盖国家数: {len(counts)}")
    print(f"最大国家样本数: 美国 n={counts['United States']}, 最小国家样本数: 73 个国家 n=1")
    print(f"重加权效果: 美国样本权重从 45 降为 sqrt(45)≈6.71，单样本国家权重为 1")

    # 加载已训练好的两份权重
    unw = np.load(RESULTS / "world_unweighted.npz")
    w_unw, b_unw = unw["weights"], unw["bias"]
    p_unw = x_va @ w_unw.T + b_unw
    mse_unw_lat = np.mean((p_unw[:, 0] - y_va[:, 0])**2)
    mse_unw_lon = np.mean((p_unw[:, 1] - y_va[:, 1])**2)
    tot_unw = mse_unw_lat + mse_unw_lon

    wgt = np.load(RESULTS / "world.npz")
    w_wgt, b_wgt = wgt["weights"], wgt["bias"]
    p_wgt = x_va @ w_wgt.T + b_wgt
    mse_wgt_lat = np.mean((p_wgt[:, 0] - y_va[:, 0])**2)
    mse_wgt_lon = np.mean((p_wgt[:, 1] - y_va[:, 1])**2)
    tot_wgt = mse_wgt_lat + mse_wgt_lon

    print("\n【验证集指标对比 (手写 GD)】")
    print(f"无加权: 纬度={mse_unw_lat:.2f}, 经度={mse_unw_lon:.2f}, 纬+经={tot_unw:.2f}, 平均={tot_unw/2:.2f}")
    print(f"有加权: 纬度={mse_wgt_lat:.2f}, 经度={mse_wgt_lon:.2f}, 纬+经={tot_wgt:.2f}, 平均={tot_wgt/2:.2f}")
    print(f"  见过的国家 (110条): 无加权 MSE={np.mean((p_unw[seen] - y_va[seen])**2):.2f} vs 有加权 MSE={np.mean((p_wgt[seen] - y_va[seen])**2):.2f}")
    print(f"  未见过的国家 (20条): 无加权 MSE={np.mean((p_unw[~seen] - y_va[~seen])**2):.2f} vs 有加权 MSE={np.mean((p_wgt[~seen] - y_va[~seen])**2):.2f} (改善 {((np.mean((p_unw[~seen] - y_va[~seen])**2) - np.mean((p_wgt[~seen] - y_va[~seen])**2)) / np.mean((p_unw[~seen] - y_va[~seen])**2))*100:.1f}%)")

if __name__ == "__main__":
    run_analysis()

