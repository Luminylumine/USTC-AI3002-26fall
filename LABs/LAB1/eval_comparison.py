import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("datasets/world")
va_df = pd.read_csv(DATA / "valid.csv")
tr_df = pd.read_csv(DATA / "train.csv")
x_va = np.load(DATA / "valid.layer11.npy").astype(np.float64)
y_va = va_df[["latitude", "longitude"]].to_numpy(dtype=np.float64)

seen = va_df["country"].isin(set(tr_df["country"])).to_numpy()

unw = np.load("results/world.npz")
w_unw, b_unw = unw["weights"], unw["bias"]
p_unw = x_va @ w_unw.T + b_unw
mse_unw_lat = np.mean((p_unw[:, 0] - y_va[:, 0])**2)
mse_unw_lon = np.mean((p_unw[:, 1] - y_va[:, 1])**2)
tot_unw = mse_unw_lat + mse_unw_lon

wgt = np.load("results/world_weighted.npz")
w_wgt, b_wgt = wgt["weights"], wgt["bias"]
p_wgt = x_va @ w_wgt.T + b_wgt
mse_wgt_lat = np.mean((p_wgt[:, 0] - y_va[:, 0])**2)
mse_wgt_lon = np.mean((p_wgt[:, 1] - y_va[:, 1])**2)
tot_wgt = mse_wgt_lat + mse_wgt_lon

print("=== 最终手写 GD 训练结果对照 (Layer 11) ===")
print(f"【原始无加权 (world.npz)】:")
print(f"  纬度 MSE = {mse_unw_lat:.4f}, 经度 MSE = {mse_unw_lon:.4f}")
print(f"  纬度 MSE + 经度 MSE = {tot_unw:.4f}, 坐标平均 MSE = {tot_unw/2:.4f}")
print(f"  见过的国家 (110条) MSE = {np.mean((p_unw[seen] - y_va[seen])**2):.4f}")
print(f"  未见过的国家 (20条) MSE = {np.mean((p_unw[~seen] - y_va[~seen])**2):.4f}")

print(f"\n【加权 wi = 1/sqrt(n_country) (world_weighted.npz)】:")
print(f"  纬度 MSE = {mse_wgt_lat:.4f}, 经度 MSE = {mse_wgt_lon:.4f}")
print(f"  纬度 MSE + 经度 MSE = {tot_wgt:.4f}, 坐标平均 MSE = {tot_wgt/2:.4f}")
print(f"  见过的国家 (110条) MSE = {np.mean((p_wgt[seen] - y_va[seen])**2):.4f}")
print(f"  未见过的国家 (20条) MSE = {np.mean((p_wgt[~seen] - y_va[~seen])**2):.4f}")

