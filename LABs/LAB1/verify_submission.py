import numpy as np

w = np.load("results/world.npz")
print("world.npz files:", w.files)
print("  weights shape:", w["weights"].shape, "dtype:", w["weights"].dtype)
print("  bias shape:", w["bias"].shape, "dtype:", w["bias"].dtype)

t = np.load("results/truth.npz")
print("truth.npz files:", t.files)
print("  weights shape:", t["weights"].shape, "dtype:", t["weights"].dtype)
print("  bias shape:", t["bias"].shape, "dtype:", t["bias"].dtype)

