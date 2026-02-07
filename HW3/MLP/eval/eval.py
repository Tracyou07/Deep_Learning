# HW3 Problem 2(c)
# 对 n ∈ logspace(10, 1000) 共 20 个取值，每个 n 重复 R 次，训练并计算 δ_in/δ_out
# 输出：打印统计并画 log-log 曲线（需要 matplotlib）

import math, csv
import numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt

from train.train import train_mlp
from dataset.dataset import generate_dataset

def f_star_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(10.0 * math.pi * (x ** 4))

@torch.no_grad()
def delta_fin(model: nn.Module, n_points=1000, device="cpu"):
    xs = torch.linspace(0.0, 1.0, steps=n_points, device=device).unsqueeze(1)
    return float(torch.max(torch.abs(model(xs) - f_star_torch(xs))).item())
@torch.no_grad()
def delta_fout(model: nn.Module, n_points=1000, device="cpu"):
    xs = torch.linspace(0.0, 1.5, steps=n_points, device=device).unsqueeze(1)
    return float(torch.max(torch.abs(model(xs) - f_star_torch(xs))).item())

# ===== 实验主流程 =====
if __name__ == "__main__":
    epochs, lr, wd = 4000, 1e-2, 1e-4
    repeats, hidden, layers, batch_size, seed0 = 5, 64, 3, 0, 0

    ns = np.logspace(1, 3, 20).astype(int)  # 10 → 1000
    R = repeats
    deltas_in = np.zeros((len(ns), R), dtype=np.float64)
    deltas_out = np.zeros((len(ns), R), dtype=np.float64)
    train_mse_mat = np.zeros((len(ns), R), dtype=np.float64)

    for i, n in enumerate(ns):
        for r in range(R):
            seed = seed0 + i*100 + r
            x_np, y_np = generate_dataset(n, seed)
            model, final_mse = train_mlp(x_np, y_np, epochs=epochs, lr=lr, weight_decay=wd,
                                         batch_size=batch_size, seed=seed, hidden=hidden, layers=layers)
            train_mse_mat[i, r] = final_mse
            deltas_in[i, r] = delta_fin(model)
            deltas_out[i, r] = delta_fout(model)

        print(f"[n={n:4d}] train MSE mean={train_mse_mat[i].mean():.3e}  "
              f"δ_in mean={deltas_in[i].mean():.3e}  δ_out mean={deltas_out[i].mean():.3e}")

    mean_in, std_in = deltas_in.mean(axis=1), deltas_in.std(axis=1)
    mean_out, std_out = deltas_out.mean(axis=1), deltas_out.std(axis=1)

    # 结果表（如需保存 CSV，可自行写入文件）
    header = ["n", "delta_in_mean", "delta_in_std", "delta_out_mean", "delta_out_std", "train_mse_mean"]
    print("\nSummary (first 5 rows):")
    print(header)
    for i in range(min(5, len(ns))):
        print([int(ns[i]), mean_in[i], std_in[i], mean_out[i], std_out[i], train_mse_mat[i].mean()])

    # 画图
    plt.figure(figsize=(7,5))
    plt.xscale("log"); plt.yscale("log")
    plt.plot(ns, mean_in, label=r"$\delta_{fin}$ mean")
    plt.fill_between(ns, np.maximum(mean_in - std_in, 1e-12), mean_in + std_in, alpha=0.2)
    plt.plot(ns, mean_out, label=r"$\delta_{fout}$ mean")
    plt.fill_between(ns, np.maximum(mean_out - std_out, 1e-12), mean_out + std_out, alpha=0.2)
    plt.xlabel("n (number of training samples)")
    plt.ylabel("delta (max abs error)")
    plt.title("HW3 Problem 2(c): δ vs n (mean ± std)")
    plt.legend(); plt.tight_layout(); plt.show()
