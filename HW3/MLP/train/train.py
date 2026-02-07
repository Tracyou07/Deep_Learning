import math
import numpy as np

import torch
import torch.nn as nn
from dataset.dataset import generate_dataset
from model.MLP import MLP


# ===== 训练 =====
def train_mlp(x_np, y_np, epochs=3000, lr=1e-2, weight_decay=0, batch_size=0, seed=0, hidden=64, layers=3, verbose=True):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    x = torch.from_numpy(x_np).float().to(device)
    y = torch.from_numpy(y_np).float().to(device)

    model = MLP(hidden=hidden, layers=layers).to(device)
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    # —— 三阶段学习率：大 -> 中 -> 小（放在循环外初始化）——
    lr_big   = lr
    lr_mid   = lr * 0.5
    lr_small = lr * 0.4
    lr_small_2 = lr * 0.35
    lr_small_3 = lr * 0.2
    lr_small_4 = lr * 0.05
    lr_small_5 = lr * 0.01

    e1 = int(0.1 * epochs)   # [1, e1]    用 lr_big
    e2 = int(0.2 * epochs)   # (e1, e2]   用 lr_mid
    e3 = int(0.4 * epochs)
    e4 = int(0.6 * epochs)
    e5 = int(0.8 * epochs)
    e6 = int(0.9 * epochs)
                              

    n = x.shape[0]
    for ep in range(1, epochs + 1):
        # 每个 epoch 开头按阶段设置当前 lr
        if ep <= e1:
            cur_lr = lr_big
        elif ep <= e2:
            cur_lr = lr_mid
        elif ep <= e3:
            cur_lr = lr_small
        elif ep <= e4:
            cur_lr = lr_small_2
        elif ep <= e5:
            cur_lr = lr_small_3
        elif ep <= e6:
            cur_lr = lr_small_4
        else:
            cur_lr = lr_small_5           
        for g in opt.param_groups:
            g['lr'] = cur_lr

        # —— 正常训练 —— 
        if batch_size and batch_size < n:
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i+batch_size]
                xb, yb = x[idx], y[idx]
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * xb.shape[0]
            train_loss = total / n
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss = float(loss.item())

        if verbose and (ep % 500 == 0 or ep in (1, epochs)):
            print(f"[ep {ep:5d}] lr={cur_lr:.3e}  train MSE = {train_loss:.6e}")

        if train_loss < 1e-6:
            if verbose:
                print(f"[early stop] train MSE < 1e-6 at ep={ep}")
            break

    with torch.no_grad():
        final_mse = float(loss_fn(model(x), y).item())
    return model, final_mse

# ===== δ 指标 =====
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

# demo
if __name__ == "__main__":
    x_np, y_np = generate_dataset(200, seed=0)
    model, train_mse = train_mlp(x_np, y_np, epochs=400000, lr=2e-2, weight_decay=0, batch_size=0, seed=0)
    print("final train MSE =", train_mse)
    print("delta_fin  =", delta_fin(model))
    print("delta_fout =", delta_fout(model))
