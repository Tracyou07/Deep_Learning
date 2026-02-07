#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def moving_average(x, w):
    if w is None or w <= 1:
        return x
    return np.convolve(x, np.ones(w, dtype=float)/w, mode="same")

def main():
    parser = argparse.ArgumentParser(description="Plot pretty epoch-level curves from training CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Path to train_log.csv")
    parser.add_argument("--outdir", type=str, default="figs", help="Directory to save figures")
    parser.add_argument("--smooth", type=int, default=0, help="Moving average window (in epochs). 0 disables.")
    parser.add_argument("--title", type=str, default="Training Curves (per-epoch mean)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Matplotlib aesthetics (clean, publication-ready) ----
    mpl.rcParams.update({
        "figure.figsize": (9, 5.2),
        "figure.dpi": 140,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "font.family": "DejaVu Sans",
    })

    # A colorblind-friendly palette
    colors = {
        "train_loss": "#1f77b4",  # blue
        "val_loss":   "#d62728",  # red
        "val_ppl":    "#2ca02c",  # green
    }

    # ---- Load & normalize columns ----
    df = pd.read_csv(args.csv)
    df.columns = [c.lower().strip() for c in df.columns]

    # epoch column
    epoch_col = None
    for c in ["epoch", "epochs", "ep"]:
        if c in df.columns:
            epoch_col = c
            break
    if epoch_col is None:
        raise ValueError("CSV 中未找到 'epoch' 列。建议在训练时把 epoch 写入日志，或先离线添加。")

    # metrics
    train_loss_col = next((c for c in ["train_loss", "loss", "training_loss"] if c in df.columns), None)
    val_loss_col   = next((c for c in ["val_loss", "valid_loss", "validation_loss"] if c in df.columns), None)
    val_ppl_col    = next((c for c in ["val_ppl", "validation_ppl", "ppl"] if c in df.columns), None)

    if train_loss_col is None and val_loss_col is None and val_ppl_col is None:
        raise ValueError("未找到 train_loss/val_loss/val_ppl 任一列，请检查列名。")

    # ---- Aggregate per epoch (mean) ----
    group_cols = [c for c in [train_loss_col, val_loss_col, val_ppl_col] if c is not None]
    epoch_df = df.groupby(epoch_col)[group_cols].mean().reset_index().sort_values(epoch_col)
    x = epoch_df[epoch_col].values

    # optional smoothing
    W = args.smooth if args.smooth and args.smooth > 1 else None
    series = {}
    if train_loss_col:
        series["train_loss"] = moving_average(epoch_df[train_loss_col].values, W)
    if val_loss_col:
        series["val_loss"] = moving_average(epoch_df[val_loss_col].values, W)
    if val_ppl_col:
        series["val_ppl"] = moving_average(epoch_df[val_ppl_col].values, W)

    # ---- Plot: twin y-axes (loss on left, ppl on right) ----
    fig, ax_loss = plt.subplots()

    # left axis: losses
    handles = []
    labels  = []
    if "train_loss" in series:
        h1, = ax_loss.plot(x, series["train_loss"], marker="o", linewidth=2.0, markersize=4,
                           color=colors["train_loss"], label="train_loss")
        handles.append(h1); labels.append("train_loss")
    if "val_loss" in series:
        h2, = ax_loss.plot(x, series["val_loss"], marker="s", linewidth=2.0, markersize=4,
                           color=colors["val_loss"], label="val_loss")
        handles.append(h2); labels.append("val_loss")

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")

    # right axis: ppl
    if "val_ppl" in series:
        ax_ppl = ax_loss.twinx()
        # keep right spine visible but subtle
        ax_ppl.spines["right"].set_visible(True)
        h3, = ax_ppl.plot(x, series["val_ppl"], marker="^", linewidth=2.0, markersize=4,
                          color=colors["val_ppl"], label="val_ppl")
        ax_ppl.set_ylabel("perplexity")
        # nice alignment for twin axes grids
        ax_ppl.grid(False)
        handles.append(h3); labels.append("val_ppl")

    # title & legend
    title = args.title if not W else f"{args.title} (moving avg: {W})"
    ax_loss.set_title(title)
    ax_loss.legend(handles, labels, ncols=min(3, len(labels)), loc="upper right")

    # x ticks as integers
    ax_loss.set_xticks(x)
    ax_loss.set_xticklabels([str(int(e)) for e in x])

    # save
    png_path = os.path.join(args.outdir, "training_curves_epoch.png")
    pdf_path = os.path.join(args.outdir, "training_curves_epoch.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}\nSaved: {pdf_path}")

    # ---- Also save an epoch-mean CSV for record ----
    epoch_csv = os.path.join(args.outdir, "train_log_epoch_mean.csv")
    epoch_df.to_csv(epoch_csv, index=False)
    print(f"Saved epoch mean CSV: {epoch_csv}")

if __name__ == "__main__":
    main()
