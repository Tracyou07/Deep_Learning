# train_sentences.py
# Sentence-level batching + dynamic padding + real-time progress bar (tqdm)
# Usage:
#   python train_sentences.py --epochs 3 --batch_size 64 --lr 3e-4
#   (可修改文件列表 train_files/test_files)

import os, math, time, argparse, random, re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


import torch
import torch.nn.functional as F
from tqdm import tqdm
from tokenizers import Tokenizer

# -------- 项目根与路径 --------
ROOT = Path(__file__).resolve().parents[1]  # .../3
DATA_DIR = ROOT / "data"
TOKEN_DIR = ROOT / "tokenizer"
MODEL_DIR = ROOT / "model"
CKPT_DIR = ROOT / "ckpt"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


import sys
sys.path.append(str(ROOT))  
from model.transformer import GPT
from train_val_set.dataset import make_sentence_loaders

# ===================== 评估与工具 =====================

def set_seed(seed=1337):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, pad_id, vocab_size, device):
    model.eval()
    total, ntok = 0.0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        if x.numel() == 0:  # 防御性
            continue
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                               y.reshape(-1),
                               ignore_index=pad_id,
                               reduction="sum")
        total += loss.item()
        ntok += (y != pad_id).sum().item()
    avg = total / max(1, ntok)
    ppl = math.exp(avg)
    return avg, ppl


# ===================== 训练主程序 =====================

def main(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("Device:", device)

    # 1) 准备 tokenizer & vocab
    tokenizer_path = TOKEN_DIR / "bpe_tokenizer.json"
    print("Tokenizer:", tokenizer_path)
    tok = Tokenizer.from_file(str(tokenizer_path))
    PAD = tok.token_to_id("[PAD]")
    V = tok.get_vocab_size()
    print(f"Vocab={V} PAD={PAD}")

    # 2) 选择文件（可按需调整）
    tokenizer_path = TOKEN_DIR / "bpe_tokenizer.json"
    train_files = [DATA_DIR / "war_and_peace.txt", DATA_DIR / "les_miserables.txt"]
    test_files  = [DATA_DIR / "Shakespeare.txt"]

    tok, PAD, V, train_loader, val_loader, test_loader, stats = make_sentence_loaders(
        tokenizer_path=tokenizer_path,
        train_files=train_files,
        test_files=test_files,
        T=args.T,
        batch_size=args.batch_size,
        workers=args.workers,
        val_ratio=args.val_ratio,
    )
    print(f"Train sentences: {stats['n_train']} | Val: {stats['n_val']} | Test: {stats['n_test']}")


    # 5) 模型
    model = GPT(
        vocab_size=V, T=args.T, d_model=256, n_layers=args.layers,
        n_heads=2, p=32, d_ff=1024, pad_id=PAD, dropout=args.dropout
    ).to(device)

    # 6) 优化器 + 简易 warmup
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = max(1, len(train_loader) * args.epochs)
    warm_steps = max(1, int(args.warmup_ratio * total_steps))
    global_step = 0

    def set_lr():
        """linear warmup -> constant"""
        lr = args.lr * min(1.0, (global_step + 1) / warm_steps)
        for g in opt.param_groups: g["lr"] = lr
        return lr

    # 7) 训练循环（实时进度）
    train_losses, val_losses, val_ppls = [], [], []
    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", dynamic_ncols=True)
        t_start = time.time()
        tokens_counted = 0
        running = 0.0

        for x, y in pbar:
            if x.numel() == 0:  # 空batch保护
                continue
            x = x.to(device); y = y.to(device)
            logits = model(x)  # [B,L,V]
            loss = F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1), ignore_index=PAD
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr_now = set_lr()
            opt.step()
            global_step += 1

            # 进度条信息
            tokens_in_batch = (y != PAD).sum().item()
            tokens_counted += tokens_in_batch
            elapsed = time.time() - t_start
            tps = tokens_counted / max(1e-9, elapsed)

            running += loss.item()
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{running / (pbar.n if pbar.n>0 else 1):.4f}",
                                 lr=f"{lr_now:.2e}",
                                 tps=f"{tps:.0f}")

        # 验证
        if val_loader:
            val_loss, val_ppl = evaluate(model, val_loader, PAD, V, device)
            print(f"[eval] epoch {ep} | val_loss/tok {val_loss:.4f} | ppl {val_ppl:.2f}")

            # 记录训练与验证结果
            avg_train_loss = running / max(1, len(train_loader))
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            val_ppls.append(val_ppl)

            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = CKPT_DIR / "gpt_sentence.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "vocab_size": V,
                    "pad_id": PAD,
                    "args": vars(args),
                }, ckpt_path)
                print(f"✅ Saved checkpoint: {ckpt_path}")

    # 测试
    if test_loader:
        test_loss, test_ppl = evaluate(model, test_loader, PAD, V, device)
        print(f"[test] loss/tok {test_loss:.4f} | ppl {test_ppl:.2f}")

    # 采样展示（句子补全）
    if args.sample_text:
        print("\n=== Sampling ===")
        model.eval()
        prompt_ids = tok.encode(args.sample_text).ids[:args.T]
        x0 = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.generate(x0, max_new_tokens=args.sample_len, temperature=0.9, top_k=50)
        print(tok.decode(out[0].tolist()))

    # -------- all data in csv --------
    pd.DataFrame({
        "epoch": range(1, len(train_losses)+1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_ppl": val_ppls
    }).to_csv(CKPT_DIR / "train_log.csv", index=False)
    print(f"📄 Saved training log to {CKPT_DIR / 'train_log.csv'}")

    # -------- Plot loss curves --------
    if val_losses:
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, "o-", label="Train Loss")
        plt.plot(epochs, val_losses, "s-", label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)
        curve_path = CKPT_DIR / "train_curve.png"
        plt.savefig(curve_path, dpi=200)
        plt.show()
        print(f"📉 Saved loss curve to {curve_path}")


# ===================== 参数 =====================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=128, help="context length (cap for sentences)")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_ratio", type=float, default=0.05, help="split from train_sents tail")
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--sample_text", type=str, default="The mystery of life is not a problem to be solved")
    ap.add_argument("--sample_len", type=int, default=80)
    args = ap.parse_args()
    main(args)
