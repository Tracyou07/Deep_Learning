# eval_test.py
import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

# ===== 路径设置 =====
ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
DATA_DIR = ROOT / "data"
TOKENIZER_PATH = ROOT / "tokenizer" / "bpe_tokenizer.json"
CKPT_PATH = ROOT / "ckpt" / "gpt_sentence.pt"

import sys
sys.path.append(str(ROOT))
from model.transformer import GPT   # 或 HWGPT，看你模型文件名

# ===== 简易评估函数 =====
@torch.no_grad()
def evaluate(model, loader, pad_id, vocab_size, device):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    for x, y in tqdm(loader, desc="Evaluating", dynamic_ncols=True):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
            ignore_index=pad_id,
            reduction="sum"
        )
        total_loss += loss.item()
        n_tokens += (y != pad_id).sum().item()
    avg_loss = total_loss / n_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()

# ===== DataLoader (test set) =====
from train_val_set.dataset import make_sentence_loaders

train_files = [DATA_DIR / "war_and_peace.txt", DATA_DIR / "les_miserables.txt"]
test_files = [DATA_DIR / "Shakespeare.txt"]
tok, PAD, V, _, _, test_loader, stats = make_sentence_loaders(
    tokenizer_path=TOKENIZER_PATH,
    train_files=train_files,                # 不用train，只加载test
    test_files=test_files,
    T=128,
    batch_size=64,
    workers=0,
    val_ratio=0.0,
)

print(f"Loaded test set: {stats['n_test']} sentences, vocab={V}")

# ===== 加载模型权重 =====
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(CKPT_PATH, map_location=device)
model = GPT(
    vocab_size=ckpt["vocab_size"],
    T=128,
    d_model=256,
    n_layers=4,
    n_heads=2,
    p=32,
    d_ff=1024,
    pad_id=ckpt["pad_id"],
    dropout=0.1
).to(device)
model.load_state_dict(ckpt["model_state"], strict=False)

print(f"✅ Loaded checkpoint from {CKPT_PATH}")

# ===== 在 test set 上评估 =====
test_loss, test_ppl = evaluate(model, test_loader, PAD, V, device)
print(f"\n[test] loss/token = {test_loss:.4f} | PPL = {test_ppl:.2f}")
