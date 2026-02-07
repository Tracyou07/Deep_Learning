

import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer

# ===== 路径配置 =====
ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = ROOT / "tokenizer" / "bpe_tokenizer.json"
CKPT_PATH = ROOT / "ckpt" / "gpt_sentence.pt"

import sys
sys.path.append(str(ROOT))
from model.transformer import GPT  

@torch.no_grad()
def greedy_sample(model, tok, prompt: str, max_new: int = 8, T_ctx: int = 128, device="cpu"):

    model.eval()
    ids = tok.encode(prompt).ids
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) 
    for _ in range(max_new):
        x_cond = x if x.size(1) <= T_ctx else x[:, -T_ctx:]
        logits = model(x_cond)[:, -1, :]       
        next_id = torch.argmax(logits, dim=-1) 
        x = torch.cat([x, next_id.unsqueeze(-1)], dim=1)
    return tok.decode(x[0].tolist())

# ===== 主程序 =====
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载 tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    PAD = tok.token_to_id("[PAD]")

    # 2. 加载模型和权重
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = GPT(
        vocab_size=ckpt["vocab_size"],
        T=128, d_model=256, n_layers=4,
        n_heads=2, p=32, d_ff=1024,
        pad_id=ckpt["pad_id"], dropout=0.1
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # 3. prompt 示例
    prompts = [
        "The Eiffel Tower is in Paris",
        "The mystery of life is",
        "Once upon a time there was",
    ]

    # 4. 逐个采样并输出结果
    for p in prompts:
        out = greedy_sample(model, tok, prompt=p, max_new=8, T_ctx=128, device=device)
        print(f"\nPrompt: {p}\nCompletion: {out}\n")
