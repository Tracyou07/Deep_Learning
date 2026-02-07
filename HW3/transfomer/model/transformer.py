import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 正余弦位置编码 ---
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)

# --- 多头自注意力（heads=2, 每头 p=32）---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 2, p: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads, self.p = n_heads, p
        proj = n_heads * p  #\
        self.q = nn.Linear(d_model, proj, bias=False)
        self.k = nn.Linear(d_model, proj, bias=False)
        self.v = nn.Linear(d_model, proj, bias=False)
        self.out = nn.Linear(proj, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    #attention
    def forward(self, x, attn_mask):  
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.p).transpose(1, 2)  
        k = self.k(x).view(B, T, self.n_heads, self.p).transpose(1, 2)  
        v = self.v(x).view(B, T, self.n_heads, self.p).transpose(1, 2)  

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.p)             
        att = att.masked_fill(attn_mask == 0, float("-inf"))            
        att = F.softmax(att, dim=-1)                                    
        att = self.attn_drop(att)                                       # 对权重做 dropout

        y = att @ v                                                     
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads*self.p)  
        y = self.resid_drop(self.out(y))                                # W_o: projection
        return y

# --- Block：attn + 残差 + LN + MLP---
class DecoderBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=2, p=32, d_ff=1024, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, p, dropout)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        h_attn = self.ln1(x + self.attn(x, attn_mask))     
        out    = self.ln2(h_attn + self.mlp(h_attn))        
        return out

# --- 整体 Decoder-only ---
class GPT(nn.Module):
    def __init__(self, vocab_size, T=128, d_model=256, n_layers=4, n_heads=2, p=32,
                 d_ff=1024, pad_id=0, dropout=0.1):
        super().__init__()
        self.T, self.pad_id = T, pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=T)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, p, d_ff, dropout)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model, elementwise_affine=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享（可选）
        self.head.weight = self.tok_emb.weight

    @torch.no_grad()
    def _causal_mask(self, T, device):
        return torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)  

    def _build_attn_mask(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        causal = self._causal_mask(T, device)                                
        key_pad = (input_ids != self.pad_id).unsqueeze(1).unsqueeze(2)       
        return causal * key_pad                                              

    #total forward process
    def forward(self, input_ids):  
        assert input_ids.size(1) <= self.T, "sequence length exceeds context size"
        x = self.tok_emb(input_ids)                
        x = self.pos_enc(x)
        mask = self._build_attn_mask(input_ids)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)                          

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            if out.size(1) > self.T: out = out[:, -self.T:]
            logits = self.forward(out)[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, ix = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float("-inf")).scatter(1, ix, v)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            out = torch.cat([out, next_id], dim=1)
        return out
