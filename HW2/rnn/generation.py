import math, random, os, io
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Config
train_files = [
    "war_and_peace.txt",          
    "Siddhartha.txt"       
]
test_files  = [
    "Shakespeare.txt"             
]

SEQ_LEN   = 32          
BATCH     = 64
HIDDEN    = 256        
LR        = 2e-3
UPDATES   = 20000       
PRINT_EVERY = 200
EVAL_EVERY  = 1000


# build train/test data set and vocab.
def read_all(paths):
    buf = []
    for p in paths:
        with io.open(p, "r", encoding="utf-8", errors="ignore") as f:
            buf.append(f.read())
    return "".join(buf)

train_text = read_all(train_files)
test_text  = read_all(test_files)

vocab = sorted(list(set(train_text)))
UNK = "<UNK>"
if any(ch not in vocab for ch in test_text):
    vocab = [UNK] + vocab  

stoi = {ch:i for i,ch in enumerate(vocab)}
itos = {i:ch for ch,i in stoi.items()}
V = len(vocab)
print(f"Vocabulary size: {V}")

def encode(s):
    if UNK in stoi:
        return [stoi.get(ch, stoi[UNK]) for ch in s]
    else:
        return [stoi[ch] for ch in s if ch in stoi]
    
# one-hot encoding
def to_one_hot(x_idx, V):
    B, T = x_idx.shape
    oh = torch.zeros(B, T, V, device=x_idx.device)
    oh.scatter_(2, x_idx.unsqueeze(-1), 1.0)
    return oh


# Sliding window to dataloader
class CharSeqDataset(Dataset):
    def __init__(self, text, seq_len):
        self.idx = encode(text)
        self.seq_len = seq_len
        self.n = len(self.idx) - (seq_len + 1)
        self.n = max(self.n, 0)
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        x = torch.tensor(self.idx[i:i+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.idx[i+1:i+1+self.seq_len], dtype=torch.long)
        return x, y

train_ds = CharSeqDataset(train_text, SEQ_LEN)
val_ds   = CharSeqDataset(test_text,  SEQ_LEN)  
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, drop_last=True)



# Rnn model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden):
        super().__init__()
        self.rnn = nn.RNN(input_size=vocab_size,
                          hidden_size=hidden,
                          num_layers=1,
                          nonlinearity='tanh',
                          batch_first=True)
        self.fc  = nn.Linear(hidden, vocab_size)
    def forward(self, x_oh, h0=None):
        out, hT = self.rnn(x_oh, h0)          
        logits = self.fc(out)                  
        return logits, hT

model = CharRNN(V, HIDDEN).to(device)
criterion = nn.CrossEntropyLoss()           
optimizer = torch.optim.Adam(model.parameters(), lr=LR)




ckpt_path = "checkpoints/best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
print(f" Loaded checkpoint from {ckpt_path}")

vocab = ckpt["vocab"]
stoi = ckpt["stoi"]
itos = ckpt["itos"]
UNK = "<UNK>" if "<UNK>" in stoi else None
V = len(vocab)

model = CharRNN(V, HIDDEN).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()



#eval
def sample_text(seed=" ", length=400, temperature=1.0, greedy=False):

    h = torch.zeros(1, 1, HIDDEN, device=device)

    for ch in seed[:-1]:
        x_idx = torch.tensor([[stoi.get(ch, stoi.get(UNK, 0))]], device=device)
        x_oh = to_one_hot(x_idx, V)
        _, h = model(x_oh, h)

    cur = seed[-1] if seed else " "
    out = [c for c in seed]

 
    for _ in range(length):
        x_idx = torch.tensor([[stoi.get(cur, stoi.get(UNK, 0))]], device=device) 
        x_oh = to_one_hot(x_idx, V)
        logits, h = model(x_oh, h)              
        logits = logits[0, 0] / max(temperature, 1e-6)

        if greedy:
            nxt_idx = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits, dim=-1)
            nxt_idx = int(torch.multinomial(probs, 1).item())

        cur = itos[nxt_idx]
        out.append(cur)

    return "".join(out)

print("\n=== Generated text (greedy) ===")
print(sample_text(seed="The ", length=300, greedy=True))

print("\n=== Generated text (temp=1.0, sampling) ===")
print(sample_text(seed="The ", length=300, temperature=1.0, greedy=False))
