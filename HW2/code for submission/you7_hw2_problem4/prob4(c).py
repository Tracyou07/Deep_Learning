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




# evaluation
def batch_ce_and_err(logits, y):
    B,T,V = logits.shape
    loss = criterion(logits.reshape(B*T, V), y.reshape(B*T))
    with torch.no_grad():
        pred = logits.argmax(dim=-1)      
        acc  = (pred == y).float().mean().item()
        err  = 1.0 - acc
    return loss, err

def evaluate_on_val32():
    model.eval()
    with torch.no_grad():
        x,y = next(iter(val_loader))       
        x = x.to(device); y = y.to(device)
        x_oh = to_one_hot(x, V)
        logits, _ = model(x_oh)
        loss, err = batch_ce_and_err(logits, y)
    model.train()
    return loss.item(), err

# training loop
ckpt_dir = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

best_val_loss = float("inf")
last_ckpt     = os.path.join(ckpt_dir, "last.pt")
best_ckpt     = os.path.join(ckpt_dir, "best.pt")
tr_losses, tr_errs, val_losses, val_errs, steps = [], [], [], [], []
step = 0
model.train()
while step < UPDATES:
    for x, y in train_loader:
        x = x.to(device); y = y.to(device)
        x_oh = to_one_hot(x, V)                  
        logits, _ = model(x_oh)
        loss, err = batch_ce_and_err(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 稳定训练
        optimizer.step()

        step += 1
        if step % PRINT_EVERY == 0:
            tr_losses.append(loss.item()); tr_errs.append(err); steps.append(step)
            print(f"[{step}] train loss {loss.item():.3f} | err {err:.3f}")

        if step % EVAL_EVERY == 0:
            vloss, verr = evaluate_on_val32()
            if vloss < best_val_loss:
                best_val_loss = vloss
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": vloss,
                    "vocab": vocab,           
                    "stoi": stoi, "itos": itos
                }, best_ckpt)
                print(f"💾 Saved BEST to {best_ckpt} | step={step} | val_loss={vloss:.4f}")

            val_losses.append(vloss); val_errs.append(verr)
            print(f"          → val32 loss {vloss:.3f} | err {verr:.3f}")

        if step >= UPDATES:
            break

# plot
plt.figure()
plt.plot(steps, tr_losses, label="train loss")
plt.plot(steps[::EVAL_EVERY//PRINT_EVERY], val_losses, label="val loss (32-char)")
plt.xlabel("weight updates"); plt.ylabel("loss"); plt.legend(); plt.title("Loss vs updates")
plt.show()
plt.savefig('loss.png')

plt.figure()
plt.plot(steps, tr_errs, label="train error")
plt.plot(steps[::EVAL_EVERY//PRINT_EVERY], val_errs, label="val error (32-char)")
plt.xlabel("weight updates"); plt.ylabel("error rate"); plt.legend(); plt.title("Error vs updates")
plt.show()
plt.savefig('error.png')


