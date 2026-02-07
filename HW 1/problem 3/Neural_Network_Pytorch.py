
import torch
import torch.nn as nn
import torchvision as thv
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# reproducibility & device
seed = 0
torch.manual_seed(seed); np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Dataset
root = "./"
train = thv.datasets.MNIST(root, download=True, train=True)
val   = thv.datasets.MNIST(root, download=True, train=False)

def stratified_half_indices(targets: np.ndarray, rng=42):
    rs = np.random.RandomState(rng)
    keep = []
    for c in range(10):
        idx = np.where(targets == c)[0]
        rs.shuffle(idx)
        keep.append(idx[:len(idx)//2])
    keep = np.concatenate(keep)
    rs.shuffle(keep)
    return keep

x_tr = train.data.numpy().astype(np.float32)/255.0
y_tr = train.targets.numpy().astype(np.int64)
x_va = val.data.numpy().astype(np.float32)/255.0
y_va = val.targets.numpy().astype(np.int64)

tr_idx = stratified_half_indices(y_tr, rng=123)
va_idx = stratified_half_indices(y_va, rng=456)

x_tr_t = torch.from_numpy(x_tr[tr_idx]).unsqueeze(1)  # (N,1,28,28)
y_tr_t = torch.from_numpy(y_tr[tr_idx])
x_va_t = torch.from_numpy(x_va[va_idx]).unsqueeze(1)
y_va_t = torch.from_numpy(y_va[va_idx])

train_ds = TensorDataset(x_tr_t, y_tr_t)
val_ds   = TensorDataset(x_va_t, y_va_t)

# DataLoader 
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)


# Model with 4 layers
class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv2d(1, 8, kernel_size=4, stride=4, bias=True)  
        self.fc1   = nn.Linear(8*7*7, 128)
        self.relu  = nn.ReLU()
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        h = self.embed(x)         
        h = h.flatten(1)   
        # model       
        h = self.relu(self.fc1(h))
        return self.fc2(h)        

@torch.no_grad()
def evaluate(model, loader, lossf):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = lossf(logits, yb)
        total_loss += loss.item()*yb.size(0)
        total_correct += (logits.argmax(1)==yb).sum().item()
        total += yb.size(0)
    return total_loss/total, 1.0 - total_correct/total  # (loss, error)


# Training config
model = SimpleMNISTModel().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
lossf = nn.CrossEntropyLoss()

batch_size = 64
updates = 12000             
eval_every = 100          

rng = np.random.RandomState(0)
def sample_batch(ds, bs):
    n = len(ds)
    idx = rng.choice(n, size=bs, replace=False)
    return ds.tensors[0][idx], ds.tensors[1][idx]

# Train
steps = []
train_losses, train_errors = [], []
val_losses,   val_errors   = [], []

for t in range(1, updates+1):
    model.train()
    xb, yb = sample_batch(train_ds, batch_size)
    xb, yb = xb.to(device), yb.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = lossf(logits, yb)
    loss.backward()
    optimizer.step()

    if t % eval_every == 0:
        # training metrics
        with torch.no_grad():
            preds = logits.argmax(1)
            train_err = 1.0 - (preds==yb).float().mean().item()
            train_loss = loss.item()

        # validation metrics
        vloss, verr = evaluate(model, val_loader, lossf)

        steps.append(t)
        train_losses.append(train_loss); train_errors.append(train_err)
        val_losses.append(vloss);        val_errors.append(verr)

        print(f"[{t:5d}] train_loss={train_loss:.4f} train_err={train_err:.3f} | "
              f"val_loss={vloss:.4f} val_err={verr:.3f}")


# Plots
plt.figure(figsize=(9,4))
plt.plot(steps, train_losses, label='train loss')
plt.plot(steps, val_losses, label='val loss')
plt.xlabel('weight updates'); plt.ylabel('loss')
plt.title('Training/Validation Loss vs Updates')
plt.grid(True); plt.legend()
plt.show()

plt.figure(figsize=(9,4))
plt.plot(steps, train_errors, label='train error')
plt.plot(steps, val_errors, label='val error')
plt.xlabel('weight updates'); plt.ylabel('error rate')
plt.title('Training/Validation Error vs Updates')
plt.grid(True); plt.legend()
plt.show()
