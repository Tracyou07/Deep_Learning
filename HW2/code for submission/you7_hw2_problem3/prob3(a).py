# CIFAR-10 All-CNN training script (100 epochs, target <10% val error)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
from collections import defaultdict
import os, math, random
import numpy as np

def seed_everything(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
seed_everything(42)

# model
class View(nn.Module):
    def __init__(self,o): super().__init__(); self.o = o
    def forward(self,x): return x.view(-1, self.o)

def convbn(ci,co,ksz,s=1,pz=0):
    return nn.Sequential(
        nn.Conv2d(ci,co,ksz,stride=s,padding=pz), # 原实现: Conv -> ReLU -> BN
        nn.ReLU(True),
        nn.BatchNorm2d(co)
    )

class allcnn_t(nn.Module):
    def __init__(self, c1=96, c2=192):
        super().__init__()
        d = 0.5
        self.m = nn.Sequential(
            nn.Dropout(0.2),
            convbn(3,c1,3,1,1),
            convbn(c1,c1,3,1,1),
            convbn(c1,c1,3,2,1),      # 32->16
            nn.Dropout(d),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),      # 16->8
            nn.Dropout(d),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),          # -> [N,10,1,1]
            View(10)                  # -> [N,10]
        )
    def forward(self,x): return self.m(x)

# CIFAR-10 mean/std
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

root = "./data"
train_set = datasets.CIFAR10(root, train=True,  download=True, transform=train_tf)
test_set  = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

batch_size = 128
num_workers = 2
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = allcnn_t().to(device)

def accuracy(logits, targets):
    return (logits.argmax(1) == targets).float().mean().item()

bn_params, bias_params, other_params = [], [], []
for name, p in model.named_parameters():
    if not p.requires_grad: continue
    if ('bn' in name) or ('running_mean' in name) or ('running_var' in name):
        if name.endswith('weight') or name.endswith('bias'):
            bn_params.append(p)
    elif name.endswith('bias'):
        bias_params.append(p)
    else:
        other_params.append(p)

# weight decay = 1e-3
base_wd = 1e-3
optimizer = torch.optim.SGD([
    {'params': other_params, 'weight_decay': base_wd},
    {'params': bn_params,   'weight_decay': 0.0},
    {'params': bias_params, 'weight_decay': 0.0},
], lr=0.1, momentum=0.9, nesterov=True)

# lr：0-40:0.1, 40-80:0.01, 80-100:0.001
def lr_schedule(epoch):
    if epoch < 40: return 0.1
    elif epoch < 80: return 0.01
    else: return 0.001

criterion = nn.CrossEntropyLoss()

# ---------- Train / Validate ----------
history = defaultdict(list)
best_val_acc, best_state = 0.0, None

epochs = 100
for epoch in range(epochs):
    # set LR
    for g in optimizer.param_groups:
        g['lr'] = lr_schedule(epoch)

    # Train
    model.train()
    tr_loss_sum, tr_acc_sum, n_tr = 0.0, 0.0, 0
    for x,y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        tr_loss_sum += loss.item()*bs
        tr_acc_sum  += accuracy(logits, y)*bs
        n_tr += bs

    tr_loss = tr_loss_sum / n_tr
    tr_acc  = tr_acc_sum  / n_tr
    tr_err  = 1.0 - tr_acc

    # Val
    model.eval()
    va_loss_sum, va_acc_sum, n_va = 0.0, 0.0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            bs = x.size(0)
            va_loss_sum += loss.item()*bs
            va_acc_sum  += accuracy(logits, y)*bs
            n_va += bs
    va_loss = va_loss_sum / n_va
    va_acc  = va_acc_sum  / n_va
    va_err  = 1.0 - va_acc

    history['train_loss'].append(tr_loss)
    history['train_err'].append(tr_err)
    history['val_loss'].append(va_loss)
    history['val_err'].append(va_err)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

    print(f"Epoch {epoch+1:03d}/{epochs} | "
          f"LR {optimizer.param_groups[0]['lr']:.3g} | "
          f"Train loss {tr_loss:.4f} err {tr_err*100:.2f}% | "
          f"Val loss {va_loss:.4f} err {va_err*100:.2f}%")

os.makedirs("checkpoints", exist_ok=True)
torch.save(best_state, "checkpoints/allcnn_best.pth")
print(f"Best val acc: {best_val_acc*100:.2f}%, model saved to checkpoints/allcnn_best.pth")


fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].plot(history['train_loss'], label='train')
ax[0].plot(history['val_loss'],   label='val')
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(True, ls='--', alpha=0.3)

ax[1].plot([e*100 for e in history['train_err']], label='train err (%)')
ax[1].plot([e*100 for e in history['val_err']],   label='val err (%)')
ax[1].set_title("Error"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(True, ls='--', alpha=0.3)
plt.tight_layout(); plt.show()
