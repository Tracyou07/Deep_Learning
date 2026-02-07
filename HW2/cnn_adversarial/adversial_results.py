
import torch, torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Model definition 
class View(nn.Module):
    def __init__(self,o): super().__init__(); self.o=o
    def forward(self,x): return x.view(-1,self.o)

def convbn(ci,co,ksz,s=1,pz=0):
    return nn.Sequential(
        nn.Conv2d(ci,co,ksz,stride=s,padding=pz),
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
            convbn(c1,c1,3,2,1),
            nn.Dropout(d),
            convbn(c1,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,2,1),
            nn.Dropout(d),
            convbn(c2,c2,3,1,1),
            convbn(c2,c2,3,1,1),
            convbn(c2,10,1,1),
            nn.AvgPool2d(8),
            View(10)
        )
    def forward(self,x): return self.m(x)


# Dataset 
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_set = datasets.CIFAR10("./data", train=False, transform=test_tf, download=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True, num_workers=0,pin_memory=True)

mean_t = torch.tensor(mean).view(3,1,1).to(device)
std_t  = torch.tensor(std).view(3,1,1).to(device)

def denormalize(tensor):  
    return (tensor * std_t) + mean_t

def normalize_from_uint8(img_uint8): 
    img = img_uint8.float() / 255.0
    return (img - mean_t) / std_t

# Load model 

model = allcnn_t().to(device)
ckpt_path = "checkpoints/allcnn_best.pth"
if os.path.exists(ckpt_path):
    print("Loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')
    # state is likely a state_dict
    try:
        model.load_state_dict(state)
    except Exception:
        # handle if saved dict was {k: v.cpu()} earlier
        model.load_state_dict({k: v.to(device) for k, v in state.items()})

model.eval()


# -------------------------
# Part C: 1-step signed gradient attack on the batch
# -------------------------

@torch.no_grad()
def clamp_in_normalized_space(x_norm):
    x_unnorm = denormalize(x_norm)         # -> [0,1] (approx), shape [B,C,H,W]
    x_unnorm = torch.clamp(x_unnorm, 0.0, 1.0)
    return (x_unnorm - mean_t) / std_t

def accuracy_on_clean(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in tqdm(loader, desc="Evaluating clean accuracy"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def accuracy_on_fgsm1(model, loader, device, eps_pixels=8.0):
    model.eval()
    eps_unit = eps_pixels / 255.0
    eps_per_channel = (eps_unit / std_t).to(device)
    correct_adv, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()

    for x, y in tqdm(loader, desc="Running 1-step FGSM attack"):
        x, y = x.to(device), y.to(device)

        x_adv = x.detach().clone().requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        loss.backward()

        g = x_adv.grad.detach()
        x_adv = (x_adv + eps_per_channel * g.sign()).detach()
        x_adv = clamp_in_normalized_space(x_adv)

        with torch.no_grad():
            logits_adv = model(x_adv)
            pred_adv = logits_adv.argmax(1)
            correct_adv += (pred_adv == y).sum().item()
            total += y.numel()

    return correct_adv / total


# ----- run & report -----
clean_acc = accuracy_on_clean(model, test_loader, device)
fgsm1_acc = accuracy_on_fgsm1(model, test_loader, device, eps_pixels=8.0)

print(f"Clean accuracy: {clean_acc*100:.2f}%")
print(f"1-step FGSM (ε=8/255) accuracy: {fgsm1_acc*100:.2f}%")
print(f"Drop: {(clean_acc - fgsm1_acc)*100:.2f} pp")
