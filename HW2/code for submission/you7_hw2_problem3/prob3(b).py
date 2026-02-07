
import torch, torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

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
# Part A: compute input gradients dx for some correctly and incorrectly classified images
# -------------------------
# get one minibatch (100 samples)
xs, ys = next(iter(test_loader))
xs, ys = xs.to(device), ys.to(device)

# forward to get preds 
with torch.no_grad():
    logits = model(xs)
preds = logits.argmax(dim=1)
correct_mask = (preds == ys)
incorrect_mask = ~correct_mask
print(f"Batch size {xs.size(0)}: {correct_mask.sum().item()} correct, {incorrect_mask.sum().item()} incorrect")

def visualize_grads_for_indices(indices, title_prefix): 
    imgs = []
    grads = []
    titles = []
    for idx in indices:
        x = xs[idx:idx+1].detach().clone().requires_grad_(True)  # keep grad for input
        y = ys[idx:idx+1]
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        # zero grads for input
        if x.grad is not None:
            x.grad.zero_()
        # compute gradients wrt input
        loss.backward()
        dx = x.grad.detach().cpu().squeeze(0)  # shape [C,H,W]
        # Convert grad to visualization: absolute, per-channel -> combine to rgb by max or mean
        grad_img = dx.abs()
        # normalize gradient to [0,1] for visualization
        grad_vis = grad_img.numpy()
        grad_vis = grad_vis / (grad_vis.max() + 1e-12)
        # denormalize the original image for display (to [0,1])
        img_display = denormalize(x.detach().cpu().squeeze(0)).cpu().numpy()
        img_display = np.clip(img_display, 0.0, 1.0)
        imgs.append(np.transpose(img_display, (1,2,0)))
        grads.append(np.transpose(grad_vis, (1,2,0)))
        titles.append(f"{title_prefix} idx={int(idx)} label={int(y.item())} pred={int(logits.argmax(1).item())}")
    # plot
    n = len(imgs)
    if n==0:
        return
    fig, axs = plt.subplots(n, 2, figsize=(6,3*n))
    if n==1:
        axs = np.expand_dims(axs, 0)
    for i in range(n):
        axs[i,0].imshow(imgs[i])
        axs[i,0].axis('off')
        axs[i,0].set_title(titles[i] + " (image)")
        axs[i,1].imshow(grads[i])
        axs[i,1].axis('off')
        axs[i,1].set_title("abs(∂L/∂x)")
    plt.tight_layout()
    plt.show()

correct_indices = [i for i in range(xs.size(0)) if correct_mask[i].item()][:6]
incorrect_indices = [i for i in range(xs.size(0)) if incorrect_mask[i].item()][:6]

print("Visualizing gradients for correctly classified examples:", correct_indices)
visualize_grads_for_indices(correct_indices, "Correct")

print("Visualizing gradients for misclassified examples:", incorrect_indices)
visualize_grads_for_indices(incorrect_indices, "Incorrect")

# -------------------------
# Part B: 5-step signed gradient attack on the batch
# -------------------------
eps_pixels = 8.0                   
eps_unit = eps_pixels / 255.0      
eps_per_channel = (eps_unit / std_t).to(device)   

print("eps per channel in normalized space (approx):", eps_per_channel.squeeze().cpu().numpy())

num_steps = 5
batch_size = xs.size(0)
loss_fn = nn.CrossEntropyLoss(reduction='none')

x_adv = xs.detach().clone()   # already normalized
x_adv.requires_grad = True


def clamp_in_normalized_space(x_norm):
    x_unnorm = denormalize(x_norm)         
    x_unnorm = torch.clamp(x_unnorm, 0.0, 1.0)
    return (x_unnorm - mean_t) / std_t

# arrays to record average loss per step
avg_loss_per_step = []
# optionally also record fraction of images whose prediction changed
acc_per_step = []


for step in range(num_steps):
    # ensure gradients from previous step are cleared
    if x_adv.grad is not None:
        x_adv.grad.zero_()
    # forward
    logits_adv = model(x_adv)
    losses = loss_fn(logits_adv, ys)   # per-sample losses
    loss_mean = losses.mean()
    # backward to compute ∂L/∂x for the batch; here we use mean loss to aggregate
    loss_mean.backward()
    dx = x_adv.grad.detach()   # shape [B,C,H,W]
    # sign of gradient
    dx_sign = dx.sign()
    x_adv = (x_adv + eps_per_channel * dx_sign).detach()
    x_adv = clamp_in_normalized_space(x_adv)
    x_adv.requires_grad = True

    # record average loss after this perturbation (use model in eval)
    with torch.no_grad():
        logits_now = model(x_adv)
        losses_now = loss_fn(logits_now, ys)
        avg_loss = losses_now.mean().item()
        avg_loss_per_step.append(avg_loss)
        acc_now = (logits_now.argmax(dim=1) == ys).float().mean().item()
        acc_per_step.append(acc_now)
    print(f"Step {step+1}/{num_steps}: avg loss = {avg_loss:.4f}, accuracy = {acc_now*100:.2f}%")

# Plot average loss vs step
plt.figure(figsize=(6,4))
plt.plot(range(1,num_steps+1), avg_loss_per_step, marker='o')
plt.xlabel("Attack step")
plt.ylabel("Average loss on perturbed images")
plt.title(f"5-step signed-gradient attack (eps_pixels={eps_pixels})")
plt.grid(True)
plt.show()

# Plot accuracy vs step
plt.figure(figsize=(6,4))
plt.plot(range(1,num_steps+1), [1 - a for a in acc_per_step], marker='o')  # plot error %
plt.xlabel("Attack step")
plt.ylabel("Error rate on perturbed images")
plt.title(f"Error rate vs attack step (eps_pixels={eps_pixels})")
plt.grid(True)
plt.show()

# -------------------------
# Visualize a few adversarial examples and their perturbations (difference)
# -------------------------
# pick first 6 images
kvis = min(6, batch_size)
x_orig = xs[:kvis].cpu()
x_final = x_adv.detach().cpu()[:kvis]
# compute perturbation in pixel-space for visualization
pert_pixel = (denormalize(x_final) - denormalize(x_orig)).numpy()  # in [ -1, 1 ] approx
# normalize to displayable magnitude
def to_display(img_tensor):
    img = img_tensor.numpy()
    img = np.transpose(img, (1,2,0))  # H W C
    img = np.clip(img, 0.0, 1.0)
    return img

fig, axs = plt.subplots(kvis, 3, figsize=(9,3*kvis))
for i in range(kvis):
    axs[i,0].imshow(to_display(denormalize(x_orig[i]).cpu()))
    axs[i,0].axis('off'); axs[i,0].set_title("orig")
    axs[i,1].imshow(to_display(denormalize(x_final[i]).cpu()))
    axs[i,1].axis('off'); axs[i,1].set_title("adv final")
    # perturbation: scale for visibility
    p = pert_pixel[i]
    p_vis = (p - p.min()) / (p.max()-p.min()+1e-12)
    axs[i,2].imshow(np.transpose(p_vis, (1,2,0)))
    axs[i,2].axis('off'); axs[i,2].set_title("pert (scaled)")
plt.tight_layout(); plt.show()
