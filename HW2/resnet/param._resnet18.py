import torch
import torchvision.models as models

model = models.resnet18()

total_params = 0
print(f"{'Layer':40s} {'Parameters':>15s}")
print("-" * 60)
for name, param in model.named_parameters():
    if param.requires_grad:
        num_params = param.numel()
        total_params += num_params
        print(f"{name:40s} {num_params:15,d}")

print("-" * 60)
print(f"{'Total':40s} {total_params:15,d}")
