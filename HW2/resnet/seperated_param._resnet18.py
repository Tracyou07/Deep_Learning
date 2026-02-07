import torch
import torchvision.models as models
import torch.nn as nn

model = models.resnet18()


bn_params, bias_params, other_params = [], [], []

for name, param in model.named_parameters():
    if isinstance(dict(model.named_modules()).get(name.split('.')[0], None), nn.BatchNorm2d) or 'bn' in name or 'downsample.1' in name:
        bn_params.append((name, param.shape))
    elif 'bias' in name and param.requires_grad:
        bias_params.append((name, param.shape))
    else:
        other_params.append((name, param.shape))

print("\n(i) BatchNorm affine parameters:")
for n, s in bn_params:
    print(f"  {n:<40} {tuple(s)}")

print("\n(ii) Bias parameters of Conv/FC layers:")
for n, s in bias_params:
    print(f"  {n:<40} {tuple(s)}")

print("\n(iii) Other parameters (Conv/FC weights, etc.):")
for n, s in other_params:
    print(f"  {n:<40} {tuple(s)}")

print("\nSummary:")
print(f"  Total BN params: {len(bn_params)}")
print(f"  Total bias params: {len(bias_params)}")
print(f"  Total other params: {len(other_params)}")
