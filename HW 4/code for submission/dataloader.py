from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

def build_mnist_10k_dataset(root="./data"):
    # 1. 28*28->14*14
    binarize_thresh = 128.0 / 255.0
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > binarize_thresh).float())
    ])

    # 2. Minist
    full_train = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    # 3. 1000 images for each number
    indices_per_class = {k: [] for k in range(10)}
    for idx, (_, label) in enumerate(full_train):
        label = int(label)
        if len(indices_per_class[label]) < 1000:
            indices_per_class[label].append(idx)
        if all(len(v) == 1000 for v in indices_per_class.values()):
            break

    selected_indices = []
    for k in range(10):
        selected_indices.extend(indices_per_class[k])

    print(f"Total selected samples: {len(selected_indices)}")  
    mnist_10k = Subset(full_train, selected_indices)
    return mnist_10k



def build_mnist_val_dataset(root="./data"):
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ])
    mnist_val = datasets.MNIST(root=root, train=False, download=True,
                               transform=transform)
    return mnist_val