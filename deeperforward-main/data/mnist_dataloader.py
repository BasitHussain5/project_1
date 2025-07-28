import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_mnist_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222, valid=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),  # 归一化
        transforms.Resize((32, 32))
    ])

    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    valid_dataset = None

    if valid:
        torch.manual_seed(seed)
        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8
    )

    valid_loader = None
    if valid:
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=train_batch_size,
            shuffle=False,
            num_workers=8
        )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=8
    )

    if valid:
        return train_loader, valid_loader, test_loader
    return train_loader, test_loader

