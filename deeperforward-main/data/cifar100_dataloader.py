import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_cifar100_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)

    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=8
    )

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

    return train_loader, valid_loader, test_loader


