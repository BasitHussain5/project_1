import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset
from torchvision import datasets, transforms


def get_cifar10_dataloader(root, train_batch_size=128, test_batch_size=128, seed=2222):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    # valid_dataset_origin = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)

    torch.manual_seed(seed)
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])
    # valid_dataset.dataset = valid_dataset_origin

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


