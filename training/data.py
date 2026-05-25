"""Dataset loading utilities."""
from __future__ import annotations

from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    name: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    if name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    elif name == "cifar100":
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        num_classes = 100
    elif name == "tinyimagenet":
        train = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
        test = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
        num_classes = 200
    else:
        train = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
        test = datasets.ImageFolder(f"{data_dir}/val", transform=transform)
        num_classes = len(train.classes)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader, num_classes
