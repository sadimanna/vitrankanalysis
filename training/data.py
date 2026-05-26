"""Dataset loading utilities."""
from __future__ import annotations

from typing import Dict, Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    name: str,
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    augmentation: Dict[str, object] | None = None,
) -> Tuple[DataLoader, DataLoader, int]:
    augmentation = augmentation or {}
    use_aug = bool(augmentation.get("enabled", False))
    resize_size = int(augmentation.get("resize", image_size))
    crop_size = int(augmentation.get("crop", image_size))
    normalize = bool(augmentation.get("normalize", False))
    mean = augmentation.get("mean", [0.485, 0.456, 0.406])
    std = augmentation.get("std", [0.229, 0.224, 0.225])
    randaugment = bool(augmentation.get("randaugment", False))
    ra_ops = int(augmentation.get("randaugment_ops", 2))
    ra_magnitude = int(augmentation.get("randaugment_magnitude", 9))

    train_transforms = [transforms.Resize(resize_size), transforms.RandomCrop(crop_size)]

    if use_aug and randaugment:
        train_transforms.append(transforms.RandAugment(num_ops=ra_ops, magnitude=ra_magnitude))
    train_transforms.append(transforms.ToTensor())
    if normalize:
        train_transforms.append(transforms.Normalize(mean=mean, std=std))
    train_transform = transforms.Compose(train_transforms)

    test_transforms = [transforms.Resize(resize_size), transforms.CenterCrop(crop_size), transforms.ToTensor()]
    if normalize:
        test_transforms.append(transforms.Normalize(mean=mean, std=std))
    test_transform = transforms.Compose(test_transforms)
    if name == "cifar10":
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        num_classes = 10
    elif name == "cifar100":
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)
        num_classes = 100
    elif name == "tinyimagenet":
        train = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
        test = datasets.ImageFolder(f"{data_dir}/val", transform=test_transform)
        num_classes = 200
    else:
        train = datasets.ImageFolder(f"{data_dir}/train", transform=train_transform)
        test = datasets.ImageFolder(f"{data_dir}/val", transform=test_transform)
        num_classes = len(train.classes)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader, num_classes
