import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


def get_data_loaders(data_directory, config):
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    dataset = MNIST(
        data_directory,
        download=True,
        transform=transform
    )

    train_len = int(len(dataset) * config.train_split)
    val_len = len(dataset) - train_len
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_len, val_len]
    )

    train_iter = DataLoader(train_set, **config.data_loader.to_dict(), persistent_workers=True)
    val_iter = DataLoader(val_set, **config.data_loader.to_dict(), persistent_workers=True)

    return train_iter, val_iter
