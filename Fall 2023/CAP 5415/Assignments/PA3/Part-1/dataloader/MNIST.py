import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_dataloaders(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data/', train=True, download=True,
                              transform=transform)
    test_dataset = datasets.MNIST('./data/', train=False,
                              transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4)
    
    return train_loader, test_loader