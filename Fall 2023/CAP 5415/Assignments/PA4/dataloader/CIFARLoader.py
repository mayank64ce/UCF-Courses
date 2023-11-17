import torch
from torchvision import datasets, transforms


def get_data_loaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders(1)

    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx, data.shape, target)
        break
