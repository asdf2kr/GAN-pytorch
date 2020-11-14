import os
import torch
import torchvision
from torchvision import transforms
class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def prepare_dataloaders(args):
    """ Access the data in the dataset. """
    if args.dataset == 'mnist': # No1. MNIST dataset
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], # 1 for grayscale channels
                                std=[0.5])
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], # 1 for grayscale channels
                                std=[0.5])
        ])

        train_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=True,
            download=True,
            transform=train_transform
        )
        valid_dataset = torchvision.datasets.MNIST(
            root=args.data_path,
            train=False,
            download=True,
            transform=valid_transform
        )
    else:
        error("[Info] We will support {} dataset".format(args.datset))
    """
    Loading the data.
    Now that we have access to the dataset, we pass it through torch.utils.data.DataLoader.
    The DataLoader combines the dataset and sampler, returning an iterable over the dataset.
    """
    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.workers,
                    shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.workers
    )

    return train_loader, valid_loader, len(train_dataset), len(valid_dataset)
