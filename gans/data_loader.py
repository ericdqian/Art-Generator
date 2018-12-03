import os

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms


def load_data(data_type, opts):
    """Creates training and test data loaders.
    """
    transform = transforms.Compose([
                    transforms.Resize((opts.image_size, opts.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #fix normalization
                ])

    train_path = os.path.join('data', 'train_{}'.format(data_type))
    test_path = os.path.join('data', 'test_{}'.format(data_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return train_dloader, test_dloader
