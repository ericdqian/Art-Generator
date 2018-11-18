from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

def normalize(filepath, image_size):
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root = filepath, transform = transform_train)
    max_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(max_size, 4096)) # batch_size can be any big number

    pop_mean = []
    pop_std = []
    for batch_idx, (data, _) in enumerate(dataloader):
        npimage = data.numpy()
        batch_mean = np.mean(npimage, axis=(0,2,3))
        batch_std0 = np.std(npimage, axis=(0,2,3))
        
        pop_mean.append(batch_mean)
        pop_std.append(batch_std0)
    #     pop_std1.append(batch_std1)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = list(np.array(pop_mean).mean(axis=0))
    pop_std = list(np.array(pop_std).mean(axis=0))
    return [pop_mean, pop_std]
    # pop_std1 = np.array(pop_std1).mean(axis=0)

print(normalize('./data/touse/', 100))