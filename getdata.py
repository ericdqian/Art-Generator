from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Loader:
	def __init__(self, params):
		kwargs = {'num_workers': params['dataloader_workers'], 'pin_memory': params['pin_memory']} if params['cuda'] else {}

		self.normalization =  params['normalize']

		transform_train = transforms.Compose([
			transforms.Resize((params['image_size'], params['image_size'])),
			transforms.ToTensor(),
			transforms.Normalize(self.normalization[0], self.normalization[1]),
		])

		self.un_norm = transforms.Compose([ 
			# transforms.ToTensor(),
			transforms.Normalize([ 0., 0., 0. ],[ 1/self.normalization[1][0], 1/self.normalization[1][1], 1/self.normalization[1][2] ]),
            transforms.Normalize([ -self.normalization[0][0], -self.normalization[0][1], -self.normalization[0][2] ], [ 1., 1., 1. ]),
            # transforms.ToPILImage()
            ])

		# Note, since we don't really do any data augmenatations for now, the train and val sets actually
		# 	use the same transforms. If we wanted to do different transforms, we could make another transform
		#	and read in another dataset from the same folder, but specify the different transform, then call it the val_set
		#	then we could still.

		self.train_set = datasets.ImageFolder(root = params['data_path'], transform = transform_train)

		num_train = len(self.train_set)
		indices = list(range(num_train))
		split = int(np.floor(params['valid_split'] * num_train))

		np.random.seed(0) # Temporarily here for consistency!
		np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)

		self.train_loader = torch.utils.data.DataLoader(
		    self.train_set, batch_size=params['batch_size'], sampler=train_sampler, drop_last=True,
		    **kwargs
		)
		self.valid_loader = torch.utils.data.DataLoader(
		    self.train_set, batch_size=params['batch_size'], sampler=valid_sampler,
		    **kwargs
		)

