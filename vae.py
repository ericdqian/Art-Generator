import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os

from PIL import Image
from glob import glob
from torchvision import transforms, utils
from torch.autograd import Variable


class VAE(nn.Module):
	def __init__(self, params):
		super(VAE, self).__init__()
		self.params = params
		# print(self.params)

		#encoder 
		self.conv1 = nn.Conv2d(3, self.params['conv1_output_size'], kernel_size = self.params['conv1_kernel_size'], 
			padding = self.params['conv1_kernel_size']//2, stride = 2)
		self.bn1c = nn.BatchNorm2d(self.params['conv1_output_size'])
		self.conv2 = nn.Conv2d(self.params['conv1_output_size'], self.params['conv2_output_size'], kernel_size = self.params['conv2_kernel_size'], 
			padding = self.params['conv2_kernel_size']//2, stride = 2)
		self.bn2c = nn.BatchNorm2d(self.params['conv2_output_size'])

		# self.efc1 = nn.Linear(25 * 25 * self.params['conv2_output_size'], self.params['hidden2_size'])
		# self.efc1_bn = nn.BatchNorm1d(self.params['hidden2_size'])
		# self.efc21 = nn.Linear(self.params['hidden2_size'], self.params['latent_vector_size'])
		# self.efc22 = nn.Linear(self.params['hidden2_size'], self.params['latent_vector_size'])

		# #decoder 
		# self.dfc1 = nn.Linear(self.params['latent_vector_size'], self.params['hidden2_size'])
		# self.dfc1_bn = nn.BatchNorm1d(self.params['hidden2_size'])
		# self.dfc2 = nn.Linear(self.params['hidden2_size'], self.params['hidden1_size'])
		# self.dfc2_bn = nn.BatchNorm1d(self.params['hidden1_size'])


		self.params['hidden2_size'] = self.params['latent_vector_size']
		self.params['hidden1_size'] = self.params['conv2_output_size']

		self.efc1 = nn.Linear(25 * 25 * self.params['conv2_output_size'], self.params['hidden2_size'])
		self.efc1_bn = nn.BatchNorm1d(self.params['hidden2_size'])
		self.efc21 = nn.Linear(self.params['hidden2_size'], self.params['latent_vector_size'])
		self.efc22 = nn.Linear(self.params['hidden2_size'], self.params['latent_vector_size'])

		#decoder 
		self.dfc1 = nn.Linear(self.params['latent_vector_size'], self.params['hidden2_size'])
		self.dfc1_bn = nn.BatchNorm1d(self.params['hidden2_size'])
		self.dfc2 = nn.Linear(self.params['hidden2_size'], 25 * 25 * self.params['hidden1_size'])
		self.dfc2_bn = nn.BatchNorm1d(25 * 25 * self.params['hidden1_size'])

		self.deconv1 = nn.ConvTranspose2d(self.params['hidden1_size'], self.params['conv2_output_size'], kernel_size = self.params['deconv1_kernel_size'],
			padding = 1, output_padding = 1, stride = 2)
		self.bn1d = nn.BatchNorm2d(self.params['conv2_output_size'])
		self.deconv2 = nn.ConvTranspose2d(self.params['conv2_output_size'], 3, kernel_size = self.params['deconv2_kernel_size'], 
			padding = 1, output_padding = 1, stride = 2)

		self.relu = nn.ReLU(0.1)

	def encode(self, x):
		# print(x.size())
		x = self.relu(self.bn1c(self.conv1(x)))
		# print(x.size())
		x = self.relu(self.bn2c(self.conv2(x))).view(-1, 25 * 25 * self.params['conv2_output_size'])
		# print(x.size())
		x = self.relu(self.efc1_bn(self.efc1(x)))
		# print(x.size())
		return self.efc21(x), self.efc22(x)

	def reparamaterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def decode(self, z):
		# print('decoding')
		# print(z.size())
		z = self.relu(self.dfc1_bn(self.dfc1(z)))
		# print(z.size())
		z = self.relu(self.dfc2_bn(self.dfc2(z))).view(-1, self.params['conv2_output_size'], 25, 25)
		# print(z.size())
		z = self.relu(self.bn1d(self.deconv1(z)))
		# print(z.size())
		z = self.relu(self.deconv2(z))
		# print(z.size())
		z = z.view(-1, 3, 100, 100)
		return z

	def forward1(self, x):
		mu, logvar = self.encode(x)
		z = self.reparamaterize(mu, logvar)
		return self.decode(z), mu, logvar


