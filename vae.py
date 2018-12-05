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


		# self.params['hidden2_size'] = self.params['latent_vector_size']
		# self.params['hidden1_size'] = self.params['conv2_output_size']

		#encoder 
		self.conv1 = nn.Conv2d(3, self.params['conv1_output_size'], kernel_size = self.params['conv1_kernel_size'], 
			padding = self.params['conv1_kernel_size']//2, stride = 2).cuda()
		self.bn1c = nn.BatchNorm2d(self.params['conv1_output_size'])
		self.conv2 = nn.Conv2d(self.params['conv1_output_size'], self.params['conv2_output_size'], kernel_size = self.params['conv2_kernel_size'], 
			padding = self.params['conv2_kernel_size']//2, stride = 2)
		self.bn2c = nn.BatchNorm2d(self.params['conv2_output_size'])
		self.conv3 = nn.Conv2d(self.params['conv2_output_size'], self.params['conv3_output_size'], kernel_size = self.params['conv3_kernel_size'], 
			padding = self.params['conv3_kernel_size']//2, stride = 2)
		self.bn3c = nn.BatchNorm2d(self.params['conv3_output_size'])
		# self.conv4 = nn.Conv2d(self.params['conv3_output_size'], self.params['conv4_output_size'], kernel_size = self.params['conv4_kernel_size'], 
		# 	padding = self.params['conv4_kernel_size']//2, stride = 2)
		# self.bn4c = nn.BatchNorm2d(self.params['conv4_output_size'])
		# self.conv5 = nn.Conv2d(self.params['conv4_output_size'], self.params['conv5_output_size'], kernel_size = self.params['conv5_kernel_size'], 
		# 	padding = self.params['conv5_kernel_size']//2, stride = 2)
		# self.bn5c = nn.BatchNorm2d(self.params['conv5_output_size'])

		self.l1 = nn.Linear(16 * 16 * self.params['conv3_output_size'], 16 * 16 * self.params['conv3_output_size'])
		self.l1bn = nn.BatchNorm1d(16 * 16 * self.params['conv3_output_size'])

		self.efc1 = nn.Linear(16 * 16 * self.params['conv3_output_size'], self.params['latent_vector_size'])
		self.efc1_bn = nn.BatchNorm1d(self.params['latent_vector_size'])
		self.efc21 = nn.Linear(self.params['latent_vector_size'], self.params['latent_vector_size'])
		self.efc22 = nn.Linear(self.params['latent_vector_size'], self.params['latent_vector_size'])

		#decoder 
		# self.dfc1 = nn.Linear(self.params['latent_vector_size'], self.params['hidden1_size'])
		# self.dfc1_bn = nn.BatchNorm1d(self.params['hidden1_size'])
		# self.dfc2 = nn.Linear(self.params['hidden1_size'], 4 * 4 * self.params['conv5_output_size'])
		# self.dfc2_bn = nn.BatchNorm1d(4 * 4 * self.params['conv5_output_size'])

		# self.deconv0 = nn.ConvTranspose2d(self.params['conv5_output_size'], self.params['conv4_output_size'], kernel_size = self.params['deconv1_kernel_size'],
		# 	padding = 1, output_padding = 1, stride = 2)
		# self.bn0d = nn.BatchNorm2d(self.params['conv4_output_size'])
		# self.deconv1 = nn.ConvTranspose2d(self.params['conv4_output_size'], self.params['conv3_output_size'], kernel_size = self.params['deconv1_kernel_size'],
		# 	padding = 1, output_padding = 1, stride = 2)
		# self.bn1d = nn.BatchNorm2d(self.params['conv3_output_size'])
		# self.deconv2 = nn.ConvTranspose2d(self.params['conv3_output_size'], self.params['conv2_output_size'], kernel_size = self.params['deconv2_kernel_size'],
		# 	padding = 1, output_padding = 1, stride = 2)
		# self.bn2d = nn.BatchNorm2d(self.params['conv2_output_size'])
		# self.deconv3 = nn.ConvTranspose2d(self.params['conv2_output_size'], self.params['conv1_output_size'], kernel_size = self.params['deconv3_kernel_size'],
		# 	padding = 1, output_padding = 1, stride = 2)
		# self.bn3d = nn.BatchNorm2d(self.params['conv1_output_size'])

		# self.deconv4 = nn.ConvTranspose2d(self.params['conv1_output_size'], 3, kernel_size = self.params['deconv4_kernel_size'], 
		# 	padding = 1, output_padding = 1, stride = 2)

		self.deconv_linear = nn.Linear(self.params['latent_vector_size'], self.params['d'] * 16 * 4 * 4)
		self.deconv_linear_bn = nn.BatchNorm1d(self.params['d'] * 16 * 4 * 4)
		# self.deconv1 = nn.ConvTranspose2d(self.params['latent_vector_size'], 16 * self.params['d'], kernel_size = 4,
		# 	padding = 1, output_padding = 1, stride = 2)
		self.deconv2 = nn.ConvTranspose2d(16 * self.params ['d'], 8 * self.params['d'], kernel_size = 4,
			padding = 1, output_padding = 0, stride = 2)
		self.deconv2_bn = nn.BatchNorm2d(self.params['d'] * 8)
		self.deconv3 = nn.ConvTranspose2d(8 * self.params['d'], 4 * self.params['d'], kernel_size = 4,
			padding = 1, output_padding = 0, stride = 2)
		self.deconv3_bn = nn.BatchNorm2d(self.params['d'] * 4)
		self.deconv4 = nn.ConvTranspose2d(4 * self.params['d'], 2 * self.params['d'], kernel_size = 4,
			padding = 1, output_padding = 0, stride = 2)
		self.deconv4_bn = nn.BatchNorm2d(self.params['d'] * 2)
		self.deconv5 = nn.ConvTranspose2d(2 * self.params['d'], self.params['d'], kernel_size = 4,
			padding = 1, output_padding = 0, stride = 2)
		self.deconv5_bn = nn.BatchNorm2d(self.params['d'])
		self.deconv6 = nn.ConvTranspose2d(self.params['d'], 3, kernel_size = 4,
			padding = 1, output_padding = 0, stride = 2)

		self.relu = nn.LeakyReLU(0.05)

	def encode(self, x):
		# x = self.relu(self.bn1c(self.conv1(x)))
		# x = self.relu(self.bn2c(self.conv2(x)))
		# x = self.relu(self.bn3c(self.conv3(x)))
		# x = self.relu(self.bn4c(self.conv4(x)))
		# x = self.relu(self.bn5c(self.conv5(x))).view(-1, 4 * 4 * self.params['conv5_output_size'])
		# x = self.relu(self.efc1_bn(self.efc1(x)))
		# return self.efc21(x), self.efc22(x)

		x = self.relu(self.bn1c(self.conv1(x)))
		x = self.relu(self.bn2c(self.conv2(x)))
		x = self.relu(self.bn3c(self.conv3(x))).view(-1, 16 * 16 * self.params['conv3_output_size'])
		x = self.relu(self.efc1_bn(self.efc1(x)))
		return self.efc21(x), self.efc22(x)

	def reparamaterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def decode(self, z):
		# z = self.relu(self.dfc1_bn(self.dfc1(z)))
		# z = self.relu(self.dfc2_bn(self.dfc2(z))).view(-1, self.params['conv5_output_size'], 4, 4)
		# z = self.relu(self.bn0d(self.deconv0(z)))
		# z = self.relu(self.bn1d(self.deconv1(z)))
		# z = self.relu(self.bn2d(self.deconv2(z)))
		# z = self.relu(self.bn3d(self.deconv3(z)))

		# z = self.relu(self.deconv4(z))
		# z = z.view(-1, 3, 128, 128)
		# return z
		z.cuda()
		# z = z.view(-1, self.params['latent_vector_size'], 4, 4)
		# print(z.size())
		z = self.relu(self.deconv_linear_bn(self.deconv_linear(z)).view(-1, self.params['d'] * 16, 4, 4))
		# z = self.relu(self.deconv1(z))
		# print(z.size())
		z = self.relu(self.deconv2_bn(self.deconv2(z)))
		# print(z.size())
		z = self.relu(self.deconv3_bn(self.deconv3(z)))
		# print(z.size())
		z = self.relu(self.deconv4_bn(self.deconv4(z)))
		# print(z.size())
		z = self.relu(self.deconv5_bn(self.deconv5(z)))
		# print(z.size())
		z = self.relu(self.deconv6(z))
		z = z.view(-1, 3, 128, 128)
		return z

	def forward1(self, x):
		mu, logvar = self.encode(x)
		z = self.reparamaterize(mu, logvar)
		return self.decode(z), mu, logvar


