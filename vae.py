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


class VAE(nn.Module):
	def __init__(self, encoder, decoder, latent_vector_size):
		super(VAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.latent_vector_size = latent_vector_size

		#encoder 
		self.input_size = input_size
		self.output_size = output_size
		self.stride1 = stride1
		self.stride2 = stride2
		self.hidden1_size = hidden1_size
		self.hidden2_size = hidden2_size

		self.conv1 = nn.Conv2d(3, conv1_output_size, kernel_size = conv1_kernel_size)
		self.conv2 = nn.Conv2d(conv1_output_size, conv2_output_size, kernel_size = conv2_kernel_size)

		self.fc1 = nn.Linear(hidden1_size, hidden2_size)
		self.fc2 = nn.Linear(hidden2_size, output_size)

		#decoder 
		self.input_size = input_size
		self.output_size = output_size

		self.deconv1 = nn.ConvTranspose2d(input_size, deconv1_output_size, kernel_size = deconv1_kernel_size)
		self.deconv2 = nn.ConvTranspose2d(deconv1_output_size, deconv2_output_size, kernel_size = deconv2_kernel_size)
		self.outputdeconv = nn.ConvTranspose2d(deconv2_output_size, output_size, kernel_size = outputdeconv_kernel_size)

	def encode(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), self.stride1))
		x = F.relu(F.max_pool2d(self.conv2(x), self.stride2))
		x = F.view(-1, hidden1_size)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

	def decode(self, x):
		x = F.relu(self.deconv1(x))
		x = F.relu(self.deconv2(x))
		x = F.relu(self.outputdeconv(x))
		return x


def processImage(image, x_size, y_size):
	im_processed = cv2.resize(image, (x_size, y_size))
	return im_processed

def getData(folder, save = False):
	data = []
	path = './wikiart/' + folder + '/'
	for file in os.listdir(path):
		im = cv2.imread(path + file)
		im_processed = processImage(im, image_x, image_y)
		if save:
			cv2.imwrite('./resized/' + folder + '/' + file, im_processed)
		data.append(np.array(im_processed))
	return np.array(data)









#INPUTS:
image_x = 1000
image_y = 1000
input_size = image_x * image_y
latent_vector_size = 50
conv1_output_size = 20
conv1_kernel_size = 10
stride1 = 2
conv2_output_size = 80
conv2_kernel_size = 20
stride2 = 2
hidden1_size = 200
hidden2_size = 200
deconv1_output_size = 50
deconv1_kernel_size = 20
deconv2_output_size = 50
deconv2_kernel_size = 20
learning_rate = 0.1
dataset = 'Pointillism'

getData(dataset)

