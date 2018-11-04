import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Encoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Encoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

	def forward(self, x):
		x = F.relu()


class Decoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(Decoder, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

	def forward(self, x):

class VAE(nn.Module):
	def __init__(self, encoder, decoder, latent_vector_size):
		super(VAE, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.vector_size = latent_vector_size

	def 