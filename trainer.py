from torch.autograd import Variable
from torch import optim
import torch

from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm
import numpy as np
import torch as nn

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Trainer:
	def __init__(self, model, loss, data, params):
		self.model = model
		self.params = params
		self.params['start_epoch'] = 0

		self.data = data
		self.train_loader = data.train_loader
		self.test_loader = data.test_loader

		self.loss = loss
		self.optimizer = self.get_optimizer()
		# print()
		self.summary_writer = SummaryWriter(log_dir=self.params['summary_dir'])
		print(next(model.parameters()).is_cuda)

	def train(self, opts):
		self.model.train()
		kwargs = {}
		for epoch in range(self.params['start_epoch'], self.params['num_epochs']):
			loss_list = []
			print("epoch {}...".format(epoch))
			for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
				# if nn.cuda.is_available():
				# 	data = data.cuda()
				data = Variable(data)
				self.optimizer.zero_grad()
				recon_batch, mu, logvar = self.model.forward1(data)
				loss = self.loss(recon_batch, data, mu, logvar)
				loss.backward()
				self.optimizer.step()
				loss_list.append(loss.data[0].item())

			print("epoch {}: - loss: {}".format(epoch, np.mean(loss_list)))
			new_lr = self.adjust_learning_rate(epoch)
			print('learning rate:', new_lr)

			self.summary_writer.add_scalar('training/loss', np.mean(loss_list), epoch)
			self.summary_writer.add_scalar('training/learning_rate', new_lr, epoch)
			self.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.state_dict(),
				'optimizer': self.optimizer.state_dict(),
			})
			if epoch % opts.test_every == 0:
				self.print_image("training/epoch"+str(epoch))

	def print_image(self, name, rand=False):
		batch1 = self.data.train_set[0][0].unsqueeze(0)
		inp = Variable(batch1)
		self.model.eval()
		mu, logvar = self.model.encode(batch1)
		normalized_version = self.model.decode(mu)
		tvut.save_image(self.data.un_norm(normalized_version[0]), name+".png")
		self.model.train() # Set the model back in training mode

	def get_optimizer(self):
		return optim.Adam(self.model.parameters(), lr=self.params['learning_rate'],
						  weight_decay=self.params['weight_decay'])

	def adjust_learning_rate(self, epoch):
		"""Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
		learning_rate = self.params['learning_rate'] * (self.params['learning_rate_decay'] ** epoch)
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = learning_rate
		return learning_rate

	def save_checkpoint(self, state, is_best=False, filename='checkpoint2.pth.tar'):
		'''
		a function to save checkpoint of the training
		:param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
							'optimizer': self.optimizer.state_dict()}
		:param is_best: boolean to save the checkpoint aside if it has the best score so far
		:param filename: the name of the saved file
		'''
		torch.save(state, self.params['checkpoint_dir'] + filename)
		# if is_best:
		#     shutil.copyfile(self.args.checkpoint_dir + filename,
		#                     self.args.checkpoint_dir + 'model_best.pth.tar')