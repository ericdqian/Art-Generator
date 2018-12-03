import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torchvision.utils as tvut
import argparse
import torch as nn
from torch.autograd import Variable

from vae import VAE
from getdata import Loader
from trainer import Trainer
from loss import Loss

torch.set_default_tensor_type('torch.cuda.FloatTensor')

vae_params = {'conv1_output_size' : 32, 'conv1_kernel_size' : 3, 'stride1c' : 2, 
			  'conv2_output_size' : 64, 'conv2_kernel_size' : 3, 'stride2c' : 2, 
			  'conv3_output_size' : 64, 'conv3_kernel_size' : 3, 'stride3c' : 2, 
			  'conv4_output_size' : 256, 'conv4_kernel_size' : 3, 'stride4c' : 2, 
			  'conv5_output_size' : 512, 'conv5_kernel_size' : 3, 'stride5c' : 2, 
			  'hidden1_size' : 500, 'hidden2_size' : 200, 
			  'deconv0_output_size' : 256, 'deconv1_kernel_size' : 3, 'stride1d' : 2,
			  'deconv1_output_size' : 128, 'deconv1_kernel_size' : 3, 'stride1d' : 2, 
			  'deconv2_output_size' : 64, 'deconv2_kernel_size' : 3, 'stride1d' : 2,
			  'deconv3_output_size' : 32, 'deconv3_kernel_size' : 3, 'stride1d' : 2, 
			  'deconv4_output_size' : 3, 'deconv4_kernel_size' : 3, 'stride2d' : 2, 
			  'latent_vector_size': 400, 'd' :128}
data_load_params = {'batch_size' : 5, 'shuffle' : True, 'cuda': True, 'dataloader_workers' : 1, 'pin_memory' : True}

data = Loader(data_load_params)
model = VAE(vae_params)
if torch.cuda.is_available():
	model.cuda()
	print('Using GPU')

def train(opts):
	training_params = {'num_epochs' : opts.epochs, 'learning_rate' : 0.1, 'weight_decay' : 0.3, 'learning_rate_decay' : 0.9999, 'cuda' : False, 
		'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/'}

	loss = Loss()

	trainer = Trainer(model, loss, data, training_params)

	# print(trainer)

	trainer.train(opts)

# CAUTION: Note that vae_params must be the same as the checkpoint... perhaps we can save this with the checkpoint for future
def gen_image(checkpoint_file):
	checkpoint = torch.load('./runs/checkpoints/checkpoint2.pth.tar')
	model.load_state_dict(checkpoint['state_dict'])

	# print(data.train_set[0][0])
	# print(data.train_set[0][0].size())
	batch1 = data.train_set[9][0].unsqueeze(0).cuda()
	# print(batch1.size())

	inp = Variable(batch1)
	tvut.save_image(batch1, "goal2.png")
	# print(inp)
	model.eval()
	mu, logvar = model.encode(batch1)
	lat = model.reparamaterize(mu, logvar)
	
	generated = model.decode(mu)

	print(mu, logvar, generated)
	tvut.save_image(generated, "generated_image2.png")
	
def create_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type = int, default = 26)
	parser.add_argument('--test_every', type = int, default = 5)
	return parser



if __name__ == "__main__":
	parser = create_parser()
	opts = parser.parse_args()
	
	train(opts)
	gen_image('./runs/checkpoints/checkpoint2.pth.tar')