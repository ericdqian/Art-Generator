import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data
import torchvision.utils as tvut
from torch.autograd import Variable

from vae import VAE
from getdata import Loader
from trainer import Trainer
from loss import Loss

image_size = 128
vae_params = {'conv1_output_size' : 32, 'conv1_kernel_size' : 3, 'stride1c' : 2, 
			  'conv2_output_size' : 64, 'conv2_kernel_size' : 3, 'stride2c' : 2, 
			  'conv3_output_size' : 128, 'conv3_kernel_size' : 3, 'stride3c' : 2, 
			  'conv4_output_size' : 256, 'conv4_kernel_size' : 3, 'stride4c' : 2, 
			  'hidden1_size' : 200, 'hidden2_size' : 200, 
			  'deconv1_output_size' : 128, 'deconv1_kernel_size' : 3, 'stride1d' : 2, 
			  'deconv2_output_size' : 64, 'deconv2_kernel_size' : 3, 'stride1d' : 2,
			  'deconv3_output_size' : 32, 'deconv3_kernel_size' : 3, 'stride1d' : 2, 
			  'deconv4_output_size' : 3, 'deconv4_kernel_size' : 3, 'stride2d' : 2, 

			  'latent_vector_size': 500, 'image_size': image_size}
data_load_params = {'batch_size' : 10, 'shuffle' : True, 'cuda': False, 'dataloader_workers' : 1, 'pin_memory' : True,
					'image_size': image_size, 'valid_split': 0.2, 'data_path': './data/touse/',
					'normalize': [[0.57867116, 0.53378844, 0.47493997], [0.2312678, 0.21923426, 0.22560729]]}

data = Loader(data_load_params)
model = VAE(vae_params)

def train():
	training_params = {'num_epochs' : 10, 'learning_rate' : 0.1, 'weight_decay' : 0.1, 'learning_rate_decay' : 0.9999, 'cuda' : False, 
		'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/'}

	loss = Loss()

	trainer = Trainer(model, loss, data, training_params)

	print(trainer)

	trainer.train()

# CAUTION: Note that vae_params must be the same as the checkpoint... perhaps we can save this with the checkpoint for future
def gen_image(checkpoint_file):
	checkpoint = torch.load('./runs/checkpoints/checkpoint.pth.tar')
	model.load_state_dict(checkpoint['state_dict'])
	print(len(data.train_set))
	# print(data.train_set[0][0])
	print(data.train_set[0][0].size())
	batch1 = data.train_set[0][0].unsqueeze(0)
	print(batch1.size())
	print(batch1)
	to_save = data.un_norm(data.train_set[0][0])
	tvut.save_image(to_save, "goal1.png")
	model.eval()
	mu, logvar = model.encode(batch1)
	lat = model.reparamaterize(mu, logvar)
	print(mu)
	# tvut.save_image(data.un_norm(model.decode(mu)[0]), "generated_image1.png")
	


if __name__ == "__main__":
	# train()
	gen_image('./runs/checkpoints/checkpoint.pth.tar')
