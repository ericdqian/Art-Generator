import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data

from vae import VAE
from getdata import Loader
from trainer import Trainer
from loss import Loss

def main():
	data_load_params = {'batch_size' : 10, 'shuffle' : True, 'cuda': False, 'dataloader_workers' : 1, 'pin_memory' : True}
	vae_params = {'conv1_output_size' : 32, 'conv1_kernel_size' : 3, 'stride1c' : 2, 'conv2_output_size' : 64, 'conv2_kernel_size' : 3, 'stride2c' : 2, 
		'hidden1_size' : 200, 'hidden2_size' : 200, 'deconv1_output_size' : 64, 'deconv1_kernel_size' : 3, 'stride1d' : 2, 'deconv2_output_size' : 32, 
		'deconv2_kernel_size' : 3, 'stride2d' : 2, 'latent_vector_size': 500}
	training_params = {'num_epochs' : 10, 'learning_rate' : 0.1, 'weight_decay' : 0.1, 'learning_rate_decay' : 0.99, 'cuda' : False, 
		'summary_dir' : './runs/logs/', 'checkpoint_dir' : './runs/checkpoints/'}

	data = Loader(data_load_params)
	model = VAE(vae_params)
	loss = Loss()

	trainer = Trainer(model, loss, data.train_loader, data.test_loader, training_params)

	print(trainer)

	trainer.train()
	


if __name__ == "__main__":
	main()