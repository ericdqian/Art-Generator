from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Loader:
	def __init__(self, params):
		kwargs = {'num_workers': params['dataloader_workers'], 'pin_memory': params['pin_memory']} if params['cuda'] else {}

		transform_train = transforms.Compose([
			transforms.ToTensor()
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		transform_test = transforms.Compose([
			transforms.ToTensor()
			# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])

		train_set = datasets.ImageFolder(root = './data/touse/', transform = transform_train)
		self.train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=params['shuffle'],
									   **kwargs)

		test_set = datasets.ImageFolder(root = './data/touse/', transform = transform_test)
		self.test_loader = DataLoader(test_set, batch_size=params['batch_size'], shuffle=False,
									  **kwargs)

