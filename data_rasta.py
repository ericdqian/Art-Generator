import os
import random
import math
import shutil

if __name__ == "__main__":
	print("hi")
	DATA_DIR = 'data/wikiart'
	DEST_DIR = 'data/wikiart_rasta'
	TRAIN_VAL_TEST_RATIO = (0.8, 0.1, 0.1)
	for style in os.listdir(DATA_DIR):
		print(style)
		if os.path.isdir(os.path.join(DATA_DIR, style)):
			imgs = [img for img in os.listdir(os.path.join(DATA_DIR, style))]
			random.shuffle(imgs)
			val_idx = math.floor(TRAIN_VAL_TEST_RATIO[0] * len(imgs))
			test_idx = math.floor(TRAIN_VAL_TEST_RATIO[1] * len(imgs) + val_idx)
			print(len(imgs))
			for idx in range(len(imgs)):
				if idx < val_idx:
					dataset = 'train'
				elif idx < test_idx:
					dataset = 'val'
				else:
					dataset = 'test'	
				if not os.path.exists(os.path.join(DEST_DIR, dataset, style)):
					os.makedirs(os.path.join(DEST_DIR, dataset, style))
				shutil.copy(os.path.join(DATA_DIR, style, imgs[idx]), os.path.join(DEST_DIR, dataset, style, imgs[idx]))
