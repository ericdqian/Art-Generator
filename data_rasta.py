import os
import random
import math
import shutil
import argparse

DATA_DIR = 'data/wikiart'
DEST_DIR = 'data/wikiart_rasta'
TRAIN_VAL_TEST_RATIO = (0.8, 0.1, 0.1)

def copy_files_into_split(data_dir=DATA_DIR, dest_dir=DEST_DIR, ratio=TRAIN_VAL_TEST_RATIO, styles=None):
	for style in os.listdir(DATA_DIR):
		print(style)
		if os.path.isdir(os.path.join(DATA_DIR, style)) and is_valid_style(style, styles):
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

def is_valid_style(style, styles):
	return styles is None or style in styles

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--styles', action="store", nargs='+', default=None, dest='styles', help='Valid styles')
	parser.add_argument('--dest', action="store", default=DEST_DIR, dest='dest', help='Dest. dir for the copy')

	args = parser.parse_args()

	# print(type(args.styles))
	# print(args.styles)
	copy_files_into_split(dest_dir=args.dest, styles=args.styles)
