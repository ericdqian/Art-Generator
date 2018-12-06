import numpy as np
import cv2
import os


def processImage(image, x_size, y_size):
	im_processed = cv2.resize(image, (x_size, y_size))
	return im_processed

def getRawData(style, image_x, image_y, save = False):
	data = []
	path = os.path.join(os.path.dirname(__file__), style)
	for file in os.listdir(path):
		im = cv2.imread(os.path.join(path, file))
		im_processed = processImage(im, image_x, image_y)
		if save:
			cv2.imwrite('./data/resized/' + folder + '/' + file, im_processed)
		data.append(im_processed)
	return np.array(data)

def getProcessedData(folder):
	data = []
	path = './data/resized/' + folder + '/'
	for file in os.listdir(path):
		im = cv2.imread(path + file)
		data.append(np.array(im))
	return np.array(data)

def getAllData(save = False):
	path = './data/wikiart/'
	for dir in os.listdir(path):
		getRawData(dir)

styles = os.listdir('./data/wikiart')
for style in styles:
	try:
		os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'touse', 'test', style))
		os.makedirs(os.path.join(os.path.dirname(__file__), 'data', 'touse', 'train', style))
	except FileExistsError:
		pass

	path = os.path.join(os.path.dirname(__file__), 'data', 'wikiart', style)
	num = len(os.listdir(path))
	i = 0
	for file in os.listdir(path):
		# print(file)
		im = cv2.imread(os.path.join(path, file))
		im_processed = cv2.resize(im, (128,128))
		if i < (int(0.1 * num)):
			cv2.imwrite(os.path.join(os.path.dirname(__file__), 'data', 'touse', 'test', style, file), im_processed)
		else:
			cv2.imwrite(os.path.join(os.path.dirname(__file__), 'data', 'touse', 'train', style, file), im_processed)
		i += 1


# getAllData(save = True)