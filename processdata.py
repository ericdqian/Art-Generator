import numpy as np
import cv2
import os


def processImage(image, x_size, y_size):
	im_processed = cv2.resize(image, (x_size, y_size))
	return im_processed

def getRawData(folder, save = False):
	data = []
	path = './data/wikiart/' + folder + '/'
	for file in os.listdir(path):
		im = cv2.imread(path + file)
		im_processed = processImage(im, image_x, image_y)
		if save:
			cv2.imwrite('./data/resized/' + folder + '/' + file, im_processed)
		data.append(np.array(im_processed))
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


getAllData(save = True)