import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy
import skimage
import itertools
from scipy.spatial.distance import cdist

DEBUGGING = False

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	if DEBUGGING:
		i,image_path,vgg16 = args
		im = imageio.imread(image_path)
		im = preprocess_image(im)
		im = torch.from_numpy(im).unsqueeze(0)
		print(i, "start")
		fc7_out = vgg16(im)
		feature = fc7_out.detach().numpy()[-1]
		print(i, "finish", feature.shape)
		return feature
	else:
		i,image_path,vgg16_weights = args
		im = imageio.imread(image_path)
		im = preprocess_image(im)
		print(i, "start")
		feature = network_layers.extract_deep_feature(im, vgg16_weights)
		print(i, "finish", feature.shape)
		return feature

	


def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
	'''

	if image.ndim == 2:
		image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
		print(image.shape)
		# review this
	if image.shape[2] == 4:
		image = image[:, :, :3]

	image = skimage.transform.resize(image, (224, 224, 3))
	image = np.transpose(image, (2, 0, 1))
	mean = np.array([0.485,0.456,0.406])
	sd = np.array([0.229,0.224,0.225])
	# normalize
	# todo: improve this
	for i in range(3):
		image[i] = (image[i] - mean[i]) / sd[i]
	return image


def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle=True)
	T = train_data['image_names'].shape[0]
	paths = train_data['image_names']
	features = None
	labels = train_data['labels']
	
	# one processer, vgg version
	# vgg does not work after copy??
	if DEBUGGING:
		features = []
		for i in range(T):
			features.append(get_image_feature((i, '../data/'.strip() + paths[i][0], vgg16)))
		features = np.array(features)
		# np.savez('trained_system_deep_torch.npz', features=features, labels=labels)
	else:
		pool = multiprocessing.Pool(num_workers)
		results = []
		weight = util.get_VGG16_weights()
		for i in range(0, T):
			args = [(i, '../data/'.strip() + paths[i][0], weight)]
			results.append(pool.apply_async(get_image_feature, args))

		features = []
		for result in results:
			feature = result.get()
			features.append(feature)
		features = np.array(features)
		# np.savez('trained_system_deep.npz', features=features, labels=labels)


def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''
	trained_system = None
	if DEBUGGING:
		trained_system = np.load("trained_system_deep_torch.npz")
	else:
		trained_system = np.load("trained_system_deep.npz") # testing

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	T = test_data['image_names'].shape[0]
	# test
	T = T
	paths = test_data['image_names']
	features = trained_system['features']
	print(features.shape)
	labels = test_data['labels']
	C = np.zeros((8, 8))

	if DEBUGGING:
		for i in range(0, T):
			feature = get_image_feature((i, '../data/'.strip() + paths[i][0], vgg16))
			near_feature = np.argmax(distance_to_set(feature, features))
			print(near_feature, trained_system['labels'][near_feature])
			predict = trained_system['labels'][near_feature]
			C[labels[i], predict] += 1
	else:
		pool = multiprocessing.Pool(num_workers)
		results = []
		weight = util.get_VGG16_weights()
		for i in range(0, T):
			args = [(i, '../data/'.strip() + paths[i][0], weight)]
			results.append(pool.apply_async(get_image_feature, args))
		i = 0
		for result in results:
			feature = result.get()
			dist = distance_to_set(feature, features) # neg
			near_feature = np.argmax(dist)
			print(i, trained_system['labels'][near_feature])
			predict = trained_system['labels'][near_feature]
			C[labels[i], predict] += 1
			i += 1

	accuracy = np.trace(C) / np.sum(C)
	return C, accuracy


def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''

	return -cdist(np.expand_dims(feature,0), train_features, 'euclidean')