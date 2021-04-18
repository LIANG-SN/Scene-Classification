import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage.io
import multiprocessing

def build_recognition_system_helper(path):
	''' 
		i : the i th processor
		task_per_process = Total images / num_workers
	'''
	t = 0
	dictionary = np.load("dictionary.npy")
	features_part = np.zeros((len(path), dictionary.shape[0] * (4 ** 3 - 1) // 3))
	for p in path:
		features_part[t] = get_image_feature(p, dictionary, 3, dictionary.shape[0], 0)
		t += 1
		print(t)
	return features_part

def build_recognition_system_helper_simple(path):
	''' 
		i : the i th processor
		task_per_process = Total images / num_workers
	'''
	t = 0
	dictionary = np.load("dictionary.npy")
	features_part = np.zeros((len(path), dictionary.shape[0]))
	for p in path:
		features_part[t] = get_image_feature(p, dictionary, 3, dictionary.shape[0], 1)
		t += 1
		print(t)
	return features_part
	

def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	simple = False

	train_data = np.load("../data/train_data.npz", allow_pickle=True)
	dictionary = np.load("dictionary.npy")
	T = train_data['image_names'].shape[0]
	
	# test
	# T = (int)(T / 18)

	# calculate the shape
	path = ('../data/'.strip() + str((train_data['image_names'][0]))[2:-2])
	
	feature0 = get_image_feature(path, dictionary, 3, dictionary.shape[0], simple)

	# calculate features with multiprocessing
	features = np.zeros((T, feature0.shape[0]))
	T_per_process = T // num_workers # improve this
	paths = []
	for i in range(num_workers):
		path = []
		for t in range(T_per_process):
			path.append('../data/'.strip() + str((train_data['image_names'][i * T_per_process + t]))[2:-2])
		paths.append(path)
	p = multiprocessing.Pool(processes=num_workers)
	if simple:
		features = p.map(build_recognition_system_helper_simple, [p for p in paths])
		features = np.array(features).reshape((T, dictionary.shape[0]))
	else:
		features = p.map(build_recognition_system_helper, [p for p in paths])
		features = np.array(features).reshape((T, dictionary.shape[0] * (4 ** 3 - 1) // 3))
	
	labels = train_data['labels']
	# save
	# if simple:
	# 	np.savez('trained_system_simple.npz', dictionary=dictionary, features=features, labels=labels)
	# else:
	# 	np.savez('trained_system.npz', dictionary=dictionary, features=features, labels=labels, SPM_layer_num=3)

def get_image_feature(file_path,dictionary,layer_num,K, simple=0):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K*...)
	'''
	assert K == dictionary.shape[0]
	im = skimage.io.imread(file_path)
	wordmap = visual_words.get_visual_words(im, dictionary)
	if simple:
		feature = get_feature_from_wordmap(wordmap, K)
	else:
		feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return feature

def evaluate_recognition_system_helper(paths):
	predicts = np.zeros(len(paths))
	trained_system = np.load("trained_system.npz")
	
	dictionary = np.load("dictionary.npy")
	features = trained_system['features']
	t = 0
	for path in paths:
		feature = get_image_feature(path, dictionary, 3, dictionary.shape[0])
		predicts[t] = np.argmax(distance_to_set(feature, features))
		t += 1
	return predicts

def evaluate_recognition_system_helper_simple(paths):
	predicts = np.zeros(len(paths))
	trained_system = np.load("trained_system_simple.npz")
	
	dictionary = np.load("dictionary.npy")
	features = trained_system['features']
	t = 0
	for path in paths:
		feature = get_image_feature(path, dictionary, 3, dictionary.shape[0], 1)
		predicts[t] = np.argmax(distance_to_set(feature, features))
		t += 1

	return predicts

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''
	simple = False

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	if simple:
		trained_system = np.load("trained_system_simple.npz")
	else:
		trained_system = np.load("trained_system.npz")
	
	dictionary = np.load("dictionary.npy")
	features = trained_system['features']
	C = np.zeros((8, 8))
	T = test_data['labels'].shape[0]
	T = T


	T_per_process = T // num_workers # improve this
	processes = []

	paths = []
	labels = np.zeros((num_workers, T_per_process))
	for i in range(num_workers):
		path = []
		for t in range(T_per_process):
			path.append('../data/'.strip() + str((test_data['image_names'][i * T_per_process + t]))[2:-2])
			labels[i, t] = test_data['labels'][i * T_per_process + t]
		paths.append(path)

	p = multiprocessing.Pool(processes=num_workers)

	if simple:
		predicts = p.map(evaluate_recognition_system_helper_simple, [p for p in paths])
		predicts = np.array(predicts)
	else:
	    predicts = p.map(evaluate_recognition_system_helper, [p for p in paths])
	    predicts = np.array(predicts)
	
	for i in range(num_workers):
		for t in range(T_per_process):
			C[(int)(labels[i, t]), trained_system['labels'][(int)(predicts[i, t])]] += 1

	print(C)
	accuracy = np.trace(C) / np.sum(C)
	print(accuracy)
	return C, accuracy



def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	# review this
	sim = np.sum(np.minimum(histograms, word_hist), axis=1)
	return sim



def get_feature_from_wordmap(wordmap, dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''

	# create histogram
	# hist = np.zeros(dict_size)
	# add value to it
	# to do: optimize this
	# for i in range(wordmap.shape[0]):
	# 	for j in range(wordmap.shape[1]):
	# 		hist[wordmap[i, j]] += 1
	hist = np.histogram(wordmap, bins=range(dict_size + 1))[0] # include rightmost edge
	assert hist.shape[0] == dict_size
	# L1 normalize
	hist = hist / np.sum(hist)
	return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K

	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	
	# ----- TODO -----
	layer_num -= 1
	# create histogram for last layer (2^l * 2^l, K)
	hist_last_layer = np.zeros(((4 ** layer_num), dict_size))

	# grid size for last layer
	grid_h = wordmap.shape[0] // (2 ** layer_num)
	grid_w = wordmap.shape[1] // (2 ** layer_num)
	for i in range(4 ** layer_num):
		row = i // (2 ** layer_num)
		col = i % (2 ** layer_num)
		hist_last_layer[i] = np.histogram( \
			wordmap[row * grid_h : (row + 1) * grid_h, col * grid_w : (col + 1) * grid_w], \
			bins=range(dict_size + 1))[0]
	hist_prev = hist_last_layer # use prev to calculate next conveniently
	# aggregate
	hist_all = hist_last_layer.reshape((4 ** layer_num * dict_size)) * (2 ** (-layer_num))
	# test
	for l in range(layer_num - 1, -1, -1):
		grid_h = wordmap.shape[0] // (2 ** l)
		grid_w = wordmap.shape[1] // (2 ** l)
		hist = np.zeros(((4 ** l), dict_size))
		for i in range(4 ** l):
			row = i // (2 ** l)
			col = i % (2 ** l)
			hist[i] = \
				hist_prev[row*(2**(l+1)) + col]	\
				+ hist_prev[row*(2**(l+1)) + col + 1] \
				+ hist_prev[(row + 1)*(2**(l+1)) + col]	\
				+ hist_prev[(row + 1)*(2**(l+1)) + col + 1]
		hist_prev = hist
		hist_all = np.concatenate((hist_all, \
			hist.reshape(4 ** l * dict_size) * (2 ** (l - layer_num - 1))))
	return hist_all / np.sum(hist_all)