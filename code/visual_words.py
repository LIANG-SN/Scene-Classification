import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.	
	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	# F = 20 ?
	h = image.shape[0]
	w = image.shape[1]
	# check color channel
	if image.ndim == 2:
		image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
		print(image.shape)
		# review this
	if image.shape[2] == 4:
		image = image[:, :, :3]
	# check rgb value range
	if image.max() > 1.0:
		image = image.astype('float') / 255
	# convert rgb to lab
	image = skimage.color.rgb2lab(image)
	# filter
	scales = np.array([1, 2, 4, 8, 8 * np.sqrt(2)])
	output = np.zeros((h, w, 3 * 20))
	i = 0
	for scale in scales:
		output[:, :, i * 3]     = scipy.ndimage.gaussian_filter(image[:, :, 0], scale)
		output[:, :, i * 3 + 1] = scipy.ndimage.gaussian_filter(image[:, :, 1], scale)
		output[:, :, i * 3 + 2] = scipy.ndimage.gaussian_filter(image[:, :, 2], scale)
		i += 1
		output[:, :, i * 3]     = scipy.ndimage.gaussian_laplace(image[:, :, 0], scale)
		output[:, :, i * 3 + 1] = scipy.ndimage.gaussian_laplace(image[:, :, 1], scale)
		output[:, :, i * 3 + 2] = scipy.ndimage.gaussian_laplace(image[:, :, 2], scale)
		i += 1
		output[:, :, i * 3]     = scipy.ndimage.gaussian_filter(image[:, :, 0], scale, (0, 1))
		output[:, :, i * 3 + 1] = scipy.ndimage.gaussian_filter(image[:, :, 1], scale, (0, 1))
		output[:, :, i * 3 + 2] = scipy.ndimage.gaussian_filter(image[:, :, 2], scale, (0, 1))
		i += 1
		output[:, :, i * 3]     = scipy.ndimage.gaussian_filter(image[:, :, 0], scale, (1, 0))
		output[:, :, i * 3 + 1] = scipy.ndimage.gaussian_filter(image[:, :, 1], scale, (1, 0))
		output[:, :, i * 3 + 2] = scipy.ndimage.gaussian_filter(image[:, :, 2], scale, (1, 0))
		i += 1
	return output
def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''

	filtered = extract_filter_responses(image)
	h, w, f = filtered.shape
	filtered = filtered.reshape(h*w, f)
	dist_matrix = scipy.spatial.distance.cdist(filtered, dictionary, 'euclidean')
	assert dist_matrix.shape == (h*w, dictionary.shape[0])
	wordmap = np.zeros(h*w)
	wordmap[:] = dist_matrix[:].argmin(axis=-1)
	wordmap = wordmap.reshape(h, w)
	return wordmap



def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''


	i,alpha,image_path = args

	im = imageio.imread(image_path)
	h = im.shape[0]
	w = im.shape[1]
	samplesX = random.sample(range(0, w), alpha)
	samplesY = random.sample(range(0, h), alpha)
	sampled_response = np.zeros((alpha, 60))
	sampled_response[:] = extract_filter_responses(im)[samplesY[:], samplesX[:]]
	fileName = '../temporary_data/temp'.strip() + str(i).strip()
	np.save(fileName, sampled_response)

def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle=True)

	T = train_data['image_names'].shape[0]
	alpha = 50
	K = 300
	# compute filter responses

	for i in range(T):
		# improve this
		path = ('../data/'.strip() + str((train_data['image_names'][i]))[2:-2])
		compute_dictionary_one_image((i, alpha, path))
		print('compute' + str(i))
	
	# load all filter responese
	filter_responses = np.zeros((alpha * T, 60))
	for i in range(T):
		fileName = '../temporary_data/temp'.strip() + str(i).strip() + '.npy'.strip()
		filter_responses[i * alpha : i * alpha + alpha] = np.load(fileName)
		print('load' + str(i))

	kmeans = sklearn.cluster.KMeans(n_clusters=K, n_jobs=num_workers).fit(filter_responses)
	dictionary = kmeans.cluster_centers_
	np.save('dictionary.npy', dictionary)
