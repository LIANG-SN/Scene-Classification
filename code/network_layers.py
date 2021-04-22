import numpy as np
import scipy.ndimage
import os,time
import skimage

def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	feature = x # (3, 224, 224)
	weights_layers = vgg16_weights
	
	print("input", feature.shape)
	for w in weights_layers:
		if w[0] == "conv2d":
			feature = multichannel_conv2d(feature, w[1], w[2])
			# print("conv", feature.shape)
		elif w[0] == "relu":
			feature = relu(feature)
			# print("relu", feature.shape)
		elif w[0] == "maxpool2d":
			feature = max_pool2d(feature, w[1])
			# print("pool", feature.shape)
		elif w[0] == "linear":
			feature = linear(feature, w[1], w[2])
			# print("linear", feature.shape)
	return feature


def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	F = weight.shape[0] # num filters
	K = weight.shape[1] # num channels
	L = weight.shape[2] # filter size
	assert weight.shape[2] == weight.shape[3]
	assert weight.shape[0] == weight.shape[0]
	S = x.shape[1]
	new_feature = np.zeros((F, S, S))
	for f in range(F):
		# print(scipy.ndimage.convolve(feature, w[1][f]).shape)
		for k in range(K):
			new_feature[f] += scipy.ndimage.convolve(x[k], weight[f, k])
		new_feature[f] += bias[f] # add bias
	return new_feature

def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x, 0)

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	
	if x.ndim == 3:
		# new_feature = np.zeros((x.shape[0], x.shape[1]//2, x.shape[2]//2))
		# for c in range(x.shape[0]):
		# 	new_feature[c] = skimage.measure.block_reduce(x[c], (size,size), np.max)
		new_feature = skimage.measure.block_reduce(x, (1, size, size), np.max)
		x = new_feature
	else:
		x = skimage.measure.block_reduce(x, (size,size), np.max)
	return x

def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	if x.ndim > 1:
		x = x.flatten().T
	return W @ x + b

