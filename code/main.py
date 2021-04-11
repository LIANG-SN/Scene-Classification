import numpy as np
import numpy.matlib as npm
# import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
# import visual_recog
# import deep_recog
import skimage.io

if __name__ == '__main__':
   
	num_cores = util.get_num_CPU()

	# 1.1
	path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
	# path_img = "../data/kitchen/sun_agrkvwbkgrglcoga.jpg"
	# path_img = "../data/kitchen/sun_aaebjpeispxohmfv.jpg"
	
	
	image = skimage.io.imread(path_img)
	# image = image.astype('float') / 255
	# filter_responses = visual_words.extract_filter_responses(image)
	# util.display_filter_responses(filter_responses)

	# 1.2
	# visual_words.compute_dictionary(num_workers=num_cores)
	# data = np.load('dictionary.npy')
	# print(data.shape)
	
	# 1.3 test wordmaps
	plt.subplot(2, 1, 1)
	plt.imshow(image)
	dictionary = np.load('dictionary.npy')
	wordmap = visual_words.get_visual_words(image,dictionary)
	plt.subplot(2, 1, 2)
	plt.imshow(wordmap, cmap="nipy_spectral")
	plt.show()

	#util.save_wordmap(wordmap, filename)
	#visual_recog.build_recognition_system(num_workers=num_cores)

	#conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())

	#vgg16 = torchvision.models.vgg16(pretrained=True).double()
	#vgg16.eval()
	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
	#conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())

