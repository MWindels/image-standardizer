from PIL import Image
import numpy
import os

#parse.Option('depth', short_name='d', description='Adjust dynamic range to the specified bit depth.', parameters=('bit_depth',))

def bit_depths(pixels, desired_depth=None):
	'''
	Returns the bit depth of pixels and desired_depth.  If desired_depth is None, pixels' bit depth is returned twice.
	'''
	
	depth = 8 * pixels.itemsize
	return depth, depth if desired_depth is None else desired_depth

def compress(pixels, desired_depth=None):
	'''
	Compresses the dynamic range of pixels to 2^desired_depth.  If desired_depth is None, the dynamic range is unchanged.
	'''
	
	depth, desired_depth = bit_depths(pixels, desired_depth=desired_depth)
	return numpy.floor(pixels / 2 ** (depth - desired_depth)).astype(pixels.dtype)

def expand(pixels, desired_depth=None):
	'''
	Expands the dynamic range of pixels from 2^desired_depth back to the original bit depth.  If desired_depth is None, the dynamic range is unchanged.
	'''
	
	depth, desired_depth = bit_depths(pixels, desired_depth=desired_depth)
	return numpy.floor(2 ** (depth - desired_depth) * pixels).astype(pixels.dtype)



def open(img_path):
	'''
	Opens the image at img_path and returns its pixels as an array.
	'''
	
	with Image.open(img_path) as img:
		return numpy.asarray(img)

def open_histogram(file_path):
	'''
	Opens the file at file_path and returns its contents if it's a histogram.  Otherwise, assumes it contains an image and returns its histogram.
	'''
	
	try:
		contents = numpy.load(file_path)
	except (IOError, ValueError):
		return histogram(open(file_path))
	else:
		if type(contents) is numpy.ndarray and len(contents.shape) == 1:
			return contents
		elif type(contents) is not numpy.ndarray:
			contents.close()
		
		raise ValueError(f'\"{file_path}\" contains numpy data, but it is not a histogram.')

def histogram(pixels):
	'''
	Computes the histogram of pixels.  The number of bins is 2 to the power of the bit depth of pixels.
	'''
	
	depth = 8 * pixels.itemsize
	return numpy.histogram(pixels, bins=2 ** depth, range=(0, 2 ** depth))[0]

def save(pixels, save_path):
	'''
	Saves an array of pixels to a file at save_path.  Creates directories along the path as necessary.
	'''
	
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	Image.fromarray(pixels).save(save_path)