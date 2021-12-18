if __package__ is None or len(__package__) == 0:
	import image
else:
	from . import image

import numpy
import os

def open(file_path):
	'''
	Opens the file at file_path and returns its contents if it's a histogram.  Otherwise, assumes it contains an image and returns its histogram.
	'''
	
	try:
		contents = numpy.load(file_path)
	except (IOError, ValueError):
		return image.histogram(image.open(file_path))
	else:
		if type(contents) is numpy.ndarray and len(contents.shape) == 1:
			return contents
		elif type(contents) is not numpy.ndarray:
			contents.close()
		
		raise ValueError(f'\"{file_path}\" contains numpy data, but it is not a histogram.')

def normalize(histogram):
	'''
	Normalizes a histogram, putting its bins in the range [0, 1].  The sum of all bins is 1.
	'''
	
	return histogram / numpy.sum(histogram)

def cdf(histogram):
	'''
	Computes the cumulative distribution function of a histogram.
	'''
	
	return numpy.cumsum(histogram) / numpy.sum(histogram)

def resize(histogram, bins):
	'''
	Resizes a histogram to have a new number of bins.
	'''
	
	assert bins > 0
	if len(histogram) < bins:
		bin_map = numpy.floor(numpy.arange(0, bins, bins / len(histogram))).astype(numpy.int_)
		bin_widths = numpy.append(bin_map[1:], bins) - bin_map
		return numpy.repeat(histogram / bin_widths, bin_widths)
		#Interpolation might look nicer, but the interpolation points would need to be less than histogram / bin_widths (otherwise extra mass is added as a triangle formed between interpolation points).
		#return numpy.interp(numpy.arange(bins), bin_map, histogram / bin_widths)
	elif len(histogram) > bins:
		return numpy.add.reduceat(histogram, numpy.floor(numpy.arange(0, len(histogram), len(histogram) / bins)).astype(numpy.int_))
	else:
		return histogram

def save(histogram, save_path):
	'''
	Saves a histogram to a file at save_path.  Creates directories along the path as necessary.
	'''
	
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	numpy.save(save_path, histogram, allow_pickle=False)