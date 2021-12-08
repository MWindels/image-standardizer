if __package__ is None or len(__package__) == 0:
	import resolve
	import parse
	import image

import numpy
import sys
import os

def equalizer(histogram):
	'''
	Compute the transform that will equalize a given histogram.
	
	The transform is (len(histogram) - 1) * CDF(b) for each bin b in the histogram.  CDF(b) is the cumulative distribution function computed at bin b.  This is the integral of the normalized histogram (or the probability distribution function) from 0 to b.
	'''
	
	return numpy.rint((histogram.size - 1) * (numpy.cumsum(histogram) / numpy.sum(histogram))).astype(numpy.int_)

def matcher(source, destination):
	'''
	Compute the transform that will match a source histogram to a destination histogram.
	
	To match histograms, we first take pixel values in the source distribution and map them to a near-uniform distribution.  We then take the resulting values and match them (as closely as possible) with the values of a near-uniform distribution constructed from the destination distribution.  Finally, we map those values to the destination distribution itself.  In other words, the matching transform is the inverse of the destination equalization transform composed with the source equalization transform.  There are a couple issues with this approach, however.
	
	First, the equalization transforms are not onto.  This means that a value which appears in the source's near-uniform distribution may not appear in the destination's near-uniform distribution.  To resolve this, binary search is used to identify where the source value fits relative to the destination values.  Then, the source value is matched to the nearest destination value.
	
	Second, the equalization transforms are not one-to-one.  This means that the inverse of the equalization transforms may map one input to many outputs.  To resolve this issue, the binary search continues looking for matches between the values in the near-uniform distributions while minimizing the difference between the values in the source and destination distributions.  We need not necessarily select a value this way, but we do so because keeping the input and output values as close as possible will minimize deformation in the transformed images.
	'''
	
	source_equalizer = equalizer(source)
	destination_equalizer = equalizer(destination)
	#assert source_equalizer.size == destination_equalizer.size
	source_destination_matcher = numpy.empty(source_equalizer.size, dtype=numpy.int_)
	
	for src_idx in range(source_equalizer.size):
		low_dest_idx, high_dest_idx = 0, destination_equalizer.size
		nearest_match_idx = None
		
		while low_dest_idx < high_dest_idx:
			mean_dest_idx = (low_dest_idx + high_dest_idx) // 2
			if source_equalizer[src_idx] < destination_equalizer[mean_dest_idx]:
				high_dest_idx = mean_dest_idx
			elif source_equalizer[src_idx] > destination_equalizer[mean_dest_idx]:
				low_dest_idx = mean_dest_idx + 1
			else:
				nearest_match_idx = mean_dest_idx
				if src_idx < mean_dest_idx:
					high_dest_idx = mean_dest_idx
				elif src_idx > mean_dest_idx:
					low_dest_idx = mean_dest_idx + 1
				else:
					break
		
		if nearest_match_idx is not None:
			source_destination_matcher[src_idx] = nearest_match_idx
		else:
			low_dest_idx -= 1
			candidate_idxes = []
			if 0 <= low_dest_idx < destination_equalizer.size:
				candidate_idxes.append(low_dest_idx)
			if 0 <= high_dest_idx < destination_equalizer.size:
				candidate_idxes.append(high_dest_idx)
			
			candidate_idxes = numpy.array(candidate_idxes)
			
			candidate_value_distances = numpy.absolute(destination_equalizer[candidate_idxes] - source_equalizer[src_idx])
			candidate_idxes = candidate_idxes[candidate_value_distances == numpy.min(candidate_value_distances)]
			
			source_destination_matcher[src_idx] = candidate_idxes[numpy.argmin(numpy.absolute(candidate_idxes - src_idx))]
	
	return source_destination_matcher

def transform(in_path, out_path, transformer):
	'''
	Opens the image at in_path and applies a pixel transform created by calling transformer on the image's histogram.  Saves the resulting image at out_path.
	'''
	
	pixels = image.open(in_path)
	histogram = image.histogram(pixels)
	
	#Note: PIL needs the dtype of pixels to be unchanged.
	image.save(transformer(histogram).astype(pixels.dtype)[pixels], out_path)


if __name__ == '__main__':
	command_options = [\
		parse.Option('reference', short_name='r', description='Perform histogram matching using this file as reference.', parameters=('reference_image',))\
	]
	
	command_requirements = [\
		parse.Requirement('input_directory', description='The directory from which images will be read.'),\
		parse.Requirement('files', description='The files to standardize (relative to input_directory).', variadic=True),\
		parse.Requirement('output_directory', description='The directory to which images will be written.')\
	]
	
	info, options, requirements = parse.parse(command_options, command_requirements)
	
	if len(info) == 0:
		transformer = lambda hist: equalizer(hist)
		
		if 'reference' in options:
			ref_histogram = image.open_histogram(options['reference'][0])
			transformer = lambda hist: matcher(hist, ref_histogram)
		
		for file in resolve.resolve(requirements['input_directory'], *requirements['files']):
			transform(os.path.join(requirements['input_directory'], file), os.path.join(requirements['output_directory'], file), transformer)
	else:
		print(info, file=sys.stderr)