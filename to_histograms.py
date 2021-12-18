from scripts.python import resolve
from scripts.python import parse
from scripts.python import image
from scripts.python import histo
import numpy
import sys
import os

def to_histogram(in_path, out_path, mask_path=None):
	'''
	Reads an image from in_path, optionally masks it, computes its histogram and saves the result to out_path.
	'''
	
	pixels = image.open(in_path)
	
	if mask_path is not None:
		pixels = numpy.ma.array(pixels, mask=image.open(mask_path) == 0).compressed()
	
	histo.save(image.histogram(pixels), out_path)


if __name__ == '__main__':
	parse_options = [\
		parse.Option('mask', short_name='m', description='An image to apply as a mask before computing histograms.', parameters=('mask_image',))\
	]
	
	parse_requirements = [\
		parse.Requirement('input_directory', description='The directory from which to read images.'),\
		parse.Requirement('files', description='The files from which to compute histograms (relative to input_directory).', variadic=True),\
		parse.Requirement('output_directory', description='The directory to which histograms will be written.')\
	]
	
	info, options, requirements = parse.parse(parse_options, parse_requirements)
	
	if len(info) == 0:
		for file in resolve.resolve(requirements['input_directory'], *requirements['files']):
			to_histogram(os.path.join(requirements['input_directory'], file), os.path.join(requirements['output_directory'], os.path.splitext(file)[0]), options['mask'][0] if 'mask' in options else None)
	else:
		print(info, file=sys.stderr)