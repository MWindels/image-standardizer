from scripts.python import resolve
from scripts.python import parse
from scripts.python import image
import numpy
import sys
import os

def fit(pixels, type):
	'''
	Fits pixels to type's range.
	The type parameter and pixels's type are both assumed to be unsigned integer types.
	'''
	
	return numpy.rint(numpy.iinfo(type).max * (pixels / numpy.iinfo(pixels.dtype).max)).astype(type)

def to_rgb(in_dir, r_path, g_path, b_path, out_path):
	'''
	Merge three single-channel images in in_dir into one RGB image and save it to out_path.
	'''
	
	r_pixels = image.open(os.path.join(in_dir, r_path))
	g_pixels = image.open(os.path.join(in_dir, g_path))
	b_pixels = image.open(os.path.join(in_dir, b_path))
	
	assert(r_pixels.shape == g_pixels.shape == b_pixels.shape)
	assert(len(r_pixels.shape) == len(g_pixels.shape) == len(b_pixels.shape) == 2)
	
	out_type = numpy.uint8
	image.save(numpy.stack((fit(r_pixels, out_type), fit(g_pixels, out_type), fit(b_pixels, out_type)), axis=-1), out_path)


if __name__ == '__main__':
	parse_options = []
	
	parse_requirements = [\
		parse.Requirement('input_directory', description='The directory from which to read images.'),\
		parse.Requirement('red_glob', description='A glob string describing what red images to use (relative to input_directory).'),\
		parse.Requirement('green_glob', description='A glob string describing what green images to use (relative to input_directory).'),\
		parse.Requirement('blue_glob', description='A glob string describing what blue images to use (relative to input_directory).'),\
		parse.Requirement('output_directory', description='The directory to which RGB images will be written.'),\
		parse.Requirement('output_prefix', description='The string to prefix files with when saving.'),\
		parse.Requirement('output_extension', description='The extension to use when saving.')\
	]
	
	info, options, requirements = parse.parse(parse_options, parse_requirements)
	
	if len(info) == 0:
		r_paths = resolve.resolve(requirements['input_directory'], requirements['red_glob'])
		g_paths = resolve.resolve(requirements['input_directory'], requirements['green_glob'])
		b_paths = resolve.resolve(requirements['input_directory'], requirements['blue_glob'])
		assert len(r_paths) == len(g_paths) == len(b_paths) > 0
		
		idx = 1
		digits = numpy.floor(numpy.log10(len(r_paths))).astype(numpy.int_) + 1
		for r_path, g_path, b_path in zip(r_paths, g_paths, b_paths):
			to_rgb(requirements['input_directory'], r_path, g_path, b_path, os.path.join(requirements['output_directory'], f'{requirements["output_prefix"]}{idx:0{digits}d}.{requirements["output_extension"]}'))
			idx += 1
	else:
		print(info, file=sys.stderr)