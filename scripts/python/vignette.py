if __package__ is None or len(__package__) == 0:
	import parse
	import image
else:
	from . import parse
	from . import image

import functools
import numpy
import sys
import os

def vignette_polynomial(width, height, centre_x, centre_y, *coefficients):
	'''
	Computes the vignetting polynomial.
	'''
	
	distances = numpy.sqrt(numpy.expand_dims((numpy.arange(width) - centre_x) ** 2, 0) + numpy.expand_dims((numpy.arange(height) - centre_y) ** 2, 1))
	return 1 + functools.reduce(lambda sum, idxd_coeff: sum + idxd_coeff[1] * distances ** (idxd_coeff[0] + 1), enumerate(coefficients), 0)

def vignette(in_path, out_path, black_level, centre_x, centre_y, *coefficients, reciprocal=False, apply=False):
	'''
	Opens the image at in_path and either corrects for vignetting or applies it.  Saves the resulting image to out_path.
	'''
	
	pixels = image.open(in_path)
	polynomial = vignette_polynomial(pixels.shape[1], pixels.shape[0], centre_x, centre_y, *coefficients)
	polynomial = 1 / polynomial if reciprocal else polynomial
	
	vignetted = numpy.rint((pixels / polynomial) + black_level if apply else (pixels - black_level) * polynomial).astype(numpy.int_)
	
	#vignetted = numpy.rint((pixels - black_level) / polynomial if reciprocal else (pixels - black_level) * polynomial).astype(numpy.int_)
	
	image.save(numpy.clip(vignetted, 0, 2 ** (8 * pixels.itemsize) - 1).astype(pixels.dtype), out_path)


if __name__ == '__main__':
	command_options = [\
		parse.Option('reciprocal', short_name='r', description='Multiply pixels by the reciprocal of the vignetting polynomial.'),\
		parse.Option('apply', short_name='a', description='Apply vignetting rather than correcting for it.')\
	]
	
	command_requirements = [\
		parse.Requirement('input_directory', description='The directory from which to read the image.'),\
		parse.Requirement('file', description='The image to correct (or vignette).'),\
		parse.Requirement('output_directory', description='The directory to which the corrected\\vignetted image will be written.'),\
		parse.Requirement('black_level', description='The black level of the image.'),\
		parse.Requirement('c_x', description='The vignetting centre\'s x coordinate.'),\
		parse.Requirement('c_y', description='The vignetting centre\'s y coordinate.'),\
		parse.Requirement('k', description='The vignetting polynomial\'s coefficients.', variadic=True)\
	]
	
	info, options, requirements = parse.parse(options=command_options, requirements=command_requirements)
	
	if len(info) == 0:
		vignette(\
			os.path.join(requirements['input_directory'], requirements['file']),\
			os.path.join(requirements['output_directory'], requirements['file']),\
			float(requirements['black_level']),\
			float(requirements['c_x']),\
			float(requirements['c_y']),\
			*map(lambda k: float(k), requirements['k']),\
			reciprocal='reciprocal' in options,\
			apply='apply' in options\
		)
	else:
		print(info, file=sys.stderr)