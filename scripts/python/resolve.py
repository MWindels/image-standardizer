if __package__ is None or len(__package__) == 0:
	import parse

import glob
import sys
import os

def resolve(in_dir, *paths):
	'''
	Resolve wildcards in paths relative to in_dir using python's glob ruleset.
	'''
	
	return [os.path.relpath(file, start=in_dir) for path in paths for file in glob.glob(os.path.join(in_dir, path))]


if __name__ == '__main__':
	command_options = [\
		parse.Option('newline', short_name='n', description='Separate output with newlines rather than spaces.')\
	]
	
	command_requirements = [\
		parse.Requirement('input_directory', description='The directory in which paths are located.'),\
		parse.Requirement('paths', description='A list of paths which may contain * and ? wildcard symbols.', variadic=True)\
	]
	
	info, options, requirements = parse.parse(command_options, command_requirements)
	
	if len(info) == 0:
		print(*[f'\"{file}\"' for file in resolve(requirements['input_directory'], *requirements['paths'])], sep='\n' if 'newline' in options else ' ')
	else:
		print(info, file=sys.stderr)