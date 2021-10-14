import functools
import sys

class Option:
	'''
	Represents a command line option with zero or more parameters.
	'''
	
	def __init__(self, name, short_name=None, description='', parameters=()):
		self.name = name
		self.short_name = short_name
		self.description = description
		self.parameters = parameters

class Requirement:
	'''
	Represents a required command line argument.
	'''
	
	def __init__(self, name, description='', variadic=False):
		self.name = name
		self.description = description
		self.variadic = variadic

def parse(options=[], requirements=[]):
	'''
	Parses sys.argv based on the options and requirements provided.  A help option is automatically added by this function if one doesn't already exist.
	
	Returns an information string specifying whether something went wrong during parsing, a dictionary of used options mapping names to arguments, and a dictionary of used requirements mapping names to arguments.
	'''
	
	options_by_name = {o.name: o for o in options}
	options_by_short_name = {o.short_name: o for o in options if o.short_name is not None}
	
	if 'help' not in options_by_name:
		help_option = Option('help', short_name='h', description='Print this help string.')
		options_by_name['help'] = help_option
		options_by_short_name['h'] = help_option
	
	
	info = ''
	used_options = {}
	used_requirements = {}
	
	
	last_option = None
	for token in sys.argv[1:]:
		try:
			if token[0] == '-' and len(token) > 1:
				option = options_by_name[token[2:]] if token[1] == '-' else options_by_short_name[token[1:]]
				used_options[option.name] = []
				last_option = option
			else:
				raise KeyError()
		except KeyError:
			if last_option is not None:
				used_options[last_option.name].append(token)
			else:
				break	#Assume no options.
	
	leftovers = sys.argv[1:]
	if last_option is not None:
		leftovers = used_options[last_option.name][len(last_option.parameters):]
		used_options[last_option.name] = used_options[last_option.name][:len(last_option.parameters)]
	
	for name, arguments in used_options.items():
		parameters = options_by_name[name].parameters
		if len(parameters) == len(arguments):
			used_options[name] = tuple(arguments)
		else:
			info = f'Error: Expected {len(parameters)} argument{"s" if len(parameters) != 1 else ""} for option \"{name}\", but got {len(arguments)}.'
	
	
	variadic_idxes = [idx for idx, requirement in enumerate(requirements) if requirement.variadic]
	if len(variadic_idxes) == 0:
		if len(leftovers) == len(requirements):
			used_requirements = {requirement.name: argument for requirement, argument in zip(requirements, leftovers)}
		else:
			info = f'Error: Expected {len(requirements)} requirement{"s" if len(requirements) != 1 else ""}, but got {len(leftovers)}.'
	elif len(variadic_idxes) == 1:
		if len(leftovers) >= len(requirements) - 1:
			variadic_idx = variadic_idxes[0]
			variadic_end = len(leftovers) - (len(requirements) - variadic_idx - 1)
			used_requirements = {requirement.name: argument for requirement, argument in zip(requirements, leftovers[:variadic_idx] + [leftovers[variadic_idx:variadic_end]] + leftovers[variadic_end:])}
		else:
			info = f'Error: Expected at least {len(requirements) - 1} requirement{"s" if len(requirements) - 1 != 1 else ""}, but got {len(leftovers)}.'
	else:
		raise ValueError('More than one variadic requirement.')
	
	
	if 'help' in used_options:
		option_names = ''
		options_string = ''
		for option in options_by_name.values():
			parameter_names = functools.reduce(lambda acc, param: f'{acc} {param}', option.parameters, '')
			option_names += f' [{f"-{option.short_name}" if option.short_name is not None else f"--{option.name}"}{parameter_names}]'
			options_string += f'\t{f"-{option.short_name} (--{option.name})" if option.short_name is not None else f"--{option.name}"}{parameter_names}: {option.description}\n'
		
		requirement_names = ''
		requirements_string = ''
		for requirement in requirements:
			requirement_names += f' {f"[{requirement.name}...]" if requirement.variadic else f"{requirement.name}"}'
			requirements_string += f'\t{f"[{requirement.name}...]" if requirement.variadic else f"{requirement.name}"}: {requirement.description}\n'
		
		info = f'Usage: python {sys.argv[0]}{option_names}{requirement_names}\n\nOptions:\n{options_string}\nRequirements:\n{requirements_string}'
	
	return info, used_options, used_requirements