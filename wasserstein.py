from torchvision.models.squeezenet import Fire
from scripts.python import parse
from scripts.python import image
from scripts.python import histo
from torch import nn
import itertools
import torch
import numpy
import math
import copy
import csv
import sys
import os

class Histogram_Matches(torch.utils.data.Dataset):
	'''
	A object representing a set of source images matched to reference histograms.
	'''
	
	def __init__(self, data_file_path, scale_sources_to=None, scale_references_to=None):
		'''
		Initializes a new source/reference match dataset.
		'''
		
		self.matches = []
		self.channels = None
		self.source_dimensions = scale_sources_to
		self.reference_bins = scale_references_to
		self.scale_source = (lambda source: source) if scale_sources_to is None else (lambda source: image.resize(source, scale_sources_to))
		self.scale_reference = (lambda reference: reference) if scale_references_to is None else (lambda reference: histo.resize(reference, scale_references_to))
		
		with open(data_file_path, mode='r', newline='') as data_file:
			for idx, row in enumerate(csv.reader(data_file)):
				segments = [tuple(group) for is_delim, group in itertools.groupby(row, lambda entry: len(entry) == 0) if not is_delim]
				
				if len(segments) != 2:
					raise ValueError(f'Expected row {idx + 1} to take the form (<source paths...>, <empty entry>, <reference paths...>).')
				
				source_paths, reference_paths = segments
				
				if len(source_paths) != len(reference_paths):
					raise ValueError(f'Expected row {idx + 1} to have an equal number of source paths and reference paths.')
				
				if self.channels is None:
					self.channels = len(source_paths)
				elif len(source_paths) != self.channels:
					raise ValueError(f'Expected row {idx + 1} to have as many source and reference paths as previous rows.')
				
				if scale_sources_to is None:
					for source_path in source_paths:
						dims = image.open(source_path).shape
						
						if self.source_dimensions is None:
							self.source_dimensions = dims
						elif dims != self.source_dimensions:
							raise ValueError(f'Expected {source_path} (row {idx + 1}) to have the same dimensions as previous source images.')
				
				if scale_references_to is None:
					for reference_path in reference_paths:
						bins = len(histo.open(reference_path))
						
						if self.reference_bins is None:
							self.reference_bins = bins
						elif bins != self.reference_bins:
							raise ValueError(f'Expected {reference_path} (row {idx + 1}) to have the same number of bins as previous reference histograms.')
				
				self.matches.append((source_paths, reference_paths))
		
		if self.channels is None:
			self.channels = 0
	
	def __len__(self):
		return len(self.matches)
	
	def __getitem__(self, idx):
		return torch.stack([torch.from_numpy(image.normalize(self.scale_source(image.open(source_path))).astype(numpy.float32)) for source_path in self.matches[idx][0]]), torch.stack([torch.from_numpy(histo.normalize(self.scale_reference(histo.open(reference_path))).astype(numpy.float32)) for reference_path in self.matches[idx][1]])

class Wasserstein_SqueezeNet(nn.Module):
	'''
	An object representing the neural network described in "Automatic Color Correction for Multisource Remote Sensing Images with Wasserstein CNN" (J. Guo et al).
	
	This is a modified version of SqueezeNet version 1.1 (see https://paperswithcode.com/model/squeezenet?variant=squeezenet-1-1 for details).
	
	The purpose of this network is to predict reference histograms suitable for histogram matching with given source images.
	'''
	
	def __init__(self, source_dimensions, reference_bins, channels, dropout_probability=0.5):
		'''
		Initializes a SqueezeNet (1.1) implementation designed to predict suitable reference histograms.
		'''
		
		super(Wasserstein_SqueezeNet, self).__init__()
		
		self.init_args = [source_dimensions, reference_bins, channels]
		self.init_kwargs = {'dropout_probability': dropout_probability}
		
		assert len(source_dimensions) == 2
		self.source_dimensions = source_dimensions
		self.source_tensor_dimensions = (self.source_dimensions[1], self.source_dimensions[0])
		self.pool_dimensions = ((self.source_tensor_dimensions[0] - 15) // 8 + 1, (self.source_tensor_dimensions[1] - 15) // 8 + 1)
		self.reference_bins = reference_bins
		self.channels = channels
		
		#The paper says that the output from the first convolutional layer should be 96 channels, are they sure it's not 64?
		#Unlike the PyTorch SqueezeNet, we won't use ceil_mode on the max pool layers (causes issues when a dimension is 1300).
		
		self.eighth_pool = nn.AvgPool2d(kernel_size=15, stride=8)
		
		self.conv_block = nn.Sequential(
			nn.Conv2d(channels, 96, kernel_size=3, stride=2),
			nn.ReLU(inplace=True)
		)
		
		self.quarter_pool = nn.AvgPool2d(kernel_size=7, stride=4)
		
		self.initial_fire_block = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=2),
			Fire(96, 16, 64, 64),
			Fire(128, 16, 64, 64)
		)
		
		self.half_pool = nn.AvgPool2d(kernel_size=3, stride=2)
		
		self.interior_fire_block = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=2),
			Fire(128, 32, 128, 128),
			Fire(256, 32, 128, 128)
		)
		
		#No pooling necessary between these blocks.
		
		last_conv = nn.Conv2d(512, 512, kernel_size=1)
		self.final_fire_block = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=2),
			Fire(256, 48, 192, 192),
			Fire(384, 48, 192, 192),
			Fire(384, 64, 256, 256),
			Fire(512, 64, 256, 256),
			nn.Dropout(p=dropout_probability),
			last_conv,
			nn.ReLU(inplace=True)
		)
		
		self.deconvolve = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2)
		
		self.outputs = nn.ModuleList([nn.Sequential(nn.Linear(self.pool_dimensions[0] * self.pool_dimensions[1] * (self.channels + 96 + 128 + 256 + 512), self.reference_bins), nn.Softmax(dim=1)) for idx in range(self.channels)])
		
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				if module is last_conv:
					nn.init.normal_(module.weight, mean=0.0, std=0.01)
				else:
					nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
				
				if module.bias is not None:
					nn.init.constant_(module.bias, 0.0)
			elif isinstance(module, nn.Linear):
				nn.init.normal_(module.weight, mean=0.0, std=0.01)
				
				if module.bias is not None:
					nn.init.constant_(module.bias, 0.0)
	
	def forward(self, x):
		'''
		Predicts suitable reference histograms for a tensor of source images.
		'''
		
		assert x.shape[1:] == (self.channels, *self.source_tensor_dimensions)
		
		convolved = self.conv_block(x)
		initial_fired = self.initial_fire_block(convolved)
		interior_fired = self.interior_fire_block(initial_fired)
		final_fired = self.final_fire_block(interior_fired)
		
		concatenated = torch.cat([torch.flatten(self.eighth_pool(x), start_dim=1), torch.flatten(self.quarter_pool(convolved), start_dim=1), torch.flatten(self.half_pool(initial_fired), start_dim=1), torch.flatten(interior_fired, start_dim=1), torch.flatten(self.deconvolve(final_fired), start_dim=1)], dim=1)
		
		return torch.stack([output(concatenated) for output in self.outputs], dim=1)

def pseudo_wasserstein(first, second):
	'''
	Computes a Wasserstein-like distance between two distributions.
	'''
	
	return torch.sum(torch.square(first - second + 1) - 1, dim=2) #Would probably be better to penalize relative to the size of the bins involved.

def formatters(batch_size, size, epochs=1):
	'''
	Returns three functions which return strings for pretty printing epoch, batch, and summary information respectively.
	'''
	
	epoch_digits, size_digits = int(math.log10(epochs)) + 1, int(math.log10(size)) + 1
	return lambda epoch: f'Epoch {str(epoch).rjust(epoch_digits, " ")} {"-" * (12 + 2 * size_digits - epoch_digits)}', lambda loss, batch: f'Loss: {loss:.7f} ({str(batch * batch_size).zfill(size_digits)}/{str(size).zfill(size_digits)})', lambda loss_sum: f'Average Loss: {loss_sum / size:.7f}'

def train(train_loader, validation_loader, model, optimizer, device, epochs):
	'''
	Trains a given model with given data.  The parameters which produce the lowest validation loss are returned.
	'''
	
	epoch_formatter, batch_formatter, summary_formatter = formatters(train_loader.batch_size, len(train_loader.dataset), epochs)
	
	best_val_loss, best_model_state = test(validation_loader, model, device, summary_prefix='Validation'), copy.deepcopy(model.state_dict())
	
	for epoch in range(epochs):
		print(epoch_formatter(epoch + 1))
		model.train()
		
		cumulative_loss = 0.0
		
		for batch, (sources, references) in enumerate(train_loader):
			sources, references = sources.to(device), references.to(device)
			
			predictions = model(sources)
			losses = torch.mean(pseudo_wasserstein(predictions, references), dim=1)
			batch_loss = torch.mean(losses)
			
			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			
			print(batch_formatter(batch_loss.item(), batch))
			cumulative_loss += torch.sum(losses).item()
		
		print(f'(Training) {summary_formatter(cumulative_loss)}\n')
		val_loss = test(validation_loader, model, device, summary_prefix='Validation')
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_model_state = copy.deepcopy(model.state_dict())
	
	return best_model_state

def test(data_loader, model, device, summary_prefix='Testing', out_dir=None):
	'''
	Tests a given model with given data.  Returns the average loss across all given data.
	'''
	
	model.eval()
	
	_, batch_formatter, summary_formatter = formatters(data_loader.batch_size, len(data_loader.dataset))
	
	cumulative_loss = 0.0
	
	with torch.no_grad():
		for batch, (sources, references) in enumerate(data_loader):
			sources, references = sources.to(device), references.to(device)
			
			predictions = model(sources)
			losses = torch.mean(pseudo_wasserstein(model(sources), references), dim=1)
			batch_loss = torch.mean(losses)
			
			print(batch_formatter(batch_loss.item(), batch))
			cumulative_loss += torch.sum(losses).item()
			
			if out_dir is not None:
				for idx in range(len(predictions)):
					idx_out_dir = os.path.join(out_dir, str(batch * data_loader.batch_size + idx + 1))
					
					for ref_idx, reference in enumerate(predictions[idx]):
						histo.save(reference, os.path.join(idx_out_dir, f'{ref_idx + 1}.npy'))
	
	print(f'({summary_prefix}) {summary_formatter(cumulative_loss)}')
	
	return cumulative_loss / len(data_loader.dataset)

def load(file_path):
	'''
	Loads a Wasserstein_SqueezeNet from a file.
	'''
	
	contents = torch.load(file_path)
	model = Wasserstein_SqueezeNet(*(contents['args']), **(contents['kwargs']))
	model.load_state_dict(contents['model'])
	return model

def save(model, file_path):
	'''
	Saves a Wasserstein_SqueezeNet to a file.
	'''
	
	torch.save({'args': model.init_args, 'kwargs': model.init_kwargs, 'model': model.state_dict()}, file_path)


if __name__ == '__main__':
	command_options = [\
		parse.Option('train', short_name='tr', description='Train with given hyperparameters.', parameters=('train_csv', 'epochs', 'learning_rate', 'regularization')),\
		parse.Option('load', short_name='ld', description='Load the model from a given file.', parameters=('model_file',)),\
		parse.Option('save', short_name='sv', description='If training, save the best model (based on validation) to a given file.', parameters=('model_file',)),\
		parse.Option('output', short_name='o', description='If testing, output predicted reference histograms to a given directory.', parameters=('output_directory',)),\
		parse.Option('source_dimensions', short_name='sdm', description='If not loading a model, resize source images to the given dimensions.', parameters=('width', 'height')),\
		parse.Option('reference_bins', short_name='rbn', description='If not loading a model, adjust reference histograms to use the given number of bins.', parameters=('bins',)),\
		parse.Option('batch_size', short_name='bs', description='Use batches of a given size.', parameters=('size',)),\
		parse.Option('shuffle', short_name='sh', description='Shuffle loaded data.',)\
	]
	
	command_requirements = [\
		parse.Requirement('test_or_val_csv', description='The path to the file describing the testing (default) or validation (if training) data set.')\
	]
	
	info, options, requirements = parse.parse(command_options, command_requirements)
	
	if len(info) == 0:
		model = load(options['load'][0]) if 'load' in options else None
		
		scale_sources_to = None
		if model is None:
			if 'source_dimensions' in options:
				scale_sources_to = (int(options['source_dimensions'][0]), int(options['source_dimensions'][1]))
		else:
			scale_sources_to = model.source_dimensions
		
		scale_references_to = None
		if model is None:
			if 'reference_bins' in options:
				scale_references_to = int(options['reference_bins'][0])
		else:
			scale_references_to = model.reference_bins
		
		test_val_loader = torch.utils.data.DataLoader(\
			Histogram_Matches(\
				requirements['test_or_val_csv'],\
				scale_sources_to=scale_sources_to,\
				scale_references_to=scale_references_to\
			),\
			batch_size=int(options['batch_size'][0]) if 'batch_size' in options else 1,\
			shuffle='shuffle' in options\
		)
		
		if model is None:
			model = Wasserstein_SqueezeNet(test_val_loader.dataset.source_dimensions, test_val_loader.dataset.reference_bins, test_val_loader.dataset.channels)
		
		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		model.to(device)
		
		if 'train' in options:
			train_loader = torch.utils.data.DataLoader(\
				Histogram_Matches(\
					options['train'][0],\
					scale_sources_to=scale_sources_to,\
					scale_references_to=scale_references_to,\
				),\
				batch_size=int(options['batch_size'][0]) if 'batch_size' in options else 1,\
				shuffle='shuffle' in options\
			)
			
			assert train_loader.dataset.source_dimensions == test_val_loader.dataset.source_dimensions
			assert train_loader.dataset.reference_bins == test_val_loader.dataset.reference_bins
			
			optimizer = torch.optim.SGD(model.parameters(), lr=float(options['train'][2]), weight_decay=float(options['train'][3]))
			
			best_model_state = train(train_loader, test_val_loader, model, optimizer, device, int(options['train'][1]))
			
			if 'save' in options:
				model.load_state_dict(best_model_state)
				save(model, options['save'][0])
		else:
			test(test_val_loader, model, device, out_dir=options['output'][0] if 'output' in options else None)
	else:
		print(info, file=sys.stderr)