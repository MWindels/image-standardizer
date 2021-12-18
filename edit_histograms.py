from scripts.python import standardize
from scripts.python import image
from scripts.python import histo
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import ttk
import functools
import tkinter
import bisect
import numpy
import enum
import math
import os

class Control_Point:
	'''
	An object representing a control point of a spline.
	'''
	
	def __init__(self, x, y=0, bounds=(0, math.inf)):
		'''
		Initializes a new control point.
		'''
		
		self.x = x
		self.y = y if y >= 0 else 0
		self.bounds = bounds
	
	def __lt__(self, other):
		return self.x < other.x
	
	def add(self, size, transform, parent):
		'''
		Adds the control point to parent.
		'''
		
		self.remove()
		
		self.parent = parent
		tr_x, tr_y = transform(self.x, self.y)
		self.object = self.parent.create_oval(\
			tr_x - math.floor(size / 2),\
			tr_y - math.floor(size / 2),\
			tr_x + math.ceil(size / 2) + 1,\
			tr_y + math.ceil(size / 2) + 1,\
			activefill='yellow',\
			outline='black',\
			fill='black'\
		)
	
	def move(self, transform, y_new):
		'''
		Moves the control point's y coordinate to y_new.
		'''
		
		try:
			y_new_bounded = min(max(self.bounds[0], y_new), self.bounds[1])
			self.parent.move(self.object, *[pair[0] - pair[1] for pair in zip(transform(self.x, y_new_bounded), transform(self.x, self.y))])
			self.y = y_new_bounded
		except AttributeError:
			pass
	
	def remove(self):
		'''
		Removes the control point from its parent.
		'''
		
		try:
			self.parent.delete(self.object)
		except AttributeError:
			pass

class Editor_Canvas:
	'''
	An object representing a canvas with an editable histogram.
	'''
	
	def __init__(self, histogram_colour='black', parent=None):
		'''
		Initializes a new editor canvas.
		'''
		
		self.histogram_scaling = 2 / 3
		self.histogram_colour = histogram_colour.lower()
		self.control_size = self.internal_padding = 10
		self.spline_width = 4
		self.bin_width = 2
		
		self.canvas = tkinter.Canvas(parent, bg='white', height=160, width=480)
		self.rebuild(numpy.zeros(256))
		self.canvas.grid(ipadx=self.internal_padding, ipady=self.internal_padding)
		
		self.canvas.bind('<Double-Button-1>', lambda event: self.add_control(*self.cartesian_coordinates(event.x, event.y)))
	
	def canvas_coordinates(self, x, y):
		'''
		Transforms a pair of Cartesian coordinates into canvas coordinates.
		Use these coordinates for drawing/moving.
		
		Note: Other code assumes that the transform function is monotonically increasing.
		'''
		
		return x + self.internal_padding, (self.canvas.winfo_reqheight() - 1) - y + self.internal_padding
	
	def cartesian_coordinates(self, x, y):
		'''
		Transforms a pair of canvas coordinates into Cartesian coordinates.
		Perfer this coordinate system.
		'''
		
		return x - self.internal_padding, (self.canvas.winfo_reqheight() - 1) - y + self.internal_padding
	
	def rebuild(self, reference, approximate=False):
		'''
		Rebuilds the canvas with a new reference histogram.
		'''
		
		self.reference = reference
		
		try:
			for bin in self.bins:
				self.canvas.delete(bin)
			for control in self.controls:
				control.remove()
		except AttributeError:
			pass
		
		self.reduced_reference = histo.resize(self.reference, self.canvas.winfo_reqwidth() / self.bin_width)
		self.bin_max = self.reduced_reference.max()
		if self.bin_max == 0:
			self.bin_max = 1
		
		self.bins = []
		for idx, bin in enumerate(self.reduced_reference):
			self.bins.append(self.canvas.create_rectangle(*self.canvas_coordinates(idx * self.bin_width, 0), *self.canvas_coordinates((idx + 1) * self.bin_width, self.canvas.winfo_reqheight() * self.histogram_scaling * (bin / self.bin_max)), outline=self.histogram_colour, fill=self.histogram_colour))
		
		self.controls = []
		self.add_control(0, 0, removable=False, padded=False, update=False)
		self.add_control(self.canvas.winfo_reqwidth() - 1, 0, removable=False, padded=False, update=False)
		
		if approximate:
			for idx in range(1, math.floor((self.canvas.winfo_reqwidth() - 1) / self.control_size)):
				self.add_control(idx * self.control_size, 0, update=False)
			
			self.fit_controls(update=False)
		
		self.update_spline()
	
	def _control_padding_predicate(self, x, idx):
		'''
		A predicate that returns true if x (to be inserted at idx) has sufficient padding around it.
		'''
		
		return 0 < idx < len(self.controls) and self.controls[idx - 1].x + math.ceil(self.control_size / 4) < x < self.controls[idx].x - math.floor(self.control_size / 4)
	
	def add_control(self, x, y, removable=True, padded=True, update=True):
		'''
		Adds a new control point to the canvas.
		'''
		
		control = Control_Point(x, y, bounds=(0, self.canvas.winfo_reqheight() - 1))
		idx = bisect.bisect(self.controls, control)
		if idx == 0 or self.controls[idx - 1].x < x:
			if not padded or padded and self._control_padding_predicate(x, idx):
				self.controls.insert(idx, control)
				
				control.add(self.control_size, self.canvas_coordinates, self.canvas)
				self.canvas.tag_bind(control.object, '<B1-Motion>', lambda event: self.move_control(control.x, self.cartesian_coordinates(control.x, event.y)[1]))
				if removable:
					self.canvas.tag_bind(control.object, '<Button-3>', lambda event: self.remove_control(control.x))
				
				if update:
					self.update_spline()
	
	def move_control(self, x, y, update=True):
		'''
		Moves an existing control point on the canvas.
		'''
		
		idx = bisect.bisect(self.controls, Control_Point(x))
		if idx > 0 and self.controls[idx - 1].x == x:
			self.controls[idx - 1].move(self.canvas_coordinates, y)
			
			if update:
				self.update_spline()
	
	def remove_control(self, x, update=True):
		'''
		Removes a control point from the canvas.
		'''
		
		idx = bisect.bisect(self.controls, Control_Point(x))
		if idx > 0 and self.controls[idx - 1].x == x:
			self.controls[idx - 1].remove()
			self.controls.pop(idx - 1)
			
			if update:
				self.update_spline()
	
	def fit_controls(self, update=True):
		'''
		Fits the control points to the reference.
		'''
		
		for control in self.controls:
			control.move(self.canvas_coordinates, self.canvas.winfo_reqheight() * self.histogram_scaling * (self.reduced_reference[math.floor(control.x / self.bin_width)] / self.bin_max))
		
		if update:
			self.update_spline()
	
	def update_spline(self):
		'''
		Updates the spline drawn on the canvas.
		'''
		
		try:
			self.canvas.delete(self.spline)
		except AttributeError:
			pass
		
		self.spline = self.canvas.create_line(*[coordinate for point in [self.canvas_coordinates(control.x, control.y) for control in self.controls] for coordinate in point], fill='black', width=self.spline_width)
		self.canvas.tag_lower(self.spline, self.controls[0].object)	#Assumes that the first control point is always the first to be added.
	
	def histogram(self):
		'''
		Returns the histogram formed by the control points.
		'''
		
		reduced_histogram = numpy.interp(\
			numpy.arange(math.floor(self.controls[-1].x / self.bin_width) + 1),\
			numpy.asarray([math.floor(control.x / self.bin_width) for control in self.controls]),\
			numpy.asarray([self.bin_max * control.y / (self.histogram_scaling * self.canvas.winfo_reqheight()) for control in self.controls])\
		)
		
		return histo.resize(reduced_histogram, len(self.reference))

class Editor_Frame:
	'''
	An object representing the whole histogram editor.
	'''
	
	class Channels(enum.Enum):
		RED = 0
		GREEN = 1
		BLUE = 2
	
	def __init__(self, parent=None):
		'''
		Initializes a new editor frame.
		'''
		
		self.frame = tkinter.Frame(parent)
		
		self.source_canvas = tkinter.Canvas(self.frame, bg='black', width=640, height=480)
		self.source_images = {}
		self.source_image_shape = None
		self.source_canvas.grid(row=0, column=0, rowspan=len(list(self.Channels)))
		
		self.match_button = tkinter.ttk.Button(self.frame, text='Match', command=self.match_source)
		self.match_button.grid(row=0, column=1, rowspan=len(list(self.Channels)), padx=(5, 30))
		
		self.reference_names = {}
		self.reference_contents = {}
		for ch in list(self.Channels):
			reference_frame = tkinter.Frame(self.frame)
			self.reference_contents[f'{ch.name}_Frame'] = reference_frame
			
			self.reference_contents[f'{ch.name}_Load_Source'] = tkinter.ttk.Button(reference_frame, text='Load', command=functools.partial(self.load_source, ch.name))
			self.reference_contents[f'{ch.name}_Load_Source'].grid(row=1, column=0)
			
			self.reference_contents[f'{ch.name}_Source_Exists'] = tkinter.Frame(reference_frame, bg='white', width=10, height=10)
			self.reference_contents[f'{ch.name}_Source_Exists'].grid_propagate(False)
			self.reference_contents[f'{ch.name}_Source_Exists'].grid(row=2, column=0)
			
			self.reference_contents[f'{ch.name}_Clear_Source'] = tkinter.ttk.Button(reference_frame, text='Clear', command=functools.partial(self.clear_source, ch.name))
			self.reference_contents[f'{ch.name}_Clear_Source'].grid(row=3, column=0)
			
			self.reference_contents[f'{ch.name}_Canvas'] = Editor_Canvas(histogram_colour=ch.name, parent=reference_frame)
			self.reference_contents[f'{ch.name}_Canvas'].canvas.grid(row=0, column=1, rowspan=5)
			
			self.reference_contents[f'{ch.name}_Load_Reference'] = tkinter.ttk.Button(reference_frame, text='Load', command=functools.partial(self.load_reference, ch.name))
			self.reference_contents[f'{ch.name}_Load_Reference'].grid(row=1, column=2)
			
			self.reference_contents[f'{ch.name}_Fit_Reference'] = tkinter.ttk.Button(reference_frame, text='Fit', command=self.reference_contents[f'{ch.name}_Canvas'].fit_controls)
			self.reference_contents[f'{ch.name}_Fit_Reference'].grid(row=2, column=2)
			
			self.reference_contents[f'{ch.name}_Save_Reference'] = tkinter.ttk.Button(reference_frame, text='Save', command=functools.partial(self.save_reference, ch.name))
			self.reference_contents[f'{ch.name}_Save_Reference'].grid(row=3, column=2)
			
			reference_frame.grid(row=ch.value, column=2)
		
		self.frame.pack()
	
	def match_source(self):
		'''
		Matches the loaded source image to the (control point defined) reference histograms.
		'''
		
		try:
			self.source_canvas.delete(self.source_image_object)
		except AttributeError:
			pass
		
		if self.source_image_shape is not None:
			matched_source_image = numpy.zeros((*self.source_image_shape, len(self.Channels)), dtype=numpy.uint8)
			
			for channel, source_image in self.source_images.items():
				source_histogram = image.histogram(source_image)
				reference_histogram = self.reference_contents[f'{channel}_Canvas'].histogram()
				
				matched_source_image[:, :, self.Channels[channel].value] = numpy.floor(256 * (standardize.matcher(source_histogram, reference_histogram)[source_image] / len(reference_histogram))).astype(numpy.uint8) if numpy.sum(reference_histogram) > 0 else 0
			
			self.source_image_drawn = ImageTk.PhotoImage(image=Image.fromarray(matched_source_image).resize((self.source_canvas.winfo_reqwidth(), self.source_canvas.winfo_reqheight())))
			self.source_image_object = self.source_canvas.create_image(0, 0, anchor='nw', image=self.source_image_drawn)
	
	def load_source(self, channel):
		'''
		Loads a source image for a given channel and sets its reference histogram (and control points) to the source image's histogram.
		'''
		
		file = filedialog.askopenfilename(parent=self.frame, title=f'Load {channel} Source', filetypes=[('TIF File', '*.TIF')])
		
		if len(file) > 0:
			new_source_image = image.open(file)
			
			if self.source_image_shape is None or self.source_image_shape == new_source_image.shape:
				self.reference_contents[f'{channel}_Source_Exists']['bg'] = channel.lower()
				self.source_image_shape = new_source_image.shape
				self.source_images[channel] = new_source_image
				self.load_reference(channel, file=file)
			else:
				tkinter.messagebox.showerror(title='Error', message=f'Expected an image with shape {self.source_image_shape}, but {os.path.split(file)[1]} has shape {new_source_image.shape}.')
	
	def clear_source(self, channel):
		'''
		Clears a source image for a given channel.
		'''
		
		try:
			del self.source_images[channel]
		except KeyError:
			pass
		
		self.reference_contents[f'{channel}_Source_Exists']['bg'] = 'white'
		
		if len(self.source_images) == 0:
			self.source_image_shape = None
	
	def load_reference(self, channel, file=None):
		'''
		Loads a reference histogram for a given channel and sets its control points.
		'''
		
		if file is None:
			file = filedialog.askopenfilename(parent=self.frame, title=f'Load {channel} Reference', filetypes=[('Histogram', '*.npy'), ('TIF Files', '*.TIF')])
		
		if len(file) > 0:
			self.reference_names[channel] = os.path.splitext(os.path.split(file)[1])[0]
			self.reference_contents[f'{channel}_Canvas'].rebuild(histo.open(file), approximate=True)
	
	def save_reference(self, channel):
		'''
		Saves the (control point defined) reference histogram of a given channel.
		'''
		
		file = filedialog.asksaveasfilename(parent=self.frame, title=f'Save {channel} Reference', filetypes=[('Histogram', '*.npy')], defaultextension='.npy', initialfile=f'{self.reference_names[channel]}.npy' if channel in self.reference_names else '')
		
		if len(file) > 0:
			histo.save(self.reference_contents[f'{channel}_Canvas'].histogram(), file)


if __name__ == '__main__':
	window = tkinter.Tk()
	window.title('Histogram Editor')
	window.resizable(False, False)
	frame = Editor_Frame(parent=window)
	window.update_idletasks()
	window.mainloop()