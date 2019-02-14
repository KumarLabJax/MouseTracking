# Handles generation of the data readers and manipulators
# Also includes the re-scaling values (to optionally be applied)

import tensorflow as tf
import numpy as np
from .transformer import *

# Reshaping Ellipse Labels
means = [0., 0., 10., 10., -1., -1.] # All positive, min/maj axis mins are 16.97/20.75
scale = [480., 480., 60., 120., 2., 2.] # min/maj axis are 45.9/106.3 max

################################################
# Base reading and augmentation primitives
# Other functions are based on these

# Returns the angle in radians
def atan2(y, x, epsilon=1.0e-12):
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y+epsilon, y)

    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle

# Reads a single input image
def read_image(filename, input_size):
	image_contents = tf.read_file(filename)
	image = tf.image.decode_png(image_contents, channels=1)
	image = tf.image.resize_images(image, [input_size, input_size])
	return image

# Augments a single input image
def augment_image(image, input_size, noise_std=5.0, bright_percent=0.05, contrast_percent=0.05):
	image = tf.add(image, tf.random_normal([input_size, input_size, 1], stddev=noise_std)) # Random Noise
	image = tf.image.random_brightness(image, bright_percent)
	image = tf.image.random_contrast(image, 1.0-contrast_percent, 1.0+contrast_percent)
	return image

# Reads in a single ellipse fit
def read_ellipse(filename):
	ellfit = tf.read_file(filename)
	record_defaults = [[0.0],[0.0],[0.0],[0.0],[0.0]]
	ellfit = tf.string_to_number(tf.string_split([ellfit],delimiter='\t').values)
	ellfit = tf.stack([ellfit[0], ellfit[1], ellfit[2], ellfit[3], tf.sin(ellfit[4]*np.pi/180.0), tf.cos(ellfit[4]*np.pi/180.0)])
	ellfit = tf.div(tf.subtract(ellfit, means), scale)
	return ellfit

# X/Y/Diag mirroring
# Applies to the reference, ellipse-fit, and segmentation
def rand_flip_input(reference, ellfit=None, seg=None):
	# Random transformations
	# 0-7 (or 8-fold increase)
	# 0 = normal, 1 = h, 2 = v, 3 = h + v, 4 = t, 5 = t + h, 6 = t + v, 7 = t + h + v
	# Note: randnum%2 == 1 for HF operation
	# 		randnum/2%2 == 1 for VF operation
	#		randnum/4%2 == 1 for T operation
	randnum = tf.random_uniform([1], minval=0, maxval=8, dtype=tf.int32)[0]
	# Horizontal flipping
	reference = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(reference), lambda: tf.image.flip_left_right(reference))
	if ellfit is not None:
		ellfit = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(ellfit), lambda: tf.stack([tf.subtract(1.,tf.unstack(ellfit)[0]), tf.unstack(ellfit)[1], tf.unstack(ellfit)[2], tf.unstack(ellfit)[3], tf.subtract(1.,tf.unstack(ellfit)[4]), tf.unstack(ellfit)[5]])) # -sin for horizontal flip
	if seg is not None:
		seg = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(seg), lambda: tf.image.flip_left_right(seg))

	# Vertical flipping
	randnum = tf.div(randnum,2)
	reference = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(reference), lambda: tf.image.flip_up_down(reference))
	if ellfit is not None:
		ellfit = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(ellfit), lambda: tf.stack([tf.unstack(ellfit)[0], tf.subtract(1.,tf.unstack(ellfit)[1]), tf.unstack(ellfit)[2], tf.unstack(ellfit)[3], tf.unstack(ellfit)[4], tf.subtract(1.,tf.unstack(ellfit)[5])])) #-cos for vertical flip
	if seg is not None:
		seg = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(seg), lambda: tf.image.flip_up_down(seg))

	# Transpose
	randnum = tf.div(randnum,2)
	reference = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(reference), lambda: tf.image.transpose_image(reference))
	if ellfit is not None:
		ellfit = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(ellfit), lambda: tf.stack([tf.unstack(ellfit)[1], tf.unstack(ellfit)[0], tf.unstack(ellfit)[2], tf.unstack(ellfit)[3], tf.unstack(ellfit)[5], tf.unstack(ellfit)[4]])) # sin/cos reversed for transpose
	if seg is not None:
		seg = tf.cond(tf.mod(randnum,2)<1, lambda: tf.identity(seg), lambda: tf.image.transpose_image(seg))
	return reference, ellfit, seg

# Rotation and Translation augmentations
# Applies to the reference, ellipse-fit, and segmentation
def shift_augment(reference, ellfit=None, seg=None, max_trans=15.0, max_rot=5.0):
	# For segmentation-only rotate around the middle + random noise
	if ellfit is None:
		ellfit = [0.5,0.5,0.5,0.5,0.5,0.5]
	# Generate the random values
	randTransX = tf.div(tf.subtract(tf.random_uniform([1], minval=-1.0, maxval=1.0)[0]*max_trans*2,means[0]),scale[0])
	randTransY = tf.div(tf.subtract(tf.random_uniform([1], minval=-1.0, maxval=1.0)[0]*max_trans*2,means[1]),scale[1])
	randRot = tf.random_uniform([1], minval=-1.0, maxval=1.0)[0]*max_rot*np.pi/180.0
	# Enforce some boundaries for keeping the mouse on in view...
	randTransX = tf.cond(tf.unstack(ellfit)[0]-randTransX>0, lambda: tf.identity(randTransX), lambda: tf.multiply(-1.0, randTransX))
	randTransX = tf.cond(tf.unstack(ellfit)[0]-randTransX<1, lambda: tf.identity(randTransX), lambda: tf.multiply(-1.0, randTransX))
	randTransY = tf.cond(tf.unstack(ellfit)[1]-randTransY>0, lambda: tf.identity(randTransY), lambda: tf.multiply(-1.0, randTransY))
	randTransY = tf.cond(tf.unstack(ellfit)[1]-randTransY<1, lambda: tf.identity(randTransY), lambda: tf.multiply(-1.0, randTransY))

	# Apply the transform to the reference
	reference = tf.reshape(reference, [1, tf.shape(reference)[0], tf.shape(reference)[1], 1])
	alpha = tf.cos(randRot)
	beta = tf.sin(randRot)
	xloc = tf.subtract(tf.multiply(tf.unstack(ellfit)[0]-randTransX/2.0,2.0),1.0) # Range of -1 to 1
	yloc = tf.subtract(tf.multiply(tf.unstack(ellfit)[1]-randTransY/2.0,2.0),1.0) # Range of -1 to 1
	# This is essentially [opencv's rotate matrix]+[[0,0,xtrans],[0,0,ytrans]]
	affine_trans = [[alpha, beta,tf.multiply(1.0-alpha,xloc)-tf.multiply(beta,yloc)+randTransX], [-beta, alpha,tf.multiply(beta,xloc)+tf.multiply(1.0-alpha,yloc)+randTransY]]
	reference = transformer(reference, affine_trans, (tf.shape(reference)[1],tf.shape(reference)[2]))
	reference = tf.reshape(reference, [tf.shape(reference)[1],tf.shape(reference)[2], 1])

	# Apply the transform to the seg
	if seg is not None:
		seg = tf.reshape(seg, [1, tf.shape(seg)[0], tf.shape(seg)[1], 1])
		seg = transformer(seg, affine_trans, (tf.shape(seg)[1],tf.shape(seg)[2]))
		seg = tf.reshape(seg, [tf.shape(seg)[1],tf.shape(seg)[2], 1])

	# Edit the ellipse fit values
	angle = atan2(tf.add(tf.multiply(tf.unstack(ellfit)[4], scale[4]), means[4]), tf.add(tf.multiply(tf.unstack(ellfit)[5], scale[5]), means[5])) # In radians
	ellfit = tf.stack([tf.unstack(ellfit)[0]-randTransX/2.0, tf.unstack(ellfit)[1]-randTransY/2.0, tf.unstack(ellfit)[2], tf.unstack(ellfit)[3], tf.div(tf.subtract(tf.sin(angle-randRot), means[4]), scale[4]), tf.div(tf.subtract(tf.cos(angle-randRot), means[5]), scale[5])])
	return reference, ellfit, seg

################################################
# For all readers, the following information is constant...
# input_queue[0] = image file path
# input_queue[1] = label file path

# Reads the image and segmentation values
def read_image_and_seg(input_queue, input_size):
	seg = read_image(input_queue[1], input_size)
	image = read_image(input_queue[0], input_size)
	image, _, seg = rand_flip_input(image, seg=seg)
	return image, seg

# Reads the image and segmentation values
def read_augment_image_and_seg(input_queue, input_size, max_trans=120.0, max_rot=45.0):
	image, seg = read_image_and_seg(input_queue, input_size)
	image = augment_image(image, input_size)
	image, _, seg = shift_augment(image, seg=seg, max_trans=max_trans, max_rot=max_rot)
	return image, seg

# Reads the image and ellipse regression values
def read_image_and_ellreg(input_queue, input_size):
	ellfit = read_ellipse(input_queue[1])
	image = read_image(input_queue[0], input_size)
	image, ellfit, _ = rand_flip_input(image, ellfit)
	return image, ellfit

# Reads and augments the image and ellipse regression values
def read_augment_image_and_ellreg(input_queue, input_size):
	ellfit = read_ellipse(input_queue[1])
	image = read_image(input_queue[0], input_size)
	image, ellfit, _ = rand_flip_input(image, ellfit)
	image = augment_image(image, input_size)
	return image, ellfit

# Reads and augments the image and ellipse regression values
def read_augment_image_and_ellreg_v2(input_queue, input_size, max_trans=120.0, max_rot=45.0):
	ellfit = read_ellipse(input_queue[1])
	image = read_image(input_queue[0], input_size)
	image, ellfit, _ = rand_flip_input(image, ellfit)
	image = augment_image(image, input_size)
	image, ellfit, _ = shift_augment(image, ellfit, max_trans=max_trans, max_rot=max_rot)
	return image, ellfit

# Reads all 3
def read_image_and_seg_and_ellreg(input_queue, input_size):
	image = read_image(input_queue[0], input_size)
	seg = read_image(input_queue[1], input_size)
	ellfit = read_ellipse(input_queue[2])
	return image, seg, ellfit

# Reads all 3 + augmentation
def read_augment_image_and_seg_and_ellreg(input_queue, input_size, max_trans=120.0, max_rot=45.0):
	image, seg, ellfit = read_image_and_seg_and_ellreg(input_queue, input_size)
	image, ellfit, seg = rand_flip_input(image, ellfit, seg=seg)
	image, ellfit, seg = shift_augment(image, ellfit, seg, max_trans=max_trans, max_rot=max_rot)
	image = augment_image(image, input_size)
	return image, seg, ellfit

################################################
# Prepares the datasets into two callable generator tensors
# Returns image, label tensor generators
def get_train_batch_ellreg(dataset, read_threads, batch_size, input_size):
	inputs = dataset.train_images
	inputs2 = dataset.train_labels
	input_queue = tf.train.slice_input_producer([inputs, inputs2], shuffle=True)
	example_list = [read_augment_image_and_ellreg(input_queue, input_size) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[6]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, label_batch

# Prepares the datasets into two callable generator tensors
# Returns image, label tensor generators
# Uses affine transformation augmentation
def get_train_batch_ellreg_v2(dataset, read_threads, batch_size, input_size, max_rot = 45., max_trans = 120.):
	inputs = dataset.train_images
	inputs2 = dataset.train_labels
	input_queue = tf.train.slice_input_producer([inputs, inputs2], shuffle=True)
	example_list = [read_augment_image_and_ellreg_v2(input_queue, input_size, max_rot = max_rot, max_trans = max_trans) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[6]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, label_batch

# Prepares the datasets into two callable generator tensors
# Returns image, label tensor generators
def get_eval_batch_ellreg(dataset, read_threads, batch_size, input_size):
	inputs = dataset.valid_images
	inputs2 = dataset.valid_labels
	input_queue = tf.train.slice_input_producer([inputs, inputs2], shuffle=True)
	example_list = [read_image_and_ellreg(input_queue, input_size) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[6]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, label_batch

# Prepares the datasets into two callable generator tensors
# Returns image, label tensor generators
def get_train_batch_segellreg(dataset, read_threads, batch_size, input_size, max_rot = 45., max_trans = 120.):
	inputs = dataset.train_images
	inputs2 = dataset.train_seg
	inputs3 = dataset.train_labels
	input_queue = tf.train.slice_input_producer([inputs, inputs2, inputs3], shuffle=True)
	example_list = [read_augment_image_and_seg_and_ellreg(input_queue, input_size, max_trans, max_rot) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[input_size,input_size,1],[6]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, seg_batch, ellfit_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, seg_batch, ellfit_batch

def get_valid_batch_segellreg(dataset, read_threads, batch_size, input_size):
	inputs = dataset.valid_images
	inputs2 = dataset.valid_seg
	inputs3 = dataset.valid_labels
	input_queue = tf.train.slice_input_producer([inputs, inputs2, inputs3], shuffle=True)
	example_list = [read_image_and_seg_and_ellreg(input_queue, input_size) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[input_size,input_size,1],[6]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, seg_batch, ellfit_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, seg_batch, ellfit_batch

# Prepares the datasets into two callable generator tensors
# Returns image, label tensor generators
def get_train_batch_seg(dataset, read_threads, batch_size, input_size, max_trans = 0.0, max_rot = 0.0):
	inputs = dataset.train_images
	inputs2 = dataset.train_seg
	input_queue = tf.train.slice_input_producer([inputs, inputs2], shuffle=True)
	example_list = [read_augment_image_and_seg(input_queue, input_size, max_trans = max_trans, max_rot = max_rot) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[input_size,input_size,1]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, label_batch

def get_eval_batch_seg(dataset, read_threads, batch_size, input_size):
	inputs = dataset.valid_images
	inputs2 = dataset.valid_seg
	input_queue = tf.train.slice_input_producer([inputs, inputs2], shuffle=True)
	example_list = [read_image_and_seg(input_queue, input_size) for _ in range(read_threads)]
	shapes = [[input_size,input_size,1],[input_size,input_size,1]]
	min_after_dequeue = 100 # Always have 100 extra in the queue
	capacity = min_after_dequeue + 5 * batch_size
	image_batch, label_batch = tf.train.shuffle_batch_join(example_list, batch_size=batch_size, shapes=shapes, capacity=capacity, min_after_dequeue=min_after_dequeue)
	return image_batch, label_batch

# Convert ellreg input to a one-hot x/y input
def ellreg_to_xyhot(ellreg, nbins = 4800, bins2px = 10):
	label = tf.add(tf.multiply(ellreg, scale), means)
	xhot = tf.one_hot(tf.to_int32(label[:,0]*bins2px), nbins)
	yhot = tf.one_hot(tf.to_int32(label[:,1]*bins2px), nbins)
	return xhot, yhot


