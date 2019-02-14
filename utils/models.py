# My collection of available network models...

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.slim.nets import vgg
from .readers import means, scale, atan2
import scipy.ndimage.morphology as morph
import numpy as np

# Concats x/y gradients along the depth dimension
# Used in coordconvs
# Note: You should call with tf.map_fn(lambda by_batch: concat_xygrad_2d(by_batch), input_tensor)
def concat_xygrad_2d(input_tensor):
	input_shape = [int(x) for i,x in enumerate(input_tensor.get_shape())]
	xgrad = tf.reshape(tf.tile([tf.lin_space(0.0,1.0,input_shape[-2])],[input_shape[-3],1]),np.concatenate([input_shape[0:-1],[1]]))
	ygrad = tf.reshape(tf.tile(tf.reshape([tf.lin_space(0.0,1.0,input_shape[-3])],[input_shape[-3],1]),[1,input_shape[-2]]),np.concatenate([input_shape[0:-1],[1]]))
	return tf.concat([input_tensor, xgrad, ygrad], axis=-1)

# Fits an ellipse from a mask
# Assumes that the mask is of size [?,3], where [:,0] are x indices and [:,1] are y indices
def fitEll(mask):
	locs = tf.cast(tf.slice(mask,[0,0],[-1,2]),tf.float32)
	translations = tf.reduce_mean(locs, 0)
	sqlocs = tf.square(locs)
	variance = tf.reduce_mean(sqlocs,0)-tf.square(translations)
	variance_xy = tf.reduce_mean(tf.reduce_prod(locs, 1),0)-tf.reduce_prod(translations,0)
	translations = tf.reverse(translations,[0]) # Note: Moment across X-values gives you y location, so need to reverse
	tmp1 = tf.reduce_sum(variance)
	tmp2 = tf.sqrt(tf.multiply(4.0,tf.pow(variance_xy,2))+tf.pow(tf.reduce_sum(tf.multiply(variance,[1.0,-1.0])),2))
	eigA = tf.multiply(tf.sqrt((tmp1+tmp2)/2.0),4.0)
	eigB = tf.multiply(tf.sqrt((tmp1-tmp2)/2.0),4.0)
	angle = 0.5*atan2(2.0*variance_xy,tf.reduce_sum(tf.multiply(variance,[1.0,-1.0]))) # Radians
	ellfit = tf.stack([tf.slice(translations,[0],[1]),tf.slice(translations,[1],[1]),[eigB],[eigA],[tf.sin(angle)],[tf.cos(angle)]],1)
	return tf.reshape(tf.divide(tf.subtract(ellfit,means),scale),[-1])

# It appears that the issue for running this is due to nested loops in the optimizer (cannot train).
# https://github.com/tensorflow/tensorflow/issues/3726
# Both tf.where and tf.gather_nd use loops
# This can be used during inference to get slightly better results (by changing the line in the  fitEllFromSeg definition).
def fitEll_weighted(mask, seg):
	locs_orig = tf.cast(tf.slice(mask,[0,0],[-1,2]),tf.float32)
	weights = tf.gather_nd(seg, mask)
	# Normalize to sum of 1
	weights_orig = tf.exp(tf.divide(weights,tf.reduce_sum(weights)))
	weights_orig = tf.divide(weights_orig,tf.reduce_sum(weights_orig))
	weights = tf.reshape(tf.tile(weights_orig,[2]),[-1,2])
	# This is the line that breaks it:
	locs = tf.multiply(locs_orig,weights)
	translations = tf.reduce_sum(locs, 0) # Note: Moment across X-values gives you y location, so need to reverse. This is changed on the return values (index 1, then index 0)
	sqlocs = tf.multiply(tf.square(locs_orig),weights)
	variance = tf.reduce_sum(sqlocs,0)-tf.square(translations)
	variance_xy = tf.reduce_sum(tf.reduce_prod(locs_orig, 1)*weights_orig,0)-tf.reduce_prod(translations,0)
	tmp1 = tf.reduce_sum(variance)
	tmp2 = tf.sqrt(tf.multiply(4.0,tf.pow(variance_xy,2))+tf.pow(tf.reduce_sum(tf.multiply(variance,[1.0,-1.0])),2))
	eigA = tf.multiply(tf.sqrt((tmp1+tmp2)/2.0),4.0)
	eigB = tf.multiply(tf.sqrt((tmp1-tmp2)/2.0),4.0)
	angle = 0.5*atan2(2.0*variance_xy,tf.reduce_sum(tf.multiply(variance,[1.0,-1.0]))) # Radians
	ellfit = tf.stack([tf.slice(translations,[1],[1]),tf.slice(translations,[0],[1]),[eigB],[eigA],[tf.sin(angle)],[tf.cos(angle)]],1)
	return tf.reshape(tf.divide(tf.subtract(ellfit,means),scale),[-1])

# Safely applies the threshold to the mask and returns default values if no indices are classified as mouse
def fitEllFromSeg(seg, node_act):
	mask = tf.where(tf.greater(seg,node_act))
	# NOTE: See note on fitEll_weighted function definition
	#return tf.cond(tf.shape(mask)[0]>0, lambda: fitEll_weighted(mask, seg), lambda: tf.to_float([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]))
	return tf.cond(tf.shape(mask)[0]>0, lambda: fitEll(mask), lambda: tf.to_float([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]))


##########################################################################
# Begin defining all available models
##########################################################################
def construct_segellreg_v8(images, is_training):
	batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'updates_collections': None, 'center': True, 'scale': True, 'trainable': True}
	# Normalize the image inputs (map_fn used to do a "per batch" calculation)
	norm_imgs = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
	kern_size = [5,5]
	filter_size = 8
	with tf.variable_scope('SegmentEncoder'):
		with slim.arg_scope([slim.conv2d],
							activation_fn=tf.nn.relu,
							padding='SAME',
							weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
							weights_regularizer=slim.l2_regularizer(0.0005),
							normalizer_fn=slim.batch_norm,
							normalizer_params=batch_norm_params):
			c1 = slim.conv2d(norm_imgs, filter_size, kern_size)
			p1 = slim.max_pool2d(c1, [2,2], scope='pool1') #240x240
			c2 = slim.conv2d(p1, filter_size*2, kern_size)
			p2 = slim.max_pool2d(c2, [2,2], scope='pool2') #120x120
			c3 = slim.conv2d(p2, filter_size*4, kern_size)
			p3 = slim.max_pool2d(c3, [2,2], scope='pool3') #60x60
			c4 = slim.conv2d(p3, filter_size*8, kern_size)
			p4 = slim.max_pool2d(c4, [2,2], scope='pool4') # 30x30
			c5 = slim.conv2d(p4, filter_size*16, kern_size)
			p5 = slim.max_pool2d(c5, [2,2], scope='pool5') # 15x15
			c6 = slim.conv2d(p5, filter_size*32, kern_size)
			p6 = slim.max_pool2d(c6, [3,3], stride=3, scope='pool6') # 5x5
			c7 = slim.conv2d(p6, filter_size*64, kern_size)
	with tf.variable_scope('SegmentDecoder'):
		upscale = 2 # Undo the pools once at a time
		mynet = slim.conv2d_transpose(c7, filter_size*32, kern_size, stride=[3, 3], activation_fn=None)
		mynet = tf.add(mynet, c6)
		mynet = slim.conv2d_transpose(mynet, filter_size*16, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c5)
		mynet = slim.conv2d_transpose(mynet, filter_size*8, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c4)
		mynet = slim.conv2d_transpose(mynet, filter_size*4, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c3)
		mynet = slim.conv2d_transpose(mynet, filter_size*2, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c2)
		mynet = slim.conv2d_transpose(mynet, filter_size, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c1)
		seg = slim.conv2d(mynet, 2, [1,1], scope='seg')
	with tf.variable_scope('Ellfit'):
		seg_morph = tf.slice(tf.nn.softmax(seg,-1),[0,0,0,0],[-1,-1,-1,1])-tf.slice(tf.nn.softmax(seg,-1),[0,0,0,1],[-1,-1,-1,1])
		# And was kept here to just assist in the ellipse-fit for any unwanted noise
		filter1 = tf.expand_dims(tf.constant(morph.iterate_structure(morph.generate_binary_structure(2,1),4),dtype=tf.float32),-1)
		seg_morph = tf.nn.dilation2d(tf.nn.erosion2d(seg_morph,filter1,[1,1,1,1],[1,1,1,1],"SAME"),filter1,[1,1,1,1],[1,1,1,1],"SAME")
		filter2 = tf.expand_dims(tf.constant(morph.iterate_structure(morph.generate_binary_structure(2,1),5),dtype=tf.float32),-1)
		seg_morph = tf.nn.erosion2d(tf.nn.dilation2d(seg_morph,filter2,[1,1,1,1],[1,1,1,1],"SAME"),filter2,[1,1,1,1],[1,1,1,1],"SAME")
		node_act = tf.constant(0.0,dtype=tf.float32)
		# Fit the ellipse from the segmentation mask algorithmically
		ellfit = tf.map_fn(lambda mask: fitEllFromSeg(mask, node_act), seg_morph)
	with tf.variable_scope('AngleFix'):
		mynet = slim.conv2d(c7, 128, kern_size, activation_fn=tf.nn.relu, padding='SAME', weights_initializer=tf.truncated_normal_initializer(0.0, 0.01), weights_regularizer=slim.l2_regularizer(0.0005), normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
		mynet = slim.conv2d(mynet, 64, kern_size, activation_fn=tf.nn.relu, padding='SAME', weights_initializer=tf.truncated_normal_initializer(0.0, 0.01), weights_regularizer=slim.l2_regularizer(0.0005), normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params)
		mynet = slim.flatten(mynet)
		angle_bins = slim.fully_connected(mynet, 4, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='angle_bin')
		angles = tf.add(tf.multiply(tf.slice(ellfit, [0,4], [-1,2]), scale[4:5]), means[4:5]) # Extract angles to fix them
		sin_angles = tf.slice(angles,[0,0],[-1,1]) # Unmorph the sin(angles)
		ang_bins_max = tf.argmax(angle_bins,1) # Note: This is from 0-3, not 1-4
		angles = tf.where(tf.equal(ang_bins_max,2), -angles, angles) # Bin 3 always wrong
		angles = tf.where(tf.logical_and(tf.equal(ang_bins_max,1), tf.squeeze(tf.less(sin_angles, 0.0))), -angles, angles) # Bin 2 is wrong when sin(ang) < np.sin(np.pi/4.) ... Some bleedover, so < 0.0
		angles = tf.where(tf.logical_and(tf.equal(ang_bins_max,3), tf.squeeze(tf.greater(sin_angles, 0.0))), -angles, angles) # Bin 4 is wrong when sin(ang) > -np.sin(np.pi/4.) ... Some bleedover, so > 0.0
		angles = tf.divide(tf.subtract(angles, means[4:5]), scale[4:5])
		original = tf.slice(ellfit,[0,0],[-1,4])
		ellfit = tf.concat([original, angles],1)

	return seg, ellfit, angle_bins


# XY binning for 480 x and 480 y bins
def construct_xybin_v1(images, is_training, n_bins):
	batch_norm_params = {'is_training': is_training, 'decay': 0.8, 'updates_collections': None, 'center': True, 'scale': True, 'trainable': True}
	with slim.arg_scope([slim.conv2d],
						activation_fn=tf.nn.relu,
						padding='SAME',
						weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
						weights_regularizer=slim.l2_regularizer(0.0005),
						normalizer_fn=slim.batch_norm,
						normalizer_params=batch_norm_params):
		mynet = slim.repeat(images, 2, slim.conv2d, 16, [3,3], scope='conv1')
		mynet = slim.max_pool2d(mynet, [2,2], scope='pool1')
		mynet = slim.repeat(mynet, 2, slim.conv2d, 32, [3,3], scope='conv2')
		mynet = slim.max_pool2d(mynet, [2,2], scope='pool2')
		mynet = slim.repeat(mynet, 2, slim.conv2d, 64, [3,3], scope='conv3')
		mynet = slim.max_pool2d(mynet, [2,2], scope='pool3')
		mynet = slim.repeat(mynet, 2, slim.conv2d, 128, [3,3], scope='conv4')
		mynet = slim.max_pool2d(mynet, [2,2], scope='pool4')
		mynet = slim.repeat(mynet, 2, slim.conv2d, 256, [3,3], scope='conv5')
		mynet = slim.max_pool2d(mynet, [2,2], scope='pool5')
		features = slim.flatten(mynet, scope='flatten')
	with slim.arg_scope([slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
						weights_regularizer=slim.l2_regularizer(0.0005),
						normalizer_fn=slim.batch_norm,
						normalizer_params=batch_norm_params):
		# To add additional fully connected layers...
		# Our tests showed no substantial difference
		#mynet = slim.fully_connected(mynet, 4096, scope='fc5')
		#mynet = slim.dropout(mynet, 0.5, scope='dropout5')
		#mynet = slim.fully_connected(mynet, 4096, scope='fc6')
		#mynet = slim.dropout(mynet, 0.5, scope='dropout6')
		xbins = slim.fully_connected(features, n_bins, activation_fn=None, scope='xbins')
		xbins = slim.softmax(xbins, scope='smx')
		ybins = slim.fully_connected(features, n_bins, activation_fn=None, scope='ybins')
		ybins = slim.softmax(ybins, scope='smy')
		mynet = tf.stack([xbins, ybins])
	return mynet, features

# Attempt to predict the ellipse-regression directly (using resnet_v2_200)
def construct_ellreg_v3_resnet(images, is_training):
	batch_norm_params = {'is_training': is_training, 'decay': 0.8, 'updates_collections': None, 'center': True, 'scale': True, 'trainable': True}
	mynet, _ = resnet_v2.resnet_v2_200(images, None, is_training=is_training)
	features = tf.reshape(mynet, [-1, 2048])
	with slim.arg_scope([slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
						weights_regularizer=slim.l2_regularizer(0.0005),
						normalizer_fn=slim.batch_norm,
						normalizer_params=batch_norm_params):
		mynet = slim.fully_connected(features, 6, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='outlayer')
	return mynet, features

# Attempt to predict the ellipse-regression directly with coordinate convs
def construct_ellreg_v4_resnet(images, is_training):
	batch_norm_params = {'is_training': is_training, 'decay': 0.8, 'updates_collections': None, 'center': True, 'scale': True, 'trainable': True}
	input_imgs = tf.map_fn(lambda by_batch: concat_xygrad_2d(by_batch), images)
	mynet, _ = resnet_v2.resnet_v2_200(input_imgs, None, is_training=is_training)
	features = tf.reshape(mynet, [-1, 2048])
	with slim.arg_scope([slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
						weights_regularizer=slim.l2_regularizer(0.0005),
						normalizer_fn=slim.batch_norm,
						normalizer_params=batch_norm_params):
		mynet = slim.fully_connected(features, 6, activation_fn=None, normalizer_fn=None, normalizer_params=None, scope='outlayer')
	return mynet, features


# Segmentation Only Network (no angle prediction)
def construct_segsoft_v5(images, is_training):
	batch_norm_params = {'is_training': is_training, 'decay': 0.999, 'updates_collections': None, 'center': True, 'scale': True, 'trainable': True}
	# Normalize the image inputs (map_fn used to do a "per batch" calculation)
	norm_imgs = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)
	kern_size = [5,5]
	filter_size = 8
	# Run the segmentation net without pooling
	with tf.variable_scope('SegmentEncoder'):
		with slim.arg_scope([slim.conv2d],
							activation_fn=tf.nn.relu,
							padding='SAME',
							weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
							weights_regularizer=slim.l2_regularizer(0.0005),
							normalizer_fn=slim.batch_norm,
							normalizer_params=batch_norm_params):
			c1 = slim.conv2d(norm_imgs, filter_size, kern_size)
			p1 = slim.max_pool2d(c1, [2,2], scope='pool1') #240x240
			c2 = slim.conv2d(p1, filter_size*2, kern_size)
			p2 = slim.max_pool2d(c2, [2,2], scope='pool2') #120x120
			c3 = slim.conv2d(p2, filter_size*4, kern_size)
			p3 = slim.max_pool2d(c3, [2,2], scope='pool3') #60x60
			c4 = slim.conv2d(p3, filter_size*8, kern_size)
			p4 = slim.max_pool2d(c4, [2,2], scope='pool4') # 30x30
			c5 = slim.conv2d(p4, filter_size*16, kern_size)
			p5 = slim.max_pool2d(c5, [2,2], scope='pool5') # 15x15
			c6 = slim.conv2d(p5, filter_size*32, kern_size)
			p6 = slim.max_pool2d(c6, [3,3], stride=3, scope='pool6') # 5x5
			c7 = slim.conv2d(p6, filter_size*64, kern_size)
	with tf.variable_scope('SegmentDecoder'):
		upscale = 2 # Undo the pools once at a time
		mynet = slim.conv2d_transpose(c7, filter_size*32, kern_size, stride=[3, 3], activation_fn=None)
		mynet = tf.add(mynet, c6)
		mynet = slim.conv2d_transpose(mynet, filter_size*16, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c5)
		mynet = slim.conv2d_transpose(mynet, filter_size*8, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c4)
		mynet = slim.conv2d_transpose(mynet, filter_size*4, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c3)
		mynet = slim.conv2d_transpose(mynet, filter_size*2, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c2)
		mynet = slim.conv2d_transpose(mynet, filter_size, kern_size, stride=[upscale, upscale], activation_fn=None)
		mynet = tf.add(mynet, c1)
		seg = slim.conv2d(mynet, 2, [1,1], scope='seg')
	return seg
