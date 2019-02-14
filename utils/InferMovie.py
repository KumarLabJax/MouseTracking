import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import time
from datetime import datetime
import sys
from .plotters import *
from .readers import *
from .transformer import *
from .models import *
from .datasets import *
from .training import *
import imageio
from time import time

# Processes the input_movie using the network
# Ellreg-based net
def processMovie(input_movie, network, outputs):
	# Setup some non-modifiable values...
	ellfit_movie_append = '_ellfit'
	affine_movie_append = '_affine'
	crop_movie_append = '_crop'
	ellfit_output_append =  '_ellfit'
	ellfit_feature_outputs_append = '_features'

	# Set up the output streams...
	stillReading = False
	reader = imageio.get_reader(input_movie)
	if outputs['ell_mov']:
		writer_ellfit = imageio.get_writer(input_movie[:-4]+ellfit_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['aff_mov']:
		writer_affine = imageio.get_writer(input_movie[:-4]+affine_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['crop_mov']:
		writer_crop = imageio.get_writer(input_movie[:-4]+crop_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['ell_file']:
		file_ellfit = open(input_movie[:-4]+ellfit_output_append+'.npz', 'wb')
		stillReading = True
	if outputs['ell_features']:
		file_features = open(input_movie[:-4]+ellfit_feature_outputs_append+'.npz', 'wb')
		stillReading = True

	# Start processing the data
	im_iter = reader.iter_data()
	framenum = 0
	while(stillReading):
		start_time = time()
		frames = []
		framenum = framenum + 1 * network['batch_size']
		if framenum % 1000 == 0:
			print("Frame: " + str(framenum))
		for i in range(network['batch_size']):
			try:
				#frame = cv2.cvtColor(np.uint8(next(im_iter)), cv2.COLOR_BGR2GRAY)
				frame = np.uint8(next(im_iter))
				#frames.append(np.resize(frame, (network['input_size'], network['input_size'], 1)))
				frames.append(np.resize(frame[:,:,1], (network['input_size'], network['input_size'], 1)))
			except StopIteration:
				stillReading = False
				break
			except RuntimeError:
				stillReading = False
				break
		if framenum % 1000 == 0:
			print('Batch Assembled in: ' + str(time()-start_time))
		start_time = time()
		if stillReading:
			if outputs['ell_features']:
				result, result_unscaled, features = network['sess'].run(fetches=[network['network_eval_batch'], network['ellfit'], network['final_features']], feed_dict={network['image_placeholder']: frames, network['is_training']: False})
			else:
				result, result_unscaled = network['sess'].run(fetches=[network['network_eval_batch'], network['ellfit']], feed_dict={network['image_placeholder']: frames, network['is_training']: False})
			if framenum % 1000 == 0:
				print('Batch Processed in: ' + str(time()-start_time))
			start_time = time()
			# Sequentially save the data
			for i in range(network['batch_size']):
				# Save the outputs and save only if the outfile pattern was identified
				if outputs['ell_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					result_temp = plot_ellipse(plot, result[i], (255, 0, 0))
					writer_ellfit.append_data(result_temp.astype('u1'))
				if outputs['aff_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					angle = np.arctan2(result_unscaled[i,5],result_unscaled[i,4])*180/np.pi
					affine_mat = np.float32([[1,0,-result_unscaled[i,0]+outputs['affine_crop_dim']],[0,1,-result_unscaled[i,1]+outputs['affine_crop_dim']]])
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = cv2.getRotationMatrix2D((outputs['affine_crop_dim'],outputs['affine_crop_dim']),angle,1.);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = np.float32([[1,0,-outputs['affine_crop_dim']/2],[0,1,-outputs['affine_crop_dim']/2]]);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim'],outputs['affine_crop_dim']));
					writer_affine.append_data(plot.astype('u1'))
				if outputs['crop_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					angle = 0
					affine_mat = np.float32([[1,0,-result_unscaled[i,0]+outputs['affine_crop_dim']],[0,1,-result_unscaled[i,1]+outputs['affine_crop_dim']]])
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = cv2.getRotationMatrix2D((outputs['affine_crop_dim'],outputs['affine_crop_dim']),angle,1.);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = np.float32([[1,0,-outputs['affine_crop_dim']/2],[0,1,-outputs['affine_crop_dim']/2]]);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim'],outputs['affine_crop_dim']));
					writer_crop.append_data(plot.astype('u1'))
				if outputs['ell_file']:
					np.save(file_ellfit, result_unscaled[i,:], allow_pickle=False)
				if outputs['ell_features']:
					np.save(file_features, features[i,:], allow_pickle=False)
			if framenum % 1000 == 0:
				print('Batch Saved in: ' + str(time()-start_time))

	if outputs['ell_file']:
		file_ellfit.close()
	if outputs['ell_features']:
		file_features.close()


# Processes the input_movie using the network
# Segellreg-based net
def processMovie_v2(input_movie, network, outputs):
	# Setup some non-modifiable values...
	ellfit_movie_append = '_ellfit'
	affine_movie_append = '_affine'
	crop_movie_append = '_crop'
	ellfit_output_append =  '_ellfit'
	ellfit_feature_outputs_append = '_features'
	seg_movie_append = '_seg'

	# Set up the output streams...
	stillReading = False
	reader = imageio.get_reader(input_movie)
	if outputs['ell_mov']:
		writer_ellfit = imageio.get_writer(input_movie[:-4]+ellfit_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['aff_mov']:
		writer_affine = imageio.get_writer(input_movie[:-4]+affine_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['crop_mov']:
		writer_crop = imageio.get_writer(input_movie[:-4]+crop_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['ell_file']:
		file_ellfit = open(input_movie[:-4]+ellfit_output_append+'.npz', 'wb')
		stillReading = True
	if outputs['ell_features']:
		file_features = open(input_movie[:-4]+ellfit_feature_outputs_append+'.npz', 'wb')
		stillReading = True
	if outputs['seg_mov']:
		writer_seg = imageio.get_writer(input_movie[:-4]+seg_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True

	# Start processing the data
	im_iter = reader.iter_data()
	framenum = 0
	while(stillReading):
		start_time = time()
		frames = []
		framenum = framenum + 1 * network['batch_size']
		if framenum % 1000 == 0:
			print("Frame: " + str(framenum))
		for i in range(network['batch_size']):
			try:
				#frame = cv2.cvtColor(np.uint8(next(im_iter)), cv2.COLOR_BGR2GRAY)
				frame = np.uint8(next(im_iter))
				#frames.append(np.resize(frame, (network['input_size'], network['input_size'], 1)))
				frames.append(np.resize(frame[:,:,1], (network['input_size'], network['input_size'], 1)))
			except StopIteration:
				stillReading = False
				break
			except RuntimeError:
				stillReading = False
				break
		if framenum % 1000 == 0:
			print('Batch Assembled in: ' + str(time()-start_time))
		start_time = time()
		if stillReading:
			result, result_unscaled, result_seg = network['sess'].run(fetches=[network['network_eval_batch'], network['ellfit'], network['seg']], feed_dict={network['image_placeholder']: frames, network['is_training']: False})
			if framenum % 1000 == 0:
				print('Batch Processed in: ' + str(time()-start_time))
			start_time = time()
			# Sequentially save the data
			for i in range(network['batch_size']):
				# Save the outputs and save only if the outfile pattern was identified
				if outputs['ell_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					result_temp = plot_ellipse(plot, result[i], (255, 0, 0))
					writer_ellfit.append_data(result_temp.astype('u1'))
				if outputs['aff_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					angle = np.arctan2(result_unscaled[i,5],result_unscaled[i,4])*180/np.pi
					affine_mat = np.float32([[1,0,-result_unscaled[i,0]+outputs['affine_crop_dim']],[0,1,-result_unscaled[i,1]+outputs['affine_crop_dim']]])
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = cv2.getRotationMatrix2D((outputs['affine_crop_dim'],outputs['affine_crop_dim']),angle,1.);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = np.float32([[1,0,-outputs['affine_crop_dim']/2],[0,1,-outputs['affine_crop_dim']/2]]);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim'],outputs['affine_crop_dim']));
					writer_affine.append_data(plot.astype('u1'))
				if outputs['crop_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					angle = 0
					affine_mat = np.float32([[1,0,-result_unscaled[i,0]+outputs['affine_crop_dim']],[0,1,-result_unscaled[i,1]+outputs['affine_crop_dim']]])
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = cv2.getRotationMatrix2D((outputs['affine_crop_dim'],outputs['affine_crop_dim']),angle,1.);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = np.float32([[1,0,-outputs['affine_crop_dim']/2],[0,1,-outputs['affine_crop_dim']/2]]);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim'],outputs['affine_crop_dim']));
					writer_crop.append_data(plot.astype('u1'))
				if outputs['ell_file']:
					np.save(file_ellfit, result_unscaled[i,:], allow_pickle=False)
				if outputs['ell_features']:
					np.save(file_features, features[i,:], allow_pickle=False)
				if outputs['seg_mov']:
					seg_output = result_seg[i,:,:,:]
					seg_output = seg_output[:,:,0]/np.sum(seg_output,2)
					#seg_output = seg_output[:,:,0]/np.sum(seg_output,2)-seg_output[:,:,1]/np.sum(seg_output,2)
					#seg_output = seg_output+0.25
					#seg_output[seg_output<1e-6] = 0
					#seg_output[seg_output>1.0] = 1.0
					writer_seg.append_data((254*seg_output).astype('u1'))
			if framenum % 1000 == 0:
				print('Batch Saved in: ' + str(time()-start_time))

	if outputs['ell_file']:
		file_ellfit.close()
	if outputs['ell_features']:
		file_features.close()


# Processes the input_movie using the network
# Binned-based net
def processMovie_v3(input_movie, network, outputs):
	# Setup some non-modifiable values...
	ellfit_movie_append = '_xyplot'
	crop_movie_append = '_crop'

	# Set up the output streams...
	stillReading = False
	reader = imageio.get_reader(input_movie)
	if outputs['ell_mov']:
		writer_ellfit = imageio.get_writer(input_movie[:-4]+ellfit_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True
	if outputs['crop_mov']:
		writer_crop = imageio.get_writer(input_movie[:-4]+crop_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True

	# Start processing the data
	im_iter = reader.iter_data()
	framenum = 0
	while(stillReading):
		start_time = time()
		frames = []
		framenum = framenum + 1 * network['batch_size']
		if framenum % 1000 == 0:
			print("Frame: " + str(framenum))
		for i in range(network['batch_size']):
			try:
				#frame = cv2.cvtColor(np.uint8(next(im_iter)), cv2.COLOR_BGR2GRAY)
				frame = np.uint8(next(im_iter))
				#frames.append(np.resize(frame, (network['input_size'], network['input_size'], 1)))
				frames.append(np.resize(frame[:,:,1], (480, 480, 1)))
			except StopIteration:
				stillReading = False
				break
			except RuntimeError:
				stillReading = False
				break
		if framenum % 1000 == 0:
			print('Batch Assembled in: ' + str(time()-start_time))
		start_time = time()
		if stillReading:
			xhot, yhot = network['sess'].run(fetches=[network['xhot_est'], network['yhot_est']], feed_dict={network['image_placeholder']: frames, network['is_training']: False})
			if framenum % 1000 == 0:
				print('Batch Processed in: ' + str(time()-start_time))
			start_time = time()
			# Sequentially save the data
			for i in range(network['batch_size']):
				if outputs['ell_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					# Place crosshair on predicted location...
					cv2.line(plot,(np.float32(np.argmax(xhot,1)[i]/network['bin_per_px']-2), np.float32(np.argmax(yhot,1)[i]/network['bin_per_px'])),(np.float32(np.argmax(xhot,1)[i]/network['bin_per_px']+2), np.float32(np.argmax(yhot,1)[i]/network['bin_per_px'])), (255, 0, 0))
					cv2.line(plot,(np.float32(np.argmax(xhot,1)[i]/network['bin_per_px']), np.float32(np.argmax(yhot,1)[i]/network['bin_per_px']-2)),(np.float32(np.argmax(xhot,1)[i]/network['bin_per_px']), np.float32(np.argmax(yhot,1)[i]/network['bin_per_px']+2)), (255, 0, 0))
					writer_ellfit.append_data(plot.astype('u1'))
				if outputs['crop_mov']:
					plot = cv2.cvtColor(frames[i],cv2.COLOR_GRAY2RGB)
					angle = 0
					affine_mat = np.float32([[1,0,-np.argmax(xhot,1)[i]/network['bin_per_px']+outputs['affine_crop_dim']],[0,1,-np.argmax(yhot,1)[i]/network['bin_per_px']+outputs['affine_crop_dim']]])
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = cv2.getRotationMatrix2D((outputs['affine_crop_dim'],outputs['affine_crop_dim']),angle,1.);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim']*2,outputs['affine_crop_dim']*2));
					affine_mat = np.float32([[1,0,-outputs['affine_crop_dim']/2],[0,1,-outputs['affine_crop_dim']/2]]);
					plot = cv2.warpAffine(plot, affine_mat, (outputs['affine_crop_dim'],outputs['affine_crop_dim']));
					writer_crop.append_data(plot.astype('u1'))
			if framenum % 1000 == 0:
				print('Batch Saved in: ' + str(time()-start_time))

# Processes the input_movie using the network
# Segmentation ONLY based net
def processSegSoftMovie(input_movie, network, outputs):
	# Setup some non-modifiable values...
	seg_movie_append = '_seg'

	# Set up the output streams...
	stillReading = False
	reader = imageio.get_reader(input_movie)
	if outputs['seg_mov']:
		writer_seg = imageio.get_writer(input_movie[:-4]+seg_movie_append+'.avi', fps=reader.get_meta_data()['fps'], codec='mpeg4', quality=10)
		stillReading = True

	# Start processing the data
	im_iter = reader.iter_data()
	framenum = 0
	while(stillReading):
		start_time = time()
		frames = []
		framenum = framenum + 1 * network['batch_size']
		if framenum % 1000 == 0:
			print("Frame: " + str(framenum))
		for i in range(network['batch_size']):
			try:
				frame = np.uint8(next(im_iter))
				frames.append(np.resize(frame[:,:,1], (network['input_size'], network['input_size'], 1)))
			except StopIteration:
				stillReading = False
				break
			except RuntimeError:
				stillReading = False
				break
		if framenum % 1000 == 0:
			print('Batch Assembled in: ' + str(time()-start_time))
		start_time = time()
		if stillReading:
			result_seg = network['sess'].run(fetches=[network['seg']], feed_dict={network['image_placeholder']: frames, network['is_training']: False})[0]
			if framenum % 1000 == 0:
				print('Batch Processed in: ' + str(time()-start_time))
			start_time = time()
			# Sequentially save the data
			for i in range(network['batch_size']):
				# Save the outputs and save only if the outfile pattern was identified
				if outputs['seg_mov']:
					seg_output = result_seg[i,:,:]
					writer_seg.append_data((254*seg_output[:,:]).astype('u1'))
			if framenum % 1000 == 0:
				print('Batch Saved in: ' + str(time()-start_time))


def inferEllregNetwork(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, final_features = arg_dict['model_construct_function'](image_placeholder, is_training)
		ellfit = tf.add(tf.multiply(network_eval_batch, scale), means)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'network_eval_batch':network_eval_batch, 'ellfit':ellfit, 'final_features':final_features, 'image_placeholder':image_placeholder, 'is_training':is_training}
	outputs = {'ell_mov':arg_dict['ellfit_movie_output'], 'aff_mov':arg_dict['affine_movie_output'], 'crop_mov':arg_dict['crop_movie_output'], 'ell_file':arg_dict['ellfit_output'], 'ell_features':arg_dict['ellfit_features_output'], 'affine_crop_dim':arg_dict['affine_crop_dim']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))
	processMovie(arg_dict['input_movie'], network, outputs)

def inferEllregNetwork_Loop(arg_dict):
	start_time = time()
	sess = tf.Session()
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, final_features = arg_dict['model_construct_function'](image_placeholder, is_training)
		ellfit = tf.add(tf.multiply(network_eval_batch, scale), means)
	global_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'network_eval_batch':network_eval_batch, 'ellfit':ellfit, 'final_features':final_features, 'image_placeholder':image_placeholder, 'is_training':is_training}
	outputs = {'ell_mov':arg_dict['ellfit_movie_output'], 'aff_mov':arg_dict['affine_movie_output'], 'crop_mov':arg_dict['crop_movie_output'], 'ell_file':arg_dict['ellfit_output'], 'ell_features':arg_dict['ellfit_features_output'], 'affine_crop_dim':arg_dict['affine_crop_dim']}

	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))

	# Process multiple movies
	f = open(arg_dict['input_movie_list'])
	lines = f.read().split('\n')
	lines = lines[0:-1] # Remove the last split '' string
	for input_movie in lines:
		processMovie(input_movie, network, outputs)

def inferSegEllregNetwork(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		seg_eval_batch, network_eval_batch, _ = arg_dict['model_construct_function'](image_placeholder, is_training)
		ellfit = tf.add(tf.multiply(network_eval_batch, scale), means)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	# Force never to save the features...
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'network_eval_batch':network_eval_batch, 'ellfit':ellfit, 'final_features':seg_eval_batch, 'image_placeholder':image_placeholder, 'is_training':is_training, 'seg':seg_eval_batch}
	outputs = {'ell_mov':arg_dict['ellfit_movie_output'], 'aff_mov':arg_dict['affine_movie_output'], 'crop_mov':arg_dict['crop_movie_output'], 'ell_file':arg_dict['ellfit_output'], 'ell_features':False, 'affine_crop_dim':arg_dict['affine_crop_dim'], 'seg_mov':arg_dict['seg_movie_output']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))
	print('Processing ' + arg_dict['input_movie'])
	processMovie_v2(arg_dict['input_movie'], network, outputs)

def inferSegEllregNetwork_loop(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		seg_eval_batch, network_eval_batch, _ = arg_dict['model_construct_function'](image_placeholder, is_training)
		ellfit = tf.add(tf.multiply(network_eval_batch, scale), means)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	# Force never to save the features...
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'network_eval_batch':network_eval_batch, 'ellfit':ellfit, 'final_features':seg_eval_batch, 'image_placeholder':image_placeholder, 'is_training':is_training, 'seg':seg_eval_batch}
	outputs = {'ell_mov':arg_dict['ellfit_movie_output'], 'aff_mov':arg_dict['affine_movie_output'], 'crop_mov':arg_dict['crop_movie_output'], 'ell_file':arg_dict['ellfit_output'], 'ell_features':False, 'affine_crop_dim':arg_dict['affine_crop_dim'], 'seg_mov':arg_dict['seg_movie_output']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))

	f = open(arg_dict['input_movie_list'])
	lines = f.read().split('\n')
	lines = lines[0:-1] # Remove the last split '' string
	for input_movie in lines:
		processMovie_v2(input_movie, network, outputs)


def inferBinnedNetwork(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, _ = arg_dict['model_construct_function'](image_placeholder, is_training, int(arg_dict['input_size']*arg_dict['bin_per_px']))
		xhot_est, yhot_est = tf.unstack(network_eval_batch)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'bin_per_px':arg_dict['bin_per_px'], 'image_placeholder':image_placeholder, 'is_training':is_training, 'xhot_est':xhot_est, 'yhot_est':yhot_est}
	outputs = {'ell_mov':arg_dict['ellfit_movie_output'], 'crop_mov':arg_dict['crop_movie_output'], 'affine_crop_dim':arg_dict['affine_crop_dim']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))

	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))
	print('Processing ' + arg_dict['input_movie'])
	processMovie_v3(arg_dict['input_movie'], network, outputs)

def inferBinnedNetwork_loop(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, _ = arg_dict['model_construct_function'](image_placeholder, is_training, int(arg_dict['input_size']*arg_dict['bin_per_px']))
		xhot_est, yhot_est = tf.unstack(network_eval_batch)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'bin_per_px':arg_dict['bin_per_px'], 'image_placeholder':image_placeholder, 'is_training':is_training, 'xhot_est':xhot_est, 'yhot_est':yhot_est}
	outputs = {'ell_mov':arg_dict['ellfit_movie_output'], 'crop_mov':arg_dict['crop_movie_output'], 'affine_crop_dim':arg_dict['affine_crop_dim']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))

	f = open(arg_dict['input_movie_list'])
	lines = f.read().split('\n')
	lines = lines[0:-1] # Remove the last split '' string
	for input_movie in lines:
		processMovie_v3(input_movie, network, outputs)


def inferSegSoftNetwork(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		seg_eval_batch = arg_dict['model_construct_function'](image_placeholder, is_training)
		seg_eval_batch = tf.nn.softmax(seg_eval_batch)[:,:,:,0] # Only grab the "Mouse"
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	# Force never to save the features...
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'image_placeholder':image_placeholder, 'is_training':is_training, 'seg':seg_eval_batch}
	outputs = {'seg_mov':arg_dict['seg_movie_output']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))

	processSegSoftMovie(arg_dict['input_movie'], network, outputs)

def inferSegSoftNetwork_loop(arg_dict):
	start_time = time()
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		seg_eval_batch = arg_dict['model_construct_function'](image_placeholder, is_training)
		seg_eval_batch = tf.nn.softmax(seg_eval_batch)[:,:,:,0] # Only grab the "Mouse"
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])

	# Pack the parameters into a dictionary
	# Force never to save the features...
	network = {'sess':sess, 'batch_size':arg_dict['batch_size'], 'input_size':arg_dict['input_size'], 'image_placeholder':image_placeholder, 'is_training':is_training, 'seg':seg_eval_batch}
	outputs = {'seg_mov':arg_dict['seg_movie_output']}
	# Process a single movie
	time_duration = time()-start_time
	print('Initializing Network Duration: ' + str(time_duration))

	f = open(arg_dict['input_movie_list'])
	lines = f.read().split('\n')
	lines = lines[0:-1] # Remove the last split '' string
	for input_movie in lines:
		processSegSoftMovie(input_movie, network, outputs)


# Parses the argument dictionary to select the correct processing functions
def inferMovie(arg_dict):
	if arg_dict['net_type'] == 'segellreg':
		if 'input_movie_list' in arg_dict.keys():
			inferSegEllregNetwork_loop(arg_dict)
		else:
			inferSegEllregNetwork(arg_dict)
	elif arg_dict['net_type'] == 'ellreg':
		if 'input_movie_list' in arg_dict.keys():
			inferEllregNetwork_Loop(arg_dict)
		else:
			inferEllregNetwork(arg_dict)
	elif arg_dict['net_type'] == 'binned':
		if 'input_movie_list' in arg_dict.keys():
			inferBinnedNetwork_Loop(arg_dict)
		else:
			inferBinnedNetwork(arg_dict)
	elif arg_dict['net_type'] == 'seg':
		if 'input_movie_list' in arg_dict.keys():
			inferSegSoftNetwork_loop(arg_dict)
		else:
			inferSegSoftNetwork(arg_dict)
