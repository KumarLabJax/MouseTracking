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

def trainEllregNetwork(arg_dict):
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		label_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], 6])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, _ = arg_dict['model_construct_function'](image_placeholder, is_training)
	with tf.variable_scope('Loss'):
		print('Adding loss function...')
		loss, errors, angle_errs = gen_loss_ellreg(network_eval_batch, label_placeholder)
	##########################################
	with tf.variable_scope('Input_Decoding'):
		print('Populating input queues...')
		image_valid_batch, label_valid_batch = get_eval_batch_ellreg(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'])
		image_train_batch, label_train_batch = get_train_batch_ellreg_v2(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'], max_rot = arg_dict['aug_rot_max'], max_trans = arg_dict['aug_trans_max'])
		print('Starting input threads...')
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.variable_scope('Optimizer'):
		print('Initializing optimizer...')
		learn_rate, train_op = arg_dict['learn_function'](loss, arg_dict['dataset'].train_size, arg_dict['batch_size'], global_step, arg_dict['start_learn_rate'], arg_dict['epocs_per_lr_decay'], const_learn_rate=arg_dict['const_learn_rate'])
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		training_summary, validation_summary = gen_summary_ellreg(loss, errors, angle_errs, learn_rate)
		summary_writer = tf.summary.FileWriter(arg_dict['log_dir'], sess.graph)
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])
	print('Beginning training...')
	for step in range(0, arg_dict['num_steps']):
		start_time = time.time()
		img_batch, label_batch = sess.run([image_train_batch, label_train_batch])
		_, train_loss, summary_output, cur_step = sess.run(fetches=[train_op, loss, training_summary, global_step], feed_dict={image_placeholder: img_batch, label_placeholder: label_batch, is_training: True})
		duration = time.time() - start_time
		if (step+1) % 10 == 0: # CMDline updates every 10 steps
			examples_per_sec = arg_dict['batch_size'] / duration
			sec_per_batch = float(duration)
			format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)')
			print (format_str % (datetime.now(), cur_step, train_loss, examples_per_sec, sec_per_batch))
		if (step+1) % 100 == 0: # Tensorboard updates values every 100 steps
			summary_writer.add_summary(summary_output, cur_step)
			img_batch, label_batch = sess.run([image_valid_batch, label_valid_batch])
			summary_output = sess.run(fetches=[validation_summary], feed_dict={image_placeholder: img_batch, label_placeholder: label_batch, is_training: False})[0]
			summary_writer.add_summary(summary_output, cur_step)
		if (step+1) % 1000 == 0: # Save model every 1k steps
			checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=cur_step)

	# Save model after training is terminated...
	checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=cur_step)


def trainSegEllfitNetwork(arg_dict):
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		seg_label_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		ellfit_label_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], 6])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, ellfit_eval_batch, angle_fix_batch = arg_dict['model_construct_function'](image_placeholder, is_training)
	with tf.variable_scope('Loss'):
		print('Adding loss function...')
		loss1, errors1 = gen_loss_seg(network_eval_batch, seg_label_placeholder)
		# Alternative loss if we don't want morphological filtering
		#loss1, errors1 = gen_loss_seg_nomorph(network_eval_batch, seg_label_placeholder)
		loss2, errors2, angle_errs = gen_loss_ellreg(ellfit_eval_batch, ellfit_label_placeholder)
		loss3 = gen_loss_anglequadrant(angle_fix_batch, ellfit_label_placeholder)
		loss = tf.reduce_mean([loss1, loss2, loss3])
	##########################################
	with tf.variable_scope('Input_Decoding'):
		print('Populating input queues...')
		image_valid_batch, seg_valid_batch, label_valid_batch = get_valid_batch_segellreg(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'])
		image_train_batch, seg_train_batch, label_train_batch = get_train_batch_segellreg(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'], max_trans = arg_dict['aug_trans_max'], max_rot = arg_dict['aug_rot_max'])
		print('Starting input threads...')
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.variable_scope('Optimizer'):
		print('Initializing optimizer...')
		learn_rate, train_op = arg_dict['learn_function'](loss, arg_dict['dataset'].train_size, arg_dict['batch_size'], global_step, arg_dict['start_learn_rate'], arg_dict['epocs_per_lr_decay'], const_learn_rate=arg_dict['const_learn_rate'])
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		training_summary1, validation_summary1 = gen_summary_seg(loss1, errors1, learn_rate)
		training_summary2, validation_summary2 = gen_summary_ellreg(loss2, errors2, angle_errs, learn_rate)
		training_summary = tf.summary.merge([training_summary1, training_summary2])
		validation_summary = tf.summary.merge([validation_summary1, validation_summary2])
		summary_writer = tf.summary.FileWriter(arg_dict['log_dir'], sess.graph)
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])
	print('Beginning training...')
	for step in range(0, arg_dict['num_steps']):
		start_time = time.time()
		img_batch, seg_label_batch, ellfit_label_batch = sess.run([image_train_batch, seg_train_batch, label_train_batch])
		_, train_loss, summary_output, cur_step, ellfit_errs = sess.run(fetches=[train_op, loss, training_summary, global_step, errors2], feed_dict={image_placeholder: img_batch, seg_label_placeholder: seg_label_batch, ellfit_label_placeholder: ellfit_label_batch, is_training: True})
		duration = time.time() - start_time
		if (step+1) % 10 == 0: # CMDline updates every 10 steps
			examples_per_sec = arg_dict['batch_size'] / duration
			sec_per_batch = float(duration)
			format_str = ('%s: step %d, loss=%.4f, xerr=%.2f, yerr=%.2f (%.1f examples/sec)')
			print (format_str % (datetime.now(), cur_step, train_loss, ellfit_errs[0], ellfit_errs[1], examples_per_sec))
		if (step+1) % 100 == 0: # Tensorboard updates values every 100 steps
			summary_writer.add_summary(summary_output, cur_step)
			img_batch, seg_label_batch, ellfit_label_batch = sess.run([image_valid_batch, seg_valid_batch, label_valid_batch])
			summary_output = sess.run(fetches=[validation_summary], feed_dict={image_placeholder: img_batch, seg_label_placeholder: seg_label_batch, ellfit_label_placeholder: ellfit_label_batch, is_training: False})[0]
			summary_writer.add_summary(summary_output, cur_step)
		if (step+1) % 1000 == 0: # Save model every 1k steps
			checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=cur_step)
	# Save model after training is terminated...
	checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=cur_step)


def trainBinnedNetwork(arg_dict):
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		label_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], 6])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch, _ = arg_dict['model_construct_function'](image_placeholder, is_training, int(arg_dict['input_size']*arg_dict['bin_per_px']))
	with tf.variable_scope('Loss'):
		print('Adding loss function...')
		loss, errors = gen_loss_xyhot(network_eval_batch, label_placeholder, arg_dict['input_size'], int(arg_dict['input_size']*arg_dict['bin_per_px']))
	##########################################
	with tf.variable_scope('Input_Decoding'):
		print('Populating input queues...')
		image_valid_batch, label_valid_batch = get_eval_batch_ellreg(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'])
		image_train_batch, label_train_batch = get_train_batch_ellreg_v2(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'], max_rot = arg_dict['aug_rot_max'], max_trans = arg_dict['aug_trans_max'])
		print('Starting input threads...')
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.variable_scope('Optimizer'):
		print('Initializing optimizer...')
		learn_rate, train_op = arg_dict['learn_function'](loss, arg_dict['dataset'].train_size, arg_dict['batch_size'], global_step, arg_dict['start_learn_rate'], arg_dict['epocs_per_lr_decay'], const_learn_rate=arg_dict['const_learn_rate'])
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		training_summary, validation_summary = gen_summary_xyhot(loss, errors, learn_rate)
		summary_writer = tf.summary.FileWriter(arg_dict['log_dir'], sess.graph)
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])
	print('Beginning training...')
	for step in range(0, arg_dict['num_steps']):
		start_time = time.time()
		img_batch, label_batch = sess.run([image_train_batch, label_train_batch])
		_, train_loss, summary_output, cur_step, bin_errs = sess.run(fetches=[train_op, loss, training_summary, global_step, errors], feed_dict={image_placeholder: img_batch, label_placeholder: label_batch, is_training: True})
		duration = time.time() - start_time
		if (step+1) % 10 == 0: # CMDline updates every 10 steps
			examples_per_sec = arg_dict['batch_size'] / duration
			sec_per_batch = float(duration)
			format_str = ('%s: step %d, loss=%.4f, xerr=%.2f, yerr=%.2f (%.1f examples/sec)')
			print (format_str % (datetime.now(), cur_step, train_loss, bin_errs[0], bin_errs[1], examples_per_sec))
		if (step+1) % 100 == 0: # Tensorboard updates values every 100 steps
			summary_writer.add_summary(summary_output, cur_step)
			img_batch, label_batch = sess.run([image_valid_batch, label_valid_batch])
			summary_output = sess.run(fetches=[validation_summary], feed_dict={image_placeholder: img_batch, label_placeholder: label_batch, is_training: False})[0]
			summary_writer.add_summary(summary_output, cur_step)
		if (step+1) % 1000 == 0: # Save model every 1k steps
			checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=cur_step)
	# Save model after training is terminated...
	checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=cur_step)


def trainSegSoftNetwork(arg_dict):
	sess = tf.Session()
	##########################################
	with tf.variable_scope('Input_Variables'):
		image_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		label_placeholder = tf.placeholder(tf.float32, [arg_dict['batch_size'], arg_dict['input_size'], arg_dict['input_size'], 1])
		is_training = tf.placeholder(tf.bool, [], name='is_training')
	##########################################
	with tf.variable_scope('Network'):
		print('Constructing model...')
		network_eval_batch = arg_dict['model_construct_function'](image_placeholder, is_training)
	with tf.variable_scope('Loss'):
		print('Adding loss function...')
		loss, errors = gen_loss_seg_nomorph(network_eval_batch, label_placeholder)
	##########################################
	with tf.variable_scope('Input_Decoding'):
		print('Populating input queues...')
		image_valid_batch, label_valid_batch = get_eval_batch_seg(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'])
		image_train_batch, label_train_batch = get_train_batch_seg(arg_dict['dataset'], arg_dict['n_reader_threads'], arg_dict['batch_size'], arg_dict['input_size'], max_trans = arg_dict['aug_trans_max'], max_rot = arg_dict['aug_rot_max'])
		print('Starting input threads...')
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	##########################################
	global_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.variable_scope('Optimizer'):
		print('Initializing optimizer...')
		learn_rate, train_op = arg_dict['learn_function'](loss, arg_dict['dataset'].train_size, arg_dict['batch_size'], global_step, arg_dict['start_learn_rate'], arg_dict['epocs_per_lr_decay'], const_learn_rate=arg_dict['const_learn_rate'])
	##########################################
	with tf.variable_scope('Saver'):
		print('Generating summaries and savers...')
		training_summary, validation_summary = gen_summary_seg(loss, errors, learn_rate)
		summary_writer = tf.summary.FileWriter(arg_dict['log_dir'], sess.graph)
		saver = tf.train.Saver(slim.get_variables_to_restore(), max_to_keep=2)
	##########################################
	print('Initializing model...')
	sess.run(tf.global_variables_initializer())
	if 'network_to_restore' in arg_dict.keys() and arg_dict['network_to_restore'] is not None:
		saver.restore(sess,arg_dict['network_to_restore'])
	#
	for step in range(0, arg_dict['num_steps']):
		start_time = time.time()
		img_batch, label_batch = sess.run([image_train_batch, label_train_batch])
		_, train_loss, summary_output, cur_step = sess.run(fetches=[train_op, loss, training_summary, global_step], feed_dict={image_placeholder: img_batch, label_placeholder: label_batch, is_training: True})
		duration = time.time() - start_time
		if (step+1) % 10 == 0: # CMDline updates every 10 steps
			examples_per_sec = arg_dict['batch_size'] / duration
			sec_per_batch = float(duration)
			format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)')
			print (format_str % (datetime.now(), cur_step, train_loss, examples_per_sec, sec_per_batch))
		if (step+1) % 100 == 0: # Tensorboard updates values every 100 steps
			summary_writer.add_summary(summary_output, cur_step)
			img_batch, label_batch = sess.run([image_valid_batch, label_valid_batch])
			summary_output = sess.run(fetches=[validation_summary], feed_dict={image_placeholder: img_batch, label_placeholder: label_batch, is_training: False})[0]
			summary_writer.add_summary(summary_output, cur_step)
		if (step+1) % 1000 == 0: # Save model every 1k steps
			checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
			saver.save(sess, checkpoint_path, global_step=cur_step)

	# Save model after training is terminated...
	checkpoint_path = os.path.join(arg_dict['log_dir'], 'model.ckpt')
	saver.save(sess, checkpoint_path, global_step=cur_step)

def trainNetwork(arg_dict):
	if arg_dict['net_type'] == 'segellreg':
		trainSegEllfitNetwork(arg_dict)
	elif arg_dict['net_type'] == 'ellreg':
		trainEllregNetwork(arg_dict)
	elif arg_dict['net_type'] == 'binned':
		trainBinnedNetwork(arg_dict)
	elif arg_dict['net_type'] == 'seg':
		trainSegSoftNetwork(arg_dict)

