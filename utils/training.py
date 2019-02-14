from .readers import means, scale
import tensorflow as tf
import tensorflow.contrib.slim as slim
from .readers import ellreg_to_xyhot
from .readers import atan2
import scipy.ndimage.morphology as morph
import numpy as np

def gen_loss_ellreg(network_eval_batch, label_placeholder):
	loss = slim.losses.mean_squared_error(network_eval_batch, label_placeholder)
	# If angle should be ignored...
	#loss = slim.losses.mean_squared_error(tf.slice(network_eval_batch,[0,0],[-1,4]), tf.slice(label_placeholder,[0,0],[-1,4]))
	errors = tf.multiply(tf.reduce_mean(tf.abs(tf.subtract(network_eval_batch, label_placeholder)), reduction_indices=0), scale)
	gt_angle = 0.5*atan2(tf.add(tf.multiply(tf.slice(label_placeholder,[0,4],[-1,1]), scale[4]), means[4]), tf.add(tf.multiply(tf.slice(label_placeholder,[0,5],[-1,1]), scale[5]), means[5]))
	test_angle = 0.5*atan2(tf.add(tf.multiply(tf.slice(network_eval_batch,[0,4],[-1,1]), scale[4]), means[4]), tf.add(tf.multiply(tf.slice(network_eval_batch,[0,5],[-1,1]), scale[5]), means[5]))
	angles = tf.reduce_mean(tf.abs(tf.subtract(test_angle,gt_angle)))*180/np.pi
	return loss, errors, angles

def gen_loss_seg(network_eval_batch, label_placeholder):
	# Apply morphological filtering to the label
	filter1 = tf.expand_dims(tf.constant(morph.iterate_structure(morph.generate_binary_structure(2,1),5),dtype=tf.float32),-1)
	seg_morph = tf.nn.dilation2d(tf.nn.erosion2d(label_placeholder,filter1,[1,1,1,1],[1,1,1,1],"SAME"),filter1,[1,1,1,1],[1,1,1,1],"SAME")
	filter2 = tf.expand_dims(tf.constant(morph.iterate_structure(morph.generate_binary_structure(2,1),4),dtype=tf.float32),-1)
	seg_morph = tf.nn.erosion2d(tf.nn.dilation2d(seg_morph,filter2,[1,1,1,1],[1,1,1,1],"SAME"),filter2,[1,1,1,1],[1,1,1,1],"SAME")
	#seg_morph = label_placeholder

	# Create the 2 bins
	mouse_label = tf.to_float(tf.greater(seg_morph, 0.0))
	background_label = tf.to_float(tf.equal(seg_morph, 0.0))
	combined_label = tf.concat([mouse_label, background_label] ,axis=3)
	flat_combined_label = tf.reshape(combined_label, [-1, 2])
	flat_network_eval = tf.reshape(network_eval_batch, [-1, 2])
	loss = tf.losses.softmax_cross_entropy(flat_combined_label, flat_network_eval)
	# Could do something fancy with counting TP/FP/TN/FN based on a softmax/argmax between the 2
	errors = None
	return loss, errors

def gen_loss_seg_nomorph(network_eval_batch, label_placeholder):
	# Create the 2 bins
	mouse_label = tf.to_float(tf.greater(label_placeholder, 0.0))
	background_label = tf.to_float(tf.equal(label_placeholder, 0.0))
	combined_label = tf.concat([mouse_label, background_label] ,axis=3)
	flat_combined_label = tf.reshape(combined_label, [-1, 2])
	flat_network_eval = tf.reshape(network_eval_batch, [-1, 2])
	loss = tf.losses.softmax_cross_entropy(flat_combined_label, flat_network_eval)
	# Could do something fancy with counting TP/FP/TN/FN based on a softmax/argmax between the 2
	errors = None
	return loss, errors


def gen_loss_xyhot(network_eval_batch, label_placeholder, input_size, nbins):
	xhot_est, yhot_est = tf.unstack(network_eval_batch)
	xhot, yhot = ellreg_to_xyhot(label_placeholder, nbins, nbins/input_size)
	loss1 = tf.reduce_mean(-tf.reduce_sum(xhot * tf.log(xhot_est), reduction_indices=[1]))
	loss2 = tf.reduce_mean(-tf.reduce_sum(yhot * tf.log(yhot_est), reduction_indices=[1]))
	loss = tf.reduce_mean(loss1 + loss2)
	xerr = tf.reduce_mean(tf.abs(tf.subtract(tf.cast(tf.argmax(xhot_est, 1), tf.float32),tf.cast(tf.argmax(xhot,1),tf.float32))))
	yerr = tf.reduce_mean(tf.abs(tf.subtract(tf.cast(tf.argmax(yhot_est, 1), tf.float32),tf.cast(tf.argmax(yhot,1),tf.float32))))
	errors = tf.stack([xerr/nbins*input_size, yerr/nbins*input_size])
	return loss, errors

def gen_loss_rotate(rotations, label_placeholder):
	label = tf.slice(label_placeholder,[0,4],[-1,2])
	loss = slim.losses.mean_squared_error(rotations, label)
	errors = tf.multiply(tf.reduce_mean(tf.abs(tf.subtract(rotations, label)), reduction_indices=0), tf.slice(scale,[4],[2]))
	return loss, errors

# angle_probs is of size [batch, 4]
# label_placeholder is of size [batch, 6] where [batch, 4] is sin(angle) and [batch,5] is cos(angle)
def gen_loss_anglequadrant(angle_probs, label_placeholder):
	label_sin = tf.add(tf.multiply(tf.slice(label_placeholder,[0,4],[-1,1]), scale[4]), means[4])
	label_cos = tf.add(tf.multiply(tf.slice(label_placeholder,[0,5],[-1,1]), scale[5]), means[5])
	label_probs = tf.zeros_like(angle_probs)+[0.,0.,1.,0.] # Default to choice 3: Everything incorrect
	label_probs = tf.where(tf.squeeze(tf.greater(label_cos,np.sin(np.pi/4.))), tf.zeros_like(angle_probs)+[1.,0.,0.,0.], label_probs) # Choice 1: Everything is correct
	label_probs = tf.where(tf.squeeze(tf.greater(label_sin,np.sin(np.pi/4.))), tf.zeros_like(angle_probs)+[0.,1.,0.,0.], label_probs) # Choice 2: Fix when sin prediction < 0.707
	label_probs = tf.where(tf.squeeze(tf.less(label_sin,-np.sin(np.pi/4.))), tf.zeros_like(angle_probs)+[0.,0.,0.,1.], label_probs) # Choice 4: Fix when sin prediction > -0.707

	loss = tf.losses.softmax_cross_entropy(label_probs, angle_probs)
	return loss
	

def gen_summary_ellreg(loss, errors, angle_errs, learn_rate):
	learn_rate_summary = tf.summary.scalar('training/learn_rate', learn_rate)
	valid_loss_summary = tf.summary.scalar('validation/losses/loss_ellfit', loss)
	valid_xerr_summary = tf.summary.scalar('validation/xy_error/xErr', errors[0])
	valid_yerr_summary = tf.summary.scalar('validation/xy_error/yErr', errors[1])
	valid_minerr_summary = tf.summary.scalar('validation/axis_error/minErr', errors[2])
	valid_majerr_summary = tf.summary.scalar('validation/axis_error/majErr', errors[3])
	valid_sinerr_summary = tf.summary.scalar('validation/dir_error/sinAngErr', errors[4])
	valid_coserr_summary = tf.summary.scalar('validation/dir_error/cosAngErr', errors[5])
	valid_angle_summary = tf.summary.scalar('validation/dir_error/degAngErr', angle_errs)
	validation_summary = tf.summary.merge([valid_loss_summary, valid_xerr_summary, valid_yerr_summary, valid_minerr_summary, valid_majerr_summary, valid_sinerr_summary, valid_coserr_summary, valid_angle_summary])
	train_loss_summary = tf.summary.scalar('training/losses/loss_ellfit', loss)
	train_xerr_summary = tf.summary.scalar('training/xy_error/xErr', errors[0])
	train_yerr_summary = tf.summary.scalar('training/xy_error/yErr', errors[1])
	train_minerr_summary = tf.summary.scalar('training/axis_error/minErr', errors[2])
	train_majerr_summary = tf.summary.scalar('training/axis_error/majErr', errors[3])
	train_sinerr_summary = tf.summary.scalar('training/dir_error/sinAngErr', errors[4])
	train_coserr_summary = tf.summary.scalar('training/dir_error/cosAngErr', errors[5])
	train_angle_summary = tf.summary.scalar('training/dir_error/degAngErr', angle_errs)
	training_summary = tf.summary.merge([train_loss_summary, train_xerr_summary, train_yerr_summary, train_minerr_summary, train_majerr_summary, train_sinerr_summary, train_coserr_summary, learn_rate_summary, train_angle_summary])
	return training_summary, validation_summary

def gen_summary_seg(loss, errors, learn_rate):
	learn_rate_summary = tf.summary.scalar('training/learn_rate', learn_rate)
	valid_loss_summary = tf.summary.scalar('validation/losses/loss_seg', loss)
	validation_summary = tf.summary.merge([valid_loss_summary])
	train_loss_summary = tf.summary.scalar('training/losses/loss', loss)
	training_summary = tf.summary.merge([train_loss_summary, learn_rate_summary])
	return training_summary, validation_summary

def gen_summary_xyhot(loss, errors, learn_rate):
	learn_rate_summary = tf.summary.scalar('training/learn_rate', learn_rate)
	valid_loss_summary = tf.summary.scalar('validation/losses/loss_xyhot', loss)
	valid_xerr_summary = tf.summary.scalar('validation/xy_error/xErr', errors[0])
	valid_yerr_summary = tf.summary.scalar('validation/xy_error/yErr', errors[1])
	validation_summary = tf.summary.merge([valid_loss_summary, valid_xerr_summary, valid_yerr_summary])
	train_loss_summary = tf.summary.scalar('training/losses/loss', loss)
	train_xerr_summary = tf.summary.scalar('training/xy_error/xErr', errors[0])
	train_yerr_summary = tf.summary.scalar('training/xy_error/yErr', errors[1])
	training_summary = tf.summary.merge([train_loss_summary, train_xerr_summary, train_yerr_summary, learn_rate_summary])
	return training_summary, validation_summary

def gen_train_op_adam(loss, train_size, batch_size, global_step, init_learn_rate = 1e-3, num_epochs_per_decay = 50, const_learn_rate = False):
	num_batches_per_epoch = ((train_size) / batch_size)
	decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
	learning_rate_decay_factor = 0.15
	if const_learn_rate:
		learn_rate = init_learn_rate
	else:
		learn_rate = tf.train.exponential_decay(init_learn_rate, global_step, decay_steps, learning_rate_decay_factor, staircase=False)
	optimizer = tf.train.AdamOptimizer(learn_rate)
	#
	train_op = slim.learning.create_train_op(loss, optimizer)
	#train_op = optimizer.minimize(loss, global_step=global_step, colocate_gradients_with_ops=True)
	return learn_rate, train_op

