import sys, getopt, os, re, argparse
import utils.TrainNetwork as trainNet
import utils.InferMovie as inferNet
from utils import datasets
from utils import models
from utils import training
import inspect
import resource

def main(argv):
	# Parse some selections
	possible_models = [x for x in inspect.getmembers(models)]
	possible_models = {x[0]:x[1] for x in possible_models if inspect.isfunction(getattr(models, x[0]))}
	possible_learn = [x for x in inspect.getmembers(training)]
	possible_learn = {x[0]:x[1] for x in possible_learn if inspect.isfunction(getattr(training, x[0]))}
	# Filter the learning functions for the ones that actually contain "train" in their name (omits summary builders, other helper functions)
	learn_functs = [re.search('.*train.*',x).group() for x in possible_learn.keys() if re.search('.*train.*',x)]
	possible_learn = { key:value for key,value in possible_learn.items() if key in learn_functs }

	# Start up the definitions of argument parsers
	parser = argparse.ArgumentParser(description='Run Tensorflow Graphs')
	# Separate out specific training/eval/inference parameters
	subparsers = parser.add_subparsers(title='mode', description='Type of processing to do with the network', help='Additional Help', dest='mode')
	# Add general arguments
	parser.add_argument('--net_type', help='Type of network model (default ellipse_regression)', choices=['ellreg','segellreg','binned','seg'], default='segellreg')
	parser.add_argument('--batch_size', help='Batch size of the network (default 5)', type=int, default=5)
	parser.add_argument('--input_size', help='Frame input size of the network (default 480)', type=int, default=480)

	# Training parameters
	parser_train = subparsers.add_parser('Train', help='Training Parameters')
	parser_train.add_argument('--model', help='Network model to use', choices=possible_models.keys(), required=True)
	parser_train.add_argument('--network_to_restore', help='Network checkpoint to restore')
	parser_train.add_argument('--n_reader_threads', help='Number of CPU threads for fetching data (default 3)', default=3, type=int)
	parser_train.add_argument('--log_dir', help='Log folder', default='.')
	parser_train.add_argument('--train_list', help='File containing list of identifiers for training', required=True)
	parser_train.add_argument('--valid_list', help='File containing list of identifiers for validating', required=True)
	parser_train.add_argument('--dataset_folder', help='Root folder of the training set')
	parser_train.add_argument('--num_steps', help='Steps to take during training (default 500k)', type=int, default=500000)
	parser_train.add_argument('--start_learn_rate', help='Initial learning rate for training (default 5e-7)', type=float, default=5e-7)
	parser_train.add_argument('--epocs_per_lr_decay', help='Epocs per learn rate decay (default 5)', type=int, default=5)
	parser_train.add_argument('--decay_learn_rate', help='Decay learn rate (default constant)', dest='const_learn_rate', action='store_false', default=True)
	parser_train.add_argument('--learn_function', help='Learn function', choices=possible_learn.keys(), required=True)
	parser_train.add_argument('--aug_rot_max', help='Max small rotation augmentation (degrees, train set)', type=float, default=2.5)
	parser_train.add_argument('--aug_trans_max', help='Max small translation augmentation (px, train set)', type=float, default=5.0)
	parser_train.add_argument('--bin_per_px', help='Multiplier for number of bins per pixel (default 10)', type=int, default=10)

	# Inference parameters
	parser_infer = subparsers.add_parser('Infer', help='Inference Parameters')
	parser_infer.add_argument('--model', help='Network model to use', choices=possible_models.keys(), required=True)
	parser_infer.add_argument('--bin_per_px', help='Multiplier for number of bins per pixel (Binned Network ONLY default 10)', type=int, default=10)
	parser_infer.add_argument('--network_to_restore', help='Network checkpoint to restore', required=True)
	parser_infer.add_argument('--input_movie', help='Input movie to evaluate')
	parser_infer.add_argument('--ellfit_movie_output', help='Output a movie with the plotted ellipse-prediction', action='store_true', default=False)
	parser_infer.add_argument('--affine_movie_output', help='Output cropped + centered + rotated movie', action='store_true', default=False)
	parser_infer.add_argument('--crop_movie_output', help='Output center-cropped movie (uses same affine_crop_dim)', action='store_true', default=False)
	parser_infer.add_argument('--affine_crop_dim', help='Cropped dimension for affine-transformed movie (default 112)', type=int, default=112)
	parser_infer.add_argument('--ellfit_output', help='Output ellipse-fit data file (npy)', action='store_true', default=False)
	parser_infer.add_argument('--ellfit_features_output', help='Output ellipse-fit feature data file (npy)', action='store_true', default=False)
	parser_infer.add_argument('--seg_movie_output', help='Output the segmentation mask as a movie', action='store_true', default=False)

	# Multiple Inference parameters
	parser_infermany = subparsers.add_parser('InferMany', help='Inference Parameters')
	parser_infermany.add_argument('--model', help='Network model to use', choices=possible_models.keys(), required=True)
	parser_infermany.add_argument('--bin_per_px', help='Multiplier for number of bins per pixel (Binned Network ONLY default 10)', type=int, default=10)
	parser_infermany.add_argument('--network_to_restore', help='Network checkpoint to restore', required=True)
	parser_infermany.add_argument('--input_movie_list', help='Text file containing line-by-line list of movies to process')
	parser_infermany.add_argument('--ellfit_movie_output', help='Output a movie with the plotted ellipse-prediction', action='store_true', default=False)
	parser_infermany.add_argument('--affine_movie_output', help='Output cropped + centered + rotated movie', action='store_true', default=False)
	parser_infermany.add_argument('--crop_movie_output', help='Output center-cropped movie (uses same affine_crop_dim)', action='store_true', default=False)
	parser_infermany.add_argument('--affine_crop_dim', help='Cropped dimension for affine-transformed movie (default 112)', type=int, default=112)
	parser_infermany.add_argument('--ellfit_output', help='Output ellipse-fit data file (npy)', action='store_true', default=False)
	parser_infermany.add_argument('--ellfit_features_output', help='Output ellipse-fit feature data file (npy)', action='store_true', default=False)
	parser_infermany.add_argument('--seg_movie_output', help='Output the segmentation mask as a movie', action='store_true', default=False)

	# Grab all the parsed arguments
	args = parser.parse_args()
	arg_dict = args.__dict__
	arg_dict['model_construct_function'] = possible_models[args.model]
	# Other keyed selections...
	if 'learn_function' in arg_dict.keys() and arg_dict['learn_function'] is not None:
		arg_dict['learn_function'] = possible_learn[args.learn_function]

	# Prep the dataset
	if 'dataset_folder' in arg_dict.keys() and arg_dict['dataset_folder'] is not None:
		arg_dict['dataset'] = datasets.TrackingDataset(arg_dict['train_list'], arg_dict['valid_list'], arg_dict['dataset_folder'])
	elif 'train_list' in arg_dict.keys():
		arg_dict['dataset'] = datasets.TrackingDataset(arg_dict['train_list'], arg_dict['valid_list'], '.')

	# Call the correct sub-parser and send the argument dictionary for futher separation
	# Note that the keys are heavily dependent upon naming conventions...
	if args.mode == 'Train':
		trainNet.trainNetwork(arg_dict)
	elif args.mode == 'Infer' or args.mode == 'InferMany':
		inferNet.inferMovie(arg_dict)
	else:
		print('Could not understand commands:')
		print(args.__dict__)
	

if __name__ == '__main__':
	main(sys.argv[1:])

