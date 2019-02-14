# Defines class items handling a dataset.
class TrackingDataset:
	def __init__(self, train_file, valid_file, folder_prefix):
		self.train_list = open(train_file, 'r').read().splitlines()
		self.valid_list = open(valid_file, 'r').read().splitlines()
		self.train_images = [folder_prefix + '/Ref/' + train_item + '.png' for train_item in self.train_list]
		self.train_labels = [folder_prefix + '/Ell/' + train_item + '.txt' for train_item in self.train_list]
		self.train_seg = [folder_prefix + '/Seg/' + train_item + '.png' for train_item in self.train_list]
		self.train_size = len(self.train_list)
		self.valid_images = [folder_prefix + '/Ref/' + valid_item + '.png' for valid_item in self.valid_list]
		self.valid_labels = [folder_prefix + '/Ell/' + valid_item + '.txt' for valid_item in self.valid_list]
		self.valid_seg = [folder_prefix + '/Seg/' + valid_item + '.png' for valid_item in self.valid_list]
		self.valid_size = len(self.valid_list)

