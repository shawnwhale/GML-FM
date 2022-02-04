import numpy as np
import torch.utils.data as data

import config


def read_features(file, features,features_tran):
	""" Read features from the given file. """
	"""增加features_tran用以储存 key -》old """
	i = len(features)
	with open(file, 'r') as fd:
		line = fd.readline()
		while line:
			items = line.strip().split()
			for j in range(1,len(items)):
			# for item in items[1:]:
				item = items[j]
				item = item.split(':')[0]
				if item not in features:
					if j == 2:
						features_tran[i] = item
					features[item] = i
					i += 1
			line = fd.readline()
	return features


def map_features():
	""" Get the number of existing features in all the three files. """
	features = {}
	features_tran = {}
	features = read_features(config.train_libfm, features,features_tran)
	features = read_features(config.test_libfm, features,features_tran)
	print("number of features: {}".format(len(features)))

	np.save("./features_tran.npy",features_tran)
	return features, len(features)


class FMData(data.Dataset):
	""" Construct the FM pytorch dataset. """
	def __init__(self, file, feature_map):
		super(FMData, self).__init__()
		self.label = []
		self.features = []
		self.feature_values = []

		with open(file, 'r') as fd:
			line = fd.readline()

			while line:
				items = line.strip().split()

				# convert features
				raw = [item.split(':')[0] for item in items[1:]]
				self.features.append(
					np.array([feature_map[item] for item in raw]))
				self.feature_values.append(np.array(
					[item.split(':')[1] for item in items[1:]], dtype=np.float32))
				#raw是原始特征list，features是转换后的0-N,feature_values默认为1
				# convert labels
				self.label.append(np.float32(items[0]))

				line = fd.readline()

		assert all(len(item) == len(self.features[0]
			) for item in self.features), 'features are of different length'

	def __len__(self):
		return len(self.label)

	def __getitem__(self, idx):
		label = self.label[idx]
		features = self.features[idx]
		feature_values = self.feature_values[idx]
		return features, feature_values, label
