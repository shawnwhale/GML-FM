import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

import model
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr",
	type=float,
	default=0.001,
	help="learning rate")
parser.add_argument("--dropout",
	type=float,
	default=0.0,
	help="dropout rate for FM and MLP")
parser.add_argument("--batch_size",
	type=int,
	# default=256,
	default=1024,
	help="batch size for training")
parser.add_argument("--epochs",
	type=int,
	default=60,
	help="training epochs")
parser.add_argument("--hidden_factor",
	type=int,
	# default=256,
	default=32,
	help="predictive factors numbers in the model")
parser.add_argument("--layers",
	type=int,
	# default=1,
	default=4,
	help="number of layers in MLP model")
parser.add_argument("--lamda",
	type=float,
	default=0.0,
	help="regularizer for bilinear layers")
parser.add_argument("--out",
	default=True,
	help="save model or not")
parser.add_argument("--gpu",
	type=str,
	default="0",
	help="gpu card ID")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#############################  PREPARE DATASET #########################
features_map, num_features = data_utils.map_features()
np.save('./features_map', features_map)
refeatures_map = {}
for key in features_map.keys():
	value = features_map[key]
	refeatures_map[value] = key

# features_map 是 ~[特征] = 0到N

train_dataset = data_utils.FMData(config.train_libfm, features_map)
test_dataset = data_utils.FMData(config.test_libfm, features_map)

train_loader = data.DataLoader(train_dataset, pin_memory=True,
			batch_size=args.batch_size, shuffle=True, num_workers=0)
test_loader = data.DataLoader(test_dataset, pin_memory=True,
			batch_size=100, shuffle=False, num_workers=0)

##############################  CREATE MODEL ###########################
model = model.GML_FM(num_features,
	args.hidden_factor, config.activation_function,
	config.use_product, args.layers, args.dropout)
model = model.cuda()
if config.optimizer == 'Adagrad':
	optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
elif config.optimizer == 'Adam':
	optimizer = optim.Adam(
		model.parameters(), lr=args.lr)
elif config.optimizer == 'SGD':
	optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif config.optimizer == 'Momentum':
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

criterion = nn.MSELoss()
# writer = SummaryWriter() # for visualization

###############################  TRAINING ############################
count, best_hr = 0, 0.0
for epoch in range(args.epochs):
	model.train() # Enable dropout and batch_norm
	start_time = time.time()

	for features, feature_values, label in train_loader:
		features = features.cuda()
		feature_values = feature_values.cuda()
		label = label.cuda()

		model.zero_grad()
		prediction = model(features, feature_values)
		loss = criterion(prediction, label)
		loss.backward()
		optimizer.step()
		# writer.add_scalar('data/loss', loss.item(), count)
		count += 1

	model.eval()
	train_result = evaluate.metric_rmse(model, train_loader)
	hr, ndcg, HR, NDCG , ils, ILS, kendall, KENDALL, ENTROPY= evaluate.metrics(model, test_loader ,top_k=20, refeatures_map = refeatures_map )

	print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
						"%H: %M: %S", time.gmtime(time.time()-start_time)))
	print("Train_RMSE: {:.4f}, Test_hr: {:.4f}, Test_ndcg: {:.4f}, Test_ils: {:.4f}, Test_kendall: {:.4f} ENTROPY: {:.4f} ".format(
													train_result, hr, ndcg, ils, kendall,ENTROPY))

	if hr > best_hr:
		best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
		best_entr = ENTROPY
		best_kendall = kendall
		if args.out:
			if not os.path.exists(config.model_path):
				os.mkdir(config.model_path)
			torch.save(model,
				'{}{}.pth'.format(config.model_path, config.model))
			np.save('./hr_ttest.npy', np.array(HR))
			np.save('./ndcg_ttest.npy', np.array(NDCG))
			np.save('./ils_ttest.npy', np.array(ILS))
print("End. Best epoch {:03d}: Test_hr is {:.4f}, Test_ndcg is {:.4f}, Test_kendall is {:.4f}, Test_entr is {:.4f}".format(
													best_epoch, best_hr, best_ndcg,best_kendall, best_entr))
