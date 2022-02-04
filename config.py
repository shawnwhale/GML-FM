import  numpy as np

# dataset name
dataset = 'ml-1m'
# dataset = 'goodbooks'
# model name
model = 'GML-FM'

# important settings (normally default is the paper choice)
optimizer = 'Adam'
activation_function = 'tanh'
assert optimizer in ['Adagrad', 'Adam', 'SGD', 'Momentum']
assert activation_function in ['relu', 'sigmoid', 'tanh', 'identity']

# use product to map value into real space or not
use_product = True

# raw data paths
meta_dir = 'D:/PyCharm Community Edition 2019.1.3/GML-FM/'
amazon_dir = meta_dir + 'amazon/'
ml_dir = meta_dir + 'ml-1m/'
# ml_dir = meta_dir + 'books/'
mercari_dir = meta_dir + 'mercari/'

# paths
main_path = meta_dir + 'FM-Data/{}/'.format(dataset)
train_libfm = main_path + '{}.train.libfm'.format(dataset)
test_libfm = main_path + '{}.test.libfm'.format(dataset)

model_path = './models/'


#item编号增加量
item_add = 6040
# item_add = 5432

#
# item_dim = np.load('./item_dim.npy')
# features_tran = np.load("./features_tran.npy",allow_pickle=True).tolist()
# item_dict = np.load('./item_dict.npy',allow_pickle=True).tolist()