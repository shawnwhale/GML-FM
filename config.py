# dataset name
dataset = 'Books'

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
meta_dir = '/raid/guoyangyang/recommendation/'
amazon_dir = meta_dir + 'amazon/'
ml_dir = meta_dir + 'ml-1m/'
mercari_dir = meta_dir + 'mercari/'

# paths
main_path = meta_dir + '/FM-Data/{}/'.format(dataset)
train_libfm = main_path + '{}.train.libfm'.format(dataset)
test_libfm = main_path + '{}.test.libfm'.format(dataset)

model_path = './models/'
