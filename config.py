# dataset name
dataset = 'Office'
# assert dataset in ['ml-tag', 'frappe']

# model name
model = 'GML-FM'

# important settings (normally default is the paper choice)
optimizer = 'Adam'
activation_function = 'tanh'
assert optimizer in ['Adagrad', 'Adam', 'SGD', 'Momentum']
assert activation_function in ['relu', 'sigmoid', 'tanh', 'identity']

# use product to map value into real space or not
use_product = True

# paths
main_path = '/raid/guoyangyang/recommendation/FM-Data/{}/'.format(dataset)

train_libfm = main_path + '{}.train.libfm'.format(dataset)
test_libfm = main_path + '{}.test.libfm'.format(dataset)

model_path = './models/'
