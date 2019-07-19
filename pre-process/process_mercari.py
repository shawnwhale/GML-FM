import os
import sys
import argparse
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config


def reset_id(input_list):
	""" Reset index a list from 0 to max_len. """
	input_set = list(set(input_list))
	output_dict = {input_set[i]:
				i for i in range(len(input_set))}
	return output_dict


def parse_ctry(ctry_path):
	""" Find the root category for each category. """
	root_ctry = {}
	with open(ctry_path, 'r') as fd:
		fd.readline() # skip first row
		line = fd.readline()
		while line:
			entries = line.strip().split(',')
			child = entries[0]
			parent = entries[6]
			root_ctry[child] = parent
			line = fd.readline()

	# find root categories
	for child in root_ctry:
		parent = root_ctry[child]
		while root_ctry.get(parent, None) in root_ctry:
			parent = root_ctry[parent]
		root_ctry[child] = parent
	return root_ctry


def parse_user(user_dir, given_ctry, root_ctry, min_int):
	""" Find users under certain category and trim. """
	users, items, times = [], [], []
	for path in os.listdir(user_dir):
		user_int_f = os.path.join(user_dir, path)
		with open(user_int_f, 'r') as fd:
			fd.readline() # skip first row
			line = fd.readline()
			while line:
				entries = line.strip().split(',')
				if given_ctry == root_ctry[entries[4]]:
					times.append(entries[0])
					users.append(entries[1])
					items.append(entries[2])
				line = fd.readline()

	user_int = pd.DataFrame(
		list(zip(users, items, times)),
		columns=['user', 'item', 'time'])
	user_lens = user_int.groupby('user').size()
	user_int = user_int[np.in1d(
		user_int.user, user_lens[user_lens>=min_int].index)]
	return user_int, user_int.item.unique().tolist()


def parse_item(item_dir, item_sold):
	""" Find items and other info purchased in user_int. """
	item_info = {}
	for path in os.listdir(item_dir):
		item_f = os.path.join(item_dir, path)
		items = pd.read_csv(item_f,
			usecols = [0, 5, 7, 8, 12, 13, 14],
			dtype={0: str, 5: str, 7: str,
			8: str, 12: str, 13: str, 14: str})
		for row in items.itertuples():
			item = getattr(row, 'item_id_hash')
			if item in item_sold:
				ctry = getattr(row, 'category_id_hash')
				item_cdn = getattr(row, 'item_condition')
				ship_mtd = getattr(row, 'shipping_method')
				ship_dtn = getattr(row, 'shipping_duration')
				ship_area = getattr(row, 'shipping_from_area')
				item_info[item] = [
					ctry, item_cdn, ship_mtd, ship_area, ship_dtn]

	assert len(item_info) == len(item_sold), \
					'some items cannot be found'
	assert all(len(entry) > 0 for value in
		item_info.values() for entry in value), 'missing entries'
	return item_info


def norm_entry(user_int, item_info):
	""" Normalize users and items info. """
	users, items, item_info_norm = [], [], {}
	user_dict = reset_id(user_int.user.unique())
	item_dict = reset_id(user_int.item.unique())

	ctry_dict = reset_id(v[0] for v in item_info.values())
	item_cdn_dict = reset_id(v[1] for v in item_info.values())
	ship_mtd_dict = reset_id(v[2] for v in item_info.values())
	ship_area_dict = reset_id(v[3] for v in item_info.values())
	ship_dtn_dict = reset_id(v[4] for v in item_info.values())

	user_num = len(user_dict)
	item_num = len(item_dict)
	ctry_num = len(ctry_dict)
	item_cdn_num = len(item_cdn_dict)
	ship_mtd_num = len(ship_mtd_dict)
	ship_area_num = len(ship_area_dict)

	for r in user_int.itertuples():
		users.append(user_dict[getattr(r, 'user')])
		items.append(item_dict[getattr(r, 'item')])
	times = user_int['time'].tolist()
	user_int = pd.DataFrame(
		list(zip(users, items, times)),
		columns=['user', 'item', 'time'])

	# leave item norm in ng_sample
	for item in item_info:
		point_len = user_num + item_num
		info = item_info[item]
		ctry = ctry_dict[info[0]] + point_len
		point_len += ctry_num
		item_cdn = item_cdn_dict[info[1]] + point_len
		point_len += item_cdn_num
		ship_mtd = ship_mtd_dict[info[2]] + point_len
		point_len += ship_mtd_num
		ship_area = ship_area_dict[info[3]] + point_len
		point_len += ship_area_num
		ship_dtn = ship_dtn_dict[info[4]] + point_len
		item_info_norm[item_dict[item]] = [ctry, item_cdn,
							ship_mtd, ship_area, ship_dtn]

	return user_int, item_info_norm


def split_data(rating):
	""" Split data with leave-one-out trick. """
	splits = []
	per_user_len = rating.groupby('user').size().tolist()

	for user in range(len(per_user_len)):
		for _ in range(per_user_len[user] - 1):
			splits.append('train')
		splits.append('test')
	rating['split'] = splits

	train_int = rating[rating['split'] == 'train']
	test_int = rating[rating['split'] == 'test']

	return train_int.reset_index(drop=True), \
			test_int.reset_index(drop=True)


def ng_sample(rating, pst_mat,
	user_len, item_len, item_info, ng_num):
	""" Negative sampling two instances for each one. """
	rates, users, items, item_infos = [], [], [], []

	for r in rating.itertuples():
		user = getattr(r, 'user')
		item = getattr(r, 'item')
		pst_mat[user, item] = 1

	for r in rating.itertuples():
		u = getattr(r, 'user')
		i = getattr(r, 'item')
		rates.append(1.0)
		users.append(u)
		items.append(i)
		item_infos.append(item_info[i])

		for t in range(ng_num):
			j = np.random.randint(item_len)
			while (u, j) in pst_mat:
				j = np.random.randint(item_len)
				rates.append(-1.0)
				users.append(u)
				items.append(j)
				item_infos.append(item_info[j])
	rating = pd.DataFrame(
		list(zip(rates, users, items, item_infos)),
		columns=['rating', 'user', 'item', 'item_info'])
	rating['item'] = rating['item'].apply(
								lambda x: x + user_len)
	return rating, pst_mat


def to_disk(rating, path):
	""" Write to disk following the given format. """
	with open(path, 'w') as fd:
		for r in rating.itertuples():
			rate = str(getattr(r, 'rating'))
			user = str(getattr(r, 'user'))
			item = str(getattr(r, 'item'))
			item_info = getattr(r, 'item_info')

			line = rate + ' ' + user + ':1 ' + \
				item + ':1 ' + str(item_info[0]) + ':1 ' + \
				str(item_info[1]) + ':1 ' + str(item_info[2]) + ':1 ' + \
				str(item_info[3]) + ':1 ' + str(item_info[4]) + ':1'
			fd.write(line + '\n')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--category_hash',
		type=str,
		required=True,
		help='processed dataset hash')
	parser.add_argument('--category_name',
		type=str,
		required=True,
		help='processed dataset name')
	args = parser.parse_args()


	np.random.seed(2019)

	ctry_path = os.path.join(config.mercari_dir, 'categories_final.csv')
	item_dir = os.path.join(config.mercari_dir, 'items_small_sample')
	user_dir = os.path.join(config.mercari_dir, 'user_purchase_small_sample')

	root_ctry = parse_ctry(ctry_path)
	print("found root category for {} entries".format(len(root_ctry)))

	user_int, item_sold = parse_user(user_dir,
			args.category_hash, root_ctry, min_int=5)
	item_sold = {i: 'true' for i in item_sold} # find keys in dict is faster
	item_info = parse_item(item_dir, item_sold)

	user_int, item_info = norm_entry(user_int, item_info)
	user_len = len(user_int.user.unique())
	item_len = len(user_int.item.unique())
	print("number of users: {}, number of items: {}".format(
											user_len, item_len))
	print("number of instances: {}".format(len(user_int)))

	user_int = user_int.sort_values(by=['user', 'time'])
	train_int, test_int = split_data(user_int)

	pst_mat = sp.dok_matrix((user_len, item_len), dtype=np.float32)
	train_int, pst_mat = ng_sample(
		train_int, pst_mat,	user_len,
		item_len,  item_info, ng_num=2)
	test_int, _ = ng_sample(
		test_int, pst_mat, user_len,
		item_len, item_info, ng_num=99)

	if not os.path.exists(args.category_name):
		os.mkdir(args.category_name)
	to_disk(train_int, os.path.join(
		args.category_name, args.category_name + '.train.libfm'))
	to_disk(test_int, os.path.join(
		args.category_name, args.category_name + '.test.libfm'))

	print("All done.")


if __name__ == '__main__':
	main()
