import os
import sys
import gzip
import random
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config


def get_df(path):
	""" Apply raw data to pandas DataFrame. """
	idx = 0
	df = {}
	g = gzip.open(path, 'rb')
	for line in g:
		df[idx] = eval(line)
		idx += 1
	return pd.DataFrame.from_dict(df, orient='index')


def match_genre(meta_path, item_set):
	""" Match categories with item id. """
	genre_set = set()
	item_to_genre = {}
	with gzip.open(meta_path, 'rb') as g:
		for line in g:
			line = eval(line)
			item = line['asin']
			genre = line['categories']

			if item not in item_set:
				continue
			if item not in item_to_genre:
				item_to_genre[item] = []
			for g in genre:
				genre_set.add(str(g))
				item_to_genre[item].append(str(g))
	return list(genre_set), item_to_genre


def ng_sample(rating, pst_mat,
	user_len, item_len, item_to_genre, ng_num):
	""" Negative sampling two instances for each one. """
	users, items, genres, rates = [], [], [], []

	for r in rating.itertuples():
		user = getattr(r, 'user')
		item = getattr(r, 'item')
		pst_mat[user, item] = 1

	for r in rating.itertuples():
		u = getattr(r, 'user')

		users.append(u)
		items.append(getattr(r, 'item'))
		genres.append(getattr(r, 'genre'))
		rates.append(1.0)

		for t in range(ng_num):
			j = np.random.randint(item_len)
			while (u, j) in pst_mat:
				j = np.random.randint(item_len)
			g = random.choice(item_to_genre[j])

			users.append(u)
			items.append(j)
			genres.append(g)
			rates.append(-1.0)

	rating = pd.DataFrame(
		list(zip(users, items, genres, rates)),
		columns=['user', 'item', 'genre', 'rating'])

	rating['item'] = rating['item'].apply(
						lambda x: x + user_len)
	rating['genre'] = rating['genre'].apply(
						lambda x: x + user_len + item_len)
	return rating, pst_mat


def norm_data(rating, user_dict,
	item_dict, genre_dict, item_to_genre):
	""" Normalize user id, item id and genre id. """
	users, items, genres, rates = [], [], [], []
	item_to_genre_norm = {}

	for r in rating.itertuples():
		user = getattr(r, 'user')
		item = getattr(r, 'item')

		genre = random.choice(
				item_to_genre[item]) # random choose a genre

		u = user_dict[user]
		i = item_dict[item]
		g = genre_dict[genre]
		if i not in item_to_genre_norm:
			item_to_genre_norm[i] = set()
		item_to_genre_norm[i].add(g)

		users.append(u)
		items.append(i)
		genres.append(g)
		rates.append(1.0)
	rating['user'] = users
	rating['item'] = items
	rating['genre'] = genres
	rating['rating'] = rates
	item_to_genre_norm = {i: list(
		item_to_genre_norm[i]) for i in item_to_genre_norm}
	return rating, item_to_genre_norm


def split_data(rating):
	""" Split data with leave-one-out trick. """
	splits = []
	per_user_len = rating.groupby('user').size().tolist()

	for user in range(len(per_user_len)):
		for _ in range(per_user_len[user] - 1):
			splits.append('train')
		splits.append('test')
	rating['split'] = splits

	train_rating = rating[rating['split'] == 'train']
	test_rating = rating[rating['split'] == 'test']

	return train_rating.reset_index(drop=True), \
			test_rating.reset_index(drop=True)


def to_disk(rating, path):
	""" Write to disk following the given format. """
	with open(path, 'w') as fd:
		for r in rating.itertuples():
			rate = str(getattr(r, 'rating'))
			user = str(getattr(r, 'user'))
			item = str(getattr(r, 'item'))
			genre = str(getattr(r, 'genre'))

			line = rate + ' ' + user \
				+ ':1 '+ item + ':1 ' + genre + ':1'
			fd.write(line + '\n')


def main():
	random.seed(2019)
	np.random.seed(2019)

	rating_path = os.path.join(
		config.amazon_dir, 'reviews_' + config.dataset + '_5.json.gz')
	meta_path = os.path.join(
		config.amazon_dir, 'meta_' + config.dataset + '.json.gz')
	assert os.path.exists(rating_path), 'wrong rating file'
	assert os.path.exists(meta_path), 'wrong meta file'

	rating = get_df(rating_path)
	rating = rating.drop(columns=['reviewerName', 'reviewTime',
		'helpful', 'reviewText', 'summary', 'overall'])
	rating = rating.rename(index=str, columns={
		'reviewerID': 'user',
		'asin': 'item',
		'unixReviewTime': 'time'})
	rating = rating.sort_values(by=['user', 'time']) # sort users

	user_set = rating['user'].unique()
	item_set = rating['item'].unique()
	genre_set, item_to_genre = match_genre(meta_path, item_set)

	user_len = len(user_set)
	item_len = len(item_set)
	genre_len = len(genre_set)
	print("number of users {}, items {}, genres {}".format(
							user_len, item_len, genre_len))
	print("number of instances {}".format(len(rating)))

	user_dict = {user_set[i]: i for i in range(user_len)}
	item_dict = {item_set[i]: i for i in range(item_len)}
	genre_dict = {genre_set[i]: i for i in range(genre_len)}

	rating, item_to_genre_norm = norm_data(rating,
		user_dict, item_dict, genre_dict, item_to_genre)
	train_rating, test_rating = split_data(rating)

	# loading ratings as a dok matrix
	pst_mat = sp.dok_matrix(
				(user_len, item_len), dtype=np.float32)
	train_rating, pst_mat = ng_sample(train_rating, pst_mat,
			user_len, item_len, item_to_genre_norm, ng_num=2)
	test_rating, _ = ng_sample(test_rating, pst_mat,
			user_len, item_len, item_to_genre_norm, ng_num=99)

	if not os.path.exists(config.main_path):
		os.mkdir(config.main_path)
	to_disk(train_rating, os.path.join(
		config.main_path, config.dataset + '.train.libfm'))
	to_disk(test_rating, os.path.join(
		config.main_path, config.dataset + '.test.libfm'))

	print("All done.")


if __name__ == '__main__':
	main()
