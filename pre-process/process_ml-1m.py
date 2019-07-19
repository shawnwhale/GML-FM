import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import scipy.sparse as sp

import config


# pre-known config
age_len, gender_len, occupation_len = 7, 2, 21

def parse_user(user_path):
	""" Convert user info into a list. """
	user_info = {}
	gender_info = {'F': 0, 'M': 1}
	age_info = {'1': 0, '18': 1,
			'25': 2, '35': 3, '45': 4, '50': 5, '56': 6}
	with open(user_path, 'r') as fd:
		line = fd.readline()
		while line:
			u_info = []
			entries = line.strip().split('::')
			u_info.append(gender_info[entries[1]])
			u_info.append(age_info[entries[2]])
			u_info.append(int(entries[3]))

			user_info[int(entries[0])-1] = u_info
			line = fd.readline()
	return user_info


def parse_item(item_path):
	""" Convert item info into a list. """
	item_info = {}
	with open(item_path, 'r', encoding='ISO-8859-1') as fd:
		line = fd.readline()
		while line:
			entries = line.strip().split('::')
			item_info[int(entries[0])] = entries[-1]
			line = fd.readline()
	return item_info


def parse_rating(rating_path, item_info):
	""" Reading rating file and convert to pandas df. """
	users, items, timestamps = [], [] ,[]
	with open(rating_path, 'r') as fd:
		line = fd.readline()
		while line:
			entries = line.split('::')
			item = int(entries[1])
			if item in item_info:
				users.append(int(entries[0]))
				items.append(item)
				timestamps.append(int(entries[3]))
			line = fd.readline()

	# filtering items without genre
	item_set = list(set(items))
	item_dict = {item_set[i]: i for i in range(len(item_set))}
	items = map(lambda x: item_dict[x], items)
	users = map(lambda x: x - 1, users)

	# re-norm item index and genre index
	item_info = {item_dict[i]: item_info[i]
				for i in item_info if i in item_dict}
	genre_set = list(set(item_info.values()))
	genre_dict = {genre_set[i]: i for i in range(len(genre_set))}
	item_info = {i: genre_dict[item_info[i]] for i in item_info}

	rating = pd.DataFrame(
		list(zip(users, items, timestamps)),
		columns=['user', 'item', 'timestamp'])
	return rating, item_info


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


def ng_sample(rating, pst_mat,
	user_len, item_len, user_info, item_info, ng_num):
	""" Negative sampling two instances for each one. """
	rates = []
	items, genres = [], []
	users, ages, genders, occupations = [], [], [], []

	for r in rating.itertuples():
		user = getattr(r, 'user')
		item = getattr(r, 'item')
		pst_mat[user, item] = 1

	for r in rating.itertuples():
		u = getattr(r, 'user')
		i = getattr(r, 'item')

		items.append(i)
		genres.append(item_info[i])

		u_info = user_info[u]
		users.append(u)
		genders.append(u_info[0])
		ages.append(u_info[1])
		occupations.append(u_info[2])

		rates.append(1.0)

		for t in range(ng_num):
			j = np.random.randint(item_len)
			while (u, j) in pst_mat:
				j = np.random.randint(item_len)
			items.append(j)
			genres.append(item_info[j])

			u_info = user_info[u]
			users.append(u)
			genders.append(u_info[0])
			ages.append(u_info[1])
			occupations.append(u_info[2])

			rates.append(-1.0)

	rating = pd.DataFrame(
		list(zip(users, items, genders,
		ages, occupations, genres, rates)),
		columns=['user', 'item', 'gender',
		'age', 'occupation', 'genre', 'rating'])

	point_len = user_len
	rating['item'] = rating['item'].apply(
							lambda x: x + point_len)
	point_len += item_len
	rating['gender'] = rating['gender'].apply(
							lambda x: x + point_len)
	point_len += gender_len
	rating['age'] = rating['age'].apply(
							lambda x: x + point_len)
	point_len += age_len
	rating['occupation'] = rating['occupation'].apply(
							lambda x: x + point_len)
	point_len += occupation_len
	rating['genre'] = rating['genre'].apply(
							lambda x: x + point_len)
	return rating, pst_mat


def to_disk(rating, path):
	""" Write to disk following the given format. """
	with open(path, 'w') as fd:
		for r in rating.itertuples():
			rate = str(getattr(r, 'rating'))
			user = str(getattr(r, 'user'))
			item = str(getattr(r, 'item'))
			gender = str(getattr(r, 'gender'))
			age = str(getattr(r, 'age'))
			occupation = str(getattr(r, 'occupation'))
			genre = str(getattr(r, 'genre'))

			line = rate + ' ' + user \
				+ ':1 '+ item + ':1 ' + gender \
				+ ':1 ' + age + ':1 ' + occupation \
				+ ':1 ' + genre + ':1'
			fd.write(line + '\n')


def main():
	np.random.seed(2019)

	user_path = os.path.join(config.ml-1m_dir, 'users.dat')
	movie_path = os.path.join(config.ml-1m_dir, 'movies.dat')
	rating_path = os.path.join(config.ml-1m_dir, 'ratings.dat')

	user_info = parse_user(user_path)
	item_info = parse_item(movie_path)
	rating, item_info = parse_rating(rating_path, item_info)

	user_len = len(rating['user'].unique())
	item_len = len(rating['item'].unique())
	assert user_len == len(user_info)
	assert item_len == len(item_info)
	print("number of users {}, movies {}, instances {}".format(
								user_len, item_len, len(rating)))

	# sorting rating data with userid and timestamp
	rating = rating.sort_values(by=['user', 'timestamp'])
	train_rating, test_rating = split_data(rating)

	pst_mat = sp.dok_matrix((user_len, item_len), dtype=np.float32)
	train_rating, pst_mat = ng_sample(
		train_rating, pst_mat,
		user_len, item_len,
		user_info, item_info, ng_num=2)
	test_rating, _ = ng_sample(
		test_rating, pst_mat,
		user_len, item_len,
		user_info, item_info, ng_num=99)

	if not os.path.exists(config.main_path):
		os.mkdir(config.main_path)
	to_disk(train_rating, os.path.join(config.main_path, 'ml-1m.train.libfm'))
	to_disk(test_rating, os.path.join(config.main_path, 'ml-1m.test.libfm'))

	print("All done.")


if __name__ == '__main__':
	main()
