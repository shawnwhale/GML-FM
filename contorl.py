
import torch
from sklearn import preprocessing
from mabs import Explorer,Thompson
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # test_users= torch.rand(6)
    # test_items = torch.rand(6)
    # for i in range(test_users.shape[0]):
    #     print(test_users[i].item()*test_items[i].item())

    # Load Data
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],
                              engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))

    # np.save("./item_tr.npy", item_id)
    # item_tr =  np.load("./item_tr.npy")
    # item_dic = {}   #key to old
    # reitem_dic = {}  # old to key
    # for i  in range(item_tr.shape[0]):
    #     item_dic[item_tr[i,1]] = item_tr[i,0]
    #     reitem_dic[item_tr[i, 0]] = item_tr[i, 1]
    # np.save("./item_dic.npy", item_dic)
    # np.save("./reitem_dic.npy", reitem_dic)

    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))

    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data
    negatives =  sample_generator.negatives
#    sample_generator.kendall_data()

    #收集矩阵
    train_ratings = sample_generator.train_ratings
    test_ratings = sample_generator.test_ratings

    # train_np = np.zeros((6040,3706))
    # test_np = np.zeros((6040, 3706))
    # for index, line in train.iterrows():
    #     user = int(line['userId'])
    #     item = int(line['itemId'])
    #     rating =line['rating']
    #     train_np[user,item] = rating
    # train = pd.DataFrame(train_np)
    #
    #
    # for index, line in test.iterrows():
    #     user = int(line['userId'])
    #     item = int(line['itemId'])
    #     rating =line['rating']
    #     test_np[user,item] = rating
    #
    #     train_np[user, item] = -1   #不让train学
    #
    # test = pd.DataFrame(test_np)

    test = np.load("./UCBtest.npy")
    train = np.load("./UCBtrain.npy")

    # 使用矩阵分解算法来估计评分
    # MF_estimate = Matrix_Factorization.Matrix_Factorization(K=10, epoch=2)
    # MF_estimate.fit(train)
    #
    # R_hat = MF_estimate.start()
    # hit_ratio, ndcg, ils ,kendall= MF_estimate.evaluate(R_hat,evaluate_data)
    # print(hit_ratio, ndcg, ils,kendall)

    user_dim = np.load("./user_dim.npy")
    item_embedding = np.load("./item_em.npy")
    # 标准化后归一化
    item_embedding = item_embedding.reshape(3706, 8)
    item_embedding_stan = preprocessing.StandardScaler().fit_transform(item_embedding)
    item_embedding_stan = preprocessing.MinMaxScaler().fit_transform(item_embedding_stan)
    item_embedding_stan = item_embedding_stan/np.sqrt(8)

    for index, line in negatives.iterrows():
        userId = int(line['userId'])
        negative_items = line['negative_items']
        negative_samples = line['negative_samples']
        for j in negative_samples:
            j = int(j)
            train[userId, j] = -1  #不让train学

    user_embedding = np.load("./user_em.npy").reshape(6040,8)
    user_embedding_stan = preprocessing.StandardScaler().fit_transform(user_embedding)
    user_embedding_stan = preprocessing.MinMaxScaler().fit_transform(user_embedding_stan)
    user_embedding_stan = user_embedding_stan  / np.sqrt(8)

    explorer = Explorer(item_embedding,user_embedding, negatives, train_ratings,test_ratings)

