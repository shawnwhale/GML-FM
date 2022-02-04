import numpy as np
import torch
import config
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k=10, refeatures_map = {}):
	HR, NDCG = [], []
	ILS = []
	Kendall = []
	item_dim = np.load('./item_dim.npy')
	kendall_np = np.load('./user_dim.npy')
	features_tran = np.load("./features_tran.npy",allow_pickle=True).tolist()
	item_entropy_dic = {}
	sumrow = 0

	for features, feature_values, label in test_loader:
		features = features.cuda()
		#features = tensor([[   0,  597,    2,    3,    4,  598],
        #[   0, 3132,    2,    3,    4,   32],略
		feature_values = feature_values.cuda()
		#每次读一百条，第0条是目标，选中的是recommends里面的编号项目
		predictions = model(features, feature_values)
		_, indices = torch.topk(predictions, top_k)
		# recommends = torch.take(
		# 		item, indices).cpu().numpy().tolist()

		recommends = indices.cpu().numpy().tolist()
		item_seq = []
		for num in recommends:
			entries = features[num]
			real_seq = int(features_tran[int(entries[1])]) - config.item_add
			item_seq.append(real_seq)
		gt_item = 0
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

		num_k = top_k
		items_vec = item_dim[item_seq, :]
		similarity_matrix = cosine_similarity(items_vec)
		similarity_matrix_df = pd.DataFrame(similarity_matrix)
		similarity_matrix_df = (similarity_matrix_df + 1) * 0.5  # 对余弦相似度归一化处理
		L = np.tril(similarity_matrix_df, -1)
		one_ils = 2 * L.sum() / (num_k * (num_k - 1))
		ILS.append(one_ils)

		#计算kendall
		user_id = int(refeatures_map[int(features[0,0])])
		items_vec = item_dim[item_seq, :]  # 获得推荐项目的向量
		items_vec_sum = np.zeros(shape=(1, 18))
		for vec in items_vec:
			items_vec_sum += vec
		temp = kendall_np[user_id, :]
		data = np.vstack((items_vec_sum, temp))
		data = pd.DataFrame(data)
		data = pd.DataFrame(data.values.T)
		data = data.corr('kendall')
		one_kendall = data.iloc[0, 1]
		Kendall.append(one_kendall)

		#计算entropy
		for num in recommends:
			entries = features[num]
			real_seq = int(features_tran[int(entries[1])]) - config.item_add
			sumrow = sumrow + 1
			if real_seq in item_entropy_dic:
				item_entropy_dic[real_seq] = item_entropy_dic[real_seq] + 1
			else:
				item_entropy_dic[real_seq] = 1

	entropy = 0
	for i in range(3706):
		if i in item_entropy_dic:
			temp = item_entropy_dic[i] / sumrow
			entropy = entropy + temp * math.log(temp)
		else:
			continue
	ENTROPY = -entropy

	return np.mean(HR), np.mean(NDCG), HR, NDCG, np.mean(ILS), ILS, np.mean(Kendall), Kendall,ENTROPY


def metric_rmse(model, dataloader):
	RMSE = np.array([], dtype=np.float32)
	for features, feature_values, label in dataloader:
		features = features.cuda()
		feature_values = feature_values.cuda()
		label = label.cuda()

		prediction = model(features, feature_values)
		prediction = prediction.clamp(min=-1.0, max=1.0)
		SE = (prediction - label).pow(2)
		RMSE = np.append(RMSE, SE.detach().cpu().numpy())

	return np.sqrt(RMSE.mean())
