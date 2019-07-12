import numpy as np
import torch


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k=10):
	HR, NDCG = [], []

	for features, feature_values, label in test_loader:
		features = features.cuda()
		feature_values = feature_values.cuda()

		predictions = model(features, feature_values)
		_, indices = torch.topk(predictions, top_k)
		# recommends = torch.take(
		# 		item, indices).cpu().numpy().tolist()

		recommends = indices.cpu().numpy().tolist()
		gt_item = 0
		HR.append(hit(gt_item, recommends))
		NDCG.append(ndcg(gt_item, recommends))

	return np.mean(HR), np.mean(NDCG), HR, NDCG


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
