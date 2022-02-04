
import numpy as np
import pandas as pd

if __name__ == '__main__':
    train_rating=  np.load('./train_rating.npy')
    test_rating=    np.load('./test_rating.npy')
    features_tran = np.load("./features_tran.npy", allow_pickle=True).tolist()
    features_map = np.load('./features_map.npy', allow_pickle=True).tolist()
    refeatures_map = {}
    for key in features_map.keys():
        value = features_map[key]
        refeatures_map[value] = key
    print("!!!")