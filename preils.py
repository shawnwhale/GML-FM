import pandas as pd
import math
import pandas as pd
import numpy as np

if __name__ == '__main__':
       # Load Data
              ml1m_dir = './ml-1m/movies.dat'

              genre_dic = {}
              genre_dicT = {}
              genre_num=0
              genre=[]
              with open(ml1m_dir, 'r', encoding='ISO-8859-1') as fd:
                  line = fd.readline()
                  while line:
                      entries = line.strip().split('::')
                      genre.append(entries[2])
                      line = fd.readline()
              for i in range(len(genre)):

                     genres = genre[i].split('|')
                     for one in genres:
                            if genre_dic.get(one):
                                   pass
                            else:
                                   genre_num+=1
                                   genre_dic[one] = genre_num
                                   genre_dicT[genre_num] = one

              # 存np
              genre_np=np.array([],dtype= str )
              for j in range(genre_num):
                     genre_np =np.append(genre_np,genre_dicT[j+1])
              np.save('./output_genre',genre_np)
              np_out = np.load('./output_genre.npy')
              print(np_out)

              #生成多样性矩阵,此处已经被转为item_dict即set后的数值
              np_out = np.load('./output_genre.npy')
              item_map = np.load('./item_dict.npy',allow_pickle=True).tolist()
              dim_np = np.zeros(shape=(3706, 18))
              with open(ml1m_dir, 'r', encoding='ISO-8859-1') as fd:
                  line = fd.readline()
                  while line:
                      entries = line.strip().split('::')
                      if int(entries[0]) in item_map:
                           num = item_map[int(entries[0])]
                           genre = entries[2]
                           genres = genre.split('|')
                           for one in genres:
                               index = genre_dic[one]
                               dim_np[num, index - 1] = 1
                      line = fd.readline()
              np.save('./item_dim', dim_np)
              print(dim_np)

