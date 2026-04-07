import torch
import torch.nn as nn
from torch import Tensor
import random


données = t = torch.rand(3, 3, 4)

print("données sans shuffle", données)

# tensor([[[0.0152, 0.6606, 0.2826, 0.2578], #premier hit (4 features)
#          [0.8966, 0.5466, 0.5110, 0.1936], #deuxième hit (4 features)
#          [0.0927, 0.5199, 0.0343, 0.8172]], #troisième hit (4 features)

# = 1 bins

#         [[0.7012, 0.9183, 0.4309, 0.5190],
#          [0.1203, 0.9561, 0.6418, 0.9140],
#          [0.9123, 0.4539, 0.0260, 0.2727]],

#         [[0.6125, 0.2202, 0.5215, 0.3170],
#          [0.9977, 0.0897, 0.0943, 0.7777],
#          [0.1298, 0.1114, 0.7804, 0.3452]]])

#print(données.shape) 
#torch.Size([3, 2, 4])
#3 bins, 3 hits par bins, 4 features par hits

def shuffle_features(enc_hits, min_features, max_features) :

    #enc_hits = encoded_hits dans SeedTransformer
    #max_features = 207 ou 256

    for i in range(min_features, max_features + 1) :

        idx = list(range(enc_hits.size(1))) #mélange par hit au sein d'un bin pour chaque bin
        #[0,1,2]
        random.shuffle(idx)
        #[1,2,0]
        idx = torch.tensor(idx)
        #tensor([1,2,0])
        enc_hits[:, :, i] = enc_hits[:, idx, i] #i-ième feature

    return enc_hits

#encoded_h = shuffle_1p1(données, 2, 2) #shuffle seulement la 3ème feature
#feature commence à partir de 0
#range(2, 2 + 1) = [2]
#range(3, 2 + 1) = []

encoded_h = shuffle_features(données, 2, 3) #shuffle la 3ème et la 4ème features

print("données après shuffle", encoded_h)

