import random
import torch

def shuffle_features_per_i(enc_hits, feature):
    """
    We shuffle the values for one feature at a time.

    Args:
    enc_hits: Fourier encoded features
    feature: feature we shuffle
        
    Returns:
    New Fourier encoded features with a feature that is shuffled

    """
    
    for i in range(feature, feature + 1) :

        idx = list(range(enc_hits.size(1))) # shuffle per hit in a bin for each bin
        #[0,1,2]
        random.shuffle(idx)
        #[1,2,0]
        idx = torch.tensor(idx)
        #tensor([1,2,0])
        enc_hits[:, :, i] = enc_hits[:, idx, i] #i-ième feature

    return enc_hits

def shuffle_features(enc_hits, situation) :

    """
    We shuffle the values for one feature at a time.

    Args:
    enc_hits: Fourier encoded features
    situation: name of the situation where we shuffle multiple features
        
    Returns:
    New Fourier encoded features

    """

    if situation == "except_x,y,z,r":
        liste = list(range(0,202)) + [206]
        for i in liste :

            idx = list(range(enc_hits.size(1))) #mélange par hit au sein d'un bin pour chaque bin
            #[0,1,2]
            random.shuffle(idx)
            #[1,2,0]
            idx = torch.tensor(idx)
            #tensor([1,2,0])
            enc_hits[:, :, i] = enc_hits[:, idx, i] #i-ième feature

    if situation == "x,y,z,r":
        for i in list(range(202, 206)) :

            idx = list(range(enc_hits.size(1))) #mélange par hit au sein d'un bin pour chaque bin
            #[0,1,2]
            random.shuffle(idx)
            #[1,2,0]
            idx = torch.tensor(idx)
            #tensor([1,2,0])
            enc_hits[:, :, i] = enc_hits[:, idx, i] #i-ième feature

    if situation == "high_freq":
        liste = list(range(4, 100)) + list(range(108, 150)) + list(range(178, 200)) 
        for i in liste :

            idx = list(range(enc_hits.size(1))) #mélange par hit au sein d'un bin pour chaque bin
            #[0,1,2]
            random.shuffle(idx)
            #[1,2,0]
            idx = torch.tensor(idx)
            #tensor([1,2,0])
            enc_hits[:, :, i] = enc_hits[:, idx, i] #i-ième feature

    if situation == "low_freq":
        liste = list(range(0, 4)) + list(range(100, 108)) + list(range(150, 178))
        for i in liste :

            idx = list(range(enc_hits.size(1))) #mélange par hit au sein d'un bin pour chaque bin
            #[0,1,2]
            random.shuffle(idx)
            #[1,2,0]
            idx = torch.tensor(idx)
            #tensor([1,2,0])
            enc_hits[:, :, i] = enc_hits[:, idx, i] #i-ième feature

    

    return enc_hits

