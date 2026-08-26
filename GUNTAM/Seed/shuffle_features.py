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

    for i in range(feature, feature + 1):

        idx = list(range(enc_hits.size(1)))  # shuffle per hit in a bin for each bin
        # [0,1,2]
        random.shuffle(idx)
        # [1,2,0]
        idx = torch.tensor(idx)
        # tensor([1,2,0])
        enc_hits[:, :, i] = enc_hits[:, idx, i]  # i-ième feature

    return enc_hits


def shuffle_features(enc_hits, situation):
    """
    We shuffle the values for one feature at a time.

    Args:
    enc_hits: Fourier encoded features
    situation: list of the features we want to shuffle alltogether

    Returns:
    New Fourier encoded features

    """

    for i in situation:

        idx = list(range(enc_hits.size(1)))  # mélange par hit au sein d'un bin pour chaque bin
        # [0,1,2]
        random.shuffle(idx)
        # [1,2,0]
        idx = torch.tensor(idx)
        # tensor([1,2,0])
        enc_hits[:, :, i] = enc_hits[:, idx, i]  # i-ième feature

    return enc_hits
