import numpy as np
import torch
import random

__all__ = ['Sampler', 'DataSampler']

class Sampler(object):
    def __init__(self, *args, **kargs):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class DataSampler(Sampler):
    def __init__(self,
                 data,
                 user_item_matrix,
                 n_neg_samples=1,
                 batch_size=128,
                 shuffle=True,
                 seed=2022,
                 device='cuda:0'):
        super(DataSampler, self).__init__()
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.n_neg_samples = n_neg_samples
        self.b_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        np.random.seed(self.seed)
        random.seed(self.seed)

    def __len__(self):
        dataset_size = 0
        len = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group
        for i, (_, l) in enumerate(len):
            dataset_size += int(np.ceil(l.shape[0] / self.b_size))
        return dataset_size

    def __iter__(self):
        length = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group


        for i, (_, l) in enumerate(length):

            user_grp = np.array(list(l['userID']))
            itm_grp = np.array(list(l['itemID']))
            pst_his = np.array(list(l['history']))
            feed_grp = np.array(list(l['history_feedback']))

            n = user_grp.shape[0]
            idxlist = list(range(n))
            if self.shuffle:
                np.random.shuffle(idxlist)

            for _, start_idx in enumerate(range(0, n, self.b_size)):
                end_idx = min(start_idx + self.b_size, n)
                b_usr = torch.from_numpy(user_grp[idxlist[start_idx:end_idx]])
                b_itm = torch.from_numpy(itm_grp[idxlist[start_idx:end_idx]])
                b_his = torch.from_numpy(pst_his[idxlist[start_idx:end_idx]])
                b_feedb = torch.from_numpy(feed_grp[idxlist[start_idx:end_idx]])


                batch_user_item_matrix = self.user_item_matrix[b_usr].toarray()

                batch_user_unseen_items = 1 - batch_user_item_matrix

                negative_items = []
                for u in range(b_usr.size(0)):
                    u_unseen_items = batch_user_unseen_items[u].nonzero()[0]


                    rnd_negatives = u_unseen_items[random.sample(range(u_unseen_items.shape[0]), self.n_neg_samples)]

                    negative_items.append(rnd_negatives)
                batch_negative_items = torch.tensor(negative_items)

                yield b_usr.to(self.device), b_itm.to(self.device), b_his.to(self.device), \
                      b_feedb.to(self.device), batch_negative_items.to(self.device)