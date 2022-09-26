import pandas as pd
from scipy import sparse
import copy

__all__ = ['DataGenerator']

class DataGenerator(object):

    def __init__(self, original_dataset):
        self.data_set = original_dataset

        self.all_users = self.data_set['userID'].nunique()
        self.all_items = self.data_set['itemID'].nunique()

        self.user_item_matrix = self.sparse_matrix_calculation()


    def sparse_matrix_calculation(self):
        group = self.data_set.groupby("userID")
        rows, cols = [], []
        values = []
        for i, (_, g) in enumerate(group):
            u = list(g['userID'])[0]
            items = set(list(g['itemID']))
            rows.extend([u] * len(items))
            cols.extend(list(items))
            values.extend([1] * len(items))
        return sparse.csr_matrix((values, (rows, cols)), (self.all_users, self.all_items))

    def manupulate_data(self, threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5, premise_threshold=0):

        self.req_data = self.data_set.copy()
        self.req_data['rating'][self.req_data['rating'] < threshold] = 0
        self.req_data['rating'][self.req_data['rating'] >= threshold] = 1

        if order:
            self.req_data = self.req_data.sort_values(by=['timestamp', 'userID', 'itemID']).reset_index(drop=True)

        self.rem_data(leave_n, keep_n)
        self.past_data(max_hist_length=max_history_length, prem_thrhold=premise_threshold)


    def rem_data(self, leave_n=1, keep_n=5):

        req_dset = []

        c_data = self.req_data.copy()
        for unique_id, set in c_data.groupby('userID'):
            req, pos = 0, -1
            for idx in set.index:
                if not set.loc[idx, 'rating'] <= 0:
                    pos = idx
                    req += 1
                    if req < keep_n:
                        pass
                    else:
                        break
            if not pos <= 0:
                req_dset.append(set.loc[:pos])
        req_dset = pd.concat(req_dset)

        c_data = c_data.drop(req_dset.index)


        test_set = []
        for unique_id, set in c_data.groupby('userID'):
            req, pos = 0, -1
            for idx in reversed(set.index):
                if set.loc[idx, 'rating'] <= 0:
                    continue
                pos = idx
                req += 1
                if req >= leave_n:
                    break
            if pos <= 0:
                continue
            test_set.append(set.loc[pos:])
        test_set = pd.concat(test_set)
        c_data = c_data.drop(test_set.index)

        v_set = []
        for unique_id, set in c_data.groupby('userID'):
            req, pos = 0, -1
            for idx in reversed(set.index):
                if set.loc[idx, 'rating'] > 0:
                    pos = idx
                    req += 1
                    if req >= leave_n:
                        break

            if pos > 0:
                v_set.append(set.loc[pos:])
        v_set = pd.concat(v_set)
        c_data = c_data.drop(v_set.index)

        # The remaining data (after removing validation and test) are all in training data
        self.t_set = pd.concat([req_dset, c_data])
        self.v_set, self.test_set = v_set.reset_index(drop=True), test_set.reset_index(drop=True)

    def past_data(self, max_hist_length=5, prem_thrhold=0):

        past_hashmap = {}
        comments_hashmap = {}
        for df in [self.t_set, self.v_set, self.test_set]:
            history = []
            fb = []

            hist_len = [] # each element of this list indicates the number of history items of a single interaction
            uids, iids, feedbacks = df['userID'].tolist(), df['itemID'].tolist(), df['rating'].tolist()
            for i, unique_id in enumerate(uids):
                iid, feedback = iids[i], feedbacks[i]

                if unique_id in past_hashmap:
                    pass
                else:
                    past_hashmap[unique_id] = []
                    comments_hashmap[unique_id] = []

                temp = copy.deepcopy(past_hashmap[unique_id]) if max_hist_length == 0 else past_hashmap[unique_id][-max_hist_length:]

                past_feedback = copy.deepcopy(comments_hashmap[unique_id]) if max_hist_length == 0 else comments_hashmap[unique_id][-max_hist_length:]

                history.append(temp)
                fb.append(past_feedback)
                hist_len.append(len(temp))

                past_hashmap[unique_id].append(iid)
                comments_hashmap[unique_id].append(feedback)

            df['history'] = history
            df['history_feedback'] = fb
            df['history_length'] = hist_len




        if prem_thrhold != 0:
            self.t_set = self.t_set[self.t_set.history_length > prem_thrhold]
            self.v_set = self.v_set[self.v_set.history_length > prem_thrhold]
            self.test_set = self.test_set[self.test_set.history_length > prem_thrhold]

        self.data_cleaning()


    def data_cleaning(self):
        self.t_set = self.t_set[self.t_set['rating'] > 0].reset_index(drop=True)
        self.t_set = self.t_set[self.t_set['history_feedback'].map(len) > 0].reset_index(drop=True)
        self.v_set = self.v_set[self.v_set['rating'] > 0].reset_index(drop=True)
        self.v_set = self.v_set[self.v_set['history_feedback'].map(len) > 0].reset_index(drop=True)
        self.test_set = self.test_set[self.test_set['rating'] > 0].reset_index(drop=True)
        self.test_set = self.test_set[self.test_set['history_feedback'].map(len) > 0].reset_index(drop=True)