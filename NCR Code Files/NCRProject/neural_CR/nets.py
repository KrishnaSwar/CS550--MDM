import torch
import numpy as np
from torch.nn.init import normal_ as normal_init
import torch.nn.functional as F

__all__ = ['NCR']

class NCR(torch.nn.Module):
    def __init__(self, n_users, n_items, emb_size=64, dropout=0.0, seed=2022, remove_double_not=False):
        super(NCR, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_size = emb_size
        self.dropout = dropout
        self.seed = seed

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.item_embeddings = torch.nn.Embedding(self.n_items, self.embed_size)
        self.user_embeddings = torch.nn.Embedding(self.n_users, self.embed_size)


        self.true_vector = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 0.1, size=self.embed_size).astype(np.float32)),
            requires_grad=False)  # gradient is false to disable the training of the vector

        self.not_layer_1 = torch.nn.Linear(self.embed_size, self.embed_size)

        self.not_layer_2 = torch.nn.Linear(self.embed_size, self.embed_size)

        self.or_layer_1 = torch.nn.Linear(2 * self.embed_size, self.embed_size)

        self.or_layer_2 = torch.nn.Linear(self.embed_size, self.embed_size)

        self.and_layer_1 = torch.nn.Linear(2 * self.embed_size, self.embed_size)

        self.and_layer_2 = torch.nn.Linear(self.embed_size, self.embed_size)

        self.encoder_layer_1 = torch.nn.Linear(2 * self.embed_size, self.embed_size)

        self.encoder_layer_2 = torch.nn.Linear(self.embed_size, self.embed_size)

        self.dropout_layer = torch.nn.Dropout(self.dropout)

        self.start_weights()
        self.remove_double_not = remove_double_not

    def start_weights(self):
        # not
        normal_init(self.not_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.bias, mean=0.0, std=0.01)
        # or
        normal_init(self.or_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.bias, mean=0.0, std=0.01)
        # and
        normal_init(self.and_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.bias, mean=0.0, std=0.01)
        # encoder
        normal_init(self.encoder_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.bias, mean=0.0, std=0.01)
        # embeddings
        normal_init(self.user_embeddings.weight, mean=0.0, std=0.01)
        normal_init(self.item_embeddings.weight, mean=0.0, std=0.01)

    def logic_not(self, vect):

        vect = F.relu(self.not_layer_1(vect))
        if self.training:
            vect = self.dropout_layer(vect)
        res = self.not_layer_2(vect)
        return res

    def logic_or(self, v1, v2, dim=1):
        vect = torch.cat((v1, v2), dim)
        vect = F.relu(self.or_layer_1(vect))
        if not self.training:
            pass
        else:
            vect = self.dropout_layer(vect)
        out = self.or_layer_2(vect)
        return out

    def logic_and(self, vector1, vector2, dim=1):
        vect = torch.cat((vector1, vector2), dim)
        vect = F.relu(self.and_layer_1(vect))
        if not self.training:
            pass
        else:
            vect = self.dropout_layer(vect)
        out = self.and_layer_2(vect)
        return out

    def enc(self, ui_vector):
        evec = F.relu(self.encoder_layer_1(ui_vector))
        if not self.training:
            pass
        else:
            evec = self.dropout_layer(evec)
        evec = self.encoder_layer_2(evec)
        return evec

    def forward(self, batch_data):
        u_id, item_ids, histories, history_feedbacks, neg_item_ids = batch_data


        u_eb = self.user_embeddings(u_id)
        i_eb = self.item_embeddings(item_ids)
        neg_item_embs = self.item_embeddings(neg_item_ids)



        rse = self.enc(torch.cat((u_eb, i_eb), dim=1))  # positive event vectors at the





        ex_u_eb = u_eb.view(u_eb.size(0), 1, u_eb.size(1))
        ex_u_eb = ex_u_eb.expand(u_eb.size(0), neg_item_embs.size(1),
                                                             u_eb.size(1))
        right_side_neg_events = self.enc(torch.cat((ex_u_eb, neg_item_embs), dim=2))  # negative event




        l_s_e = u_eb.view(u_eb.size(0), 1, u_eb.size(1))
        l_s_e = l_s_e.expand(u_eb.size(0), histories.size(1), u_eb.size(1))


        hi_it_eb = self.item_embeddings(histories)


        l_s_e = self.enc(torch.cat((l_s_e, hi_it_eb), dim=2))


        left_side_neg_events = self.logic_not(l_s_e)



        constraints = list([l_s_e])
        constraints.append(left_side_neg_events)

        feedback_tensor = history_feedbacks.view(history_feedbacks.size(0), history_feedbacks.size(1), 1)
        feedback_tensor = feedback_tensor.expand(history_feedbacks.size(0), history_feedbacks.size(1), self.embed_size)

        if not self.remove_double_not:
            l_s_e = feedback_tensor * l_s_e + (1 - feedback_tensor) * left_side_neg_events
        else:
            l_s_e = (1 - feedback_tensor) * l_s_e + feedback_tensor * left_side_neg_events

        if self.remove_double_not:
            pass
        else:
            l_s_e = self.logic_not(l_s_e)
        tmp_vector = l_s_e[:, 0]  # we take the first event of history

        shuffled_history_idx = list(range(1, histories.size(1)))  # this is needed to permute the order of the operands
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, l_s_e[:, i])
            constraints.append(tmp_vector.view(histories.size(0), -1, self.embed_size))  # this is done to have all the
        l_s_e = tmp_vector

        constraints.append(rse.view(histories.size(0), -1, self.embed_size))
        constraints.append(right_side_neg_events)  # this has already the correct shape, so it is not necessary to
        expression_events = self.logic_or(l_s_e, rse)
        constraints.append(expression_events.view(histories.size(0), -1, self.embed_size))

        exp_left_side_events = l_s_e.view(l_s_e.size(0), 1, l_s_e.size(1))
        exp_left_side_events = exp_left_side_events.expand(l_s_e.size(0), right_side_neg_events.size(1),
                                                           l_s_e.size(1))
        expression_neg_events = self.logic_or(exp_left_side_events, right_side_neg_events, dim=2)
        constraints.append(expression_neg_events)

        positive_predictions = F.cosine_similarity(expression_events, self.true_vector.view([1, -1])) * 10  # here the view is
        reshaped_expression_neg_events = expression_neg_events.reshape(expression_neg_events.size(0) *
                                                        expression_neg_events.size(1), expression_neg_events.size(2))
        negative_predictions = F.cosine_similarity(reshaped_expression_neg_events, self.true_vector.view([1, -1])) * 10
        negative_predictions = negative_predictions.reshape(expression_neg_events.size(0),
                                                            expression_neg_events.size(1))

        constraints = torch.cat(constraints, dim=1)

        constraints = constraints.view(constraints.size(0) * constraints.size(1), constraints.size(2))

        return positive_predictions, negative_predictions, constraints