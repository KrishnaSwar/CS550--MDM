import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .testeval import check_valid, chk_logic

__all__ = ['NCRTrainer']

logger = logging.getLogger(__name__)

class NCRTrainer(object):
    def __init__(self, net, learning_rate=0.001, l2_weight=1e-4, logic_reg_weight=0.01):
        self.network = net
        self.lr = learning_rate
        self.l2_weight = l2_weight
        self.r_weight = logic_reg_weight
        self.opt = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.l2_weight)

    def r_loss(self, constraints):
        n_vec = self.network.logic_not(self.network.true_vector)
        r_not_not_true = (1 - F.cosine_similarity(
            self.network.logic_not(self.network.logic_not(self.network.true_vector)), self.network.true_vector,dim=0))

        r_not_not_self = \
            (1 - F.cosine_similarity(self.network.logic_not(self.network.logic_not(constraints)), constraints)).mean()


        r_not_self = (1 + F.cosine_similarity(self.network.logic_not(constraints), constraints)).mean()


        r_not_not_not = \
            (1 + F.cosine_similarity(self.network.logic_not(self.network.logic_not(constraints)),
                                     self.network.logic_not(constraints))).mean()




        r_or_true = (1 - F.cosine_similarity(
            self.network.logic_or(constraints, self.network.true_vector.expand_as(constraints)),
            self.network.true_vector.expand_as(constraints))).mean()


        r_or_false = (1 - F.cosine_similarity(
            self.network.logic_or(constraints, n_vec.expand_as(constraints)), constraints)).mean()


        r_or_self = (1 - F.cosine_similarity(self.network.logic_or(constraints, constraints), constraints)).mean()


        r_or_not_self = (1 - F.cosine_similarity(
            self.network.logic_or(constraints, self.network.logic_not(constraints)),
            self.network.true_vector.expand_as(constraints))).mean()


        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.network.logic_or(self.network.logic_not(constraints), constraints),
            self.network.true_vector.expand_as(constraints))).mean()




        r_and_true = (1 - F.cosine_similarity(
            self.network.logic_and(constraints, self.network.true_vector.expand_as(constraints)), constraints)).mean()


        r_and_false = (1 - F.cosine_similarity(
            self.network.logic_and(constraints, n_vec.expand_as(constraints)),
            n_vec.expand_as(constraints))).mean()


        r_and_self = (1 - F.cosine_similarity(self.network.logic_and(constraints, constraints), constraints)).mean()


        r_and_not_self = (1 - F.cosine_similarity(
            self.network.logic_and(constraints, self.network.logic_not(constraints)),
            n_vec.expand_as(constraints))).mean()


        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.network.logic_and(self.network.logic_not(constraints), constraints),
            n_vec.expand_as(constraints))).mean()




        true_false = 1 + F.cosine_similarity(self.network.true_vector, n_vec, dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + r_not_not_not + \
                 r_or_true + r_or_false + r_or_self + r_or_not_self + r_or_not_self_inverse + true_false + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse

        return r_loss

    def loss_function(self, p_pred, negative_preds, constraints):
        p_pred = p_pred.view(p_pred.size(0), 1)
        p_pred = p_pred.expand(p_pred.size(0), negative_preds.size(1))
        loss = -(p_pred - negative_preds).sigmoid().log().sum()  # this is the formula in the paper

        r_loss = self.r_loss(constraints)

        return loss + self.r_weight * r_loss

    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=check_valid(chk_logic),
              num_epochs=100,
              at_least=20,
              e_stop=5,
              save_path="../saved_models/best_ncr_model.json",
              verbose=1):
        bval = 0.0
        e_count = 0
        early_stop_flag = False
        if e_stop > 1:
            early_stop_flag = True
        try:
            for epoch in range(1, num_epochs + 1):
                self.train_epoch(epoch, train_data, verbose)
                if valid_data is not None:
                    assert valid_metric is not None, \
                                "In case of validation 'valid_metric' must be provided"
                    res_val = valid_func(self, valid_data, valid_metric)
                    mval = np.mean(res_val)
                    std_err_val = np.std(res_val) / np.sqrt(len(res_val))
                    logger.info('| epoch %d | %s %.3f (%.4f) |',
                                epoch, valid_metric, mval, std_err_val)
                    if mval > bval:
                        bval = mval
                        self.save_model(save_path, epoch)

                        e_count = 0

                    else:


                        if epoch < at_least or not early_stop_flag:
                            continue
                        e_count += 1
                        if e_count != e_stop:
                            continue
                        logger.info('Traing stopped at epoch %d due to early stopping', epoch)
                        break
        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')


    def train_epoch(self, epoch, train_loader, verbose=1):
        self.network.train()
        t_loss = 0
        p_loss = 0
        e_time = time.time()
        st_tim = time.time()
        log_delay = max(10, len(train_loader) // 10**verbose)

        for batch_idx, batch_data in enumerate(train_loader):
            p_loss += self.train_batch(batch_data)
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - st_tim
                logger.info('| epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                            epoch, (batch_idx+1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            p_loss / log_delay)
                t_loss += p_loss
                p_loss = 0.0
                st_tim = time.time()
        total_loss = (t_loss + p_loss) / len(train_loader)
        time_diff = time.time() - e_time
        logger.info("| epoch %d | loss %.4f | total time: %.2fs |", epoch, total_loss, time_diff)

    def train_batch(self, batch_data):
        self.opt.zero_grad()
        p_pred, n_pred, constraints = self.network(batch_data)
        loss = self.loss_function(p_pred, n_pred, constraints)
        loss.backward()


        self.opt.step()
        return loss.item()

    def predict(self, batch_data):
        self.network.eval()
        with torch.no_grad():
            p_pred, n_pred, _ = self.network(batch_data)
        return p_pred, n_pred

    def save_model(self, filepath, cur_epoch):
        logger.info("Saving model checkpoint to %s...", filepath)
        torch.save({'epoch': cur_epoch,
                 'state_dict': self.network.state_dict(),
                 'optimizer': self.opt.state_dict()
                }, filepath)
        logger.info("Model checkpoint saved!")

    def m_load(self, filepath):
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        self.network.load_state_dict(checkpoint['state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        logger.info("Model checkpoint loaded!")
        return checkpoint

    def test(self, test_loader, test_metrics=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'], n_times=10):
        metric_dict = {}
        for i in range(n_times):

            evaluation_dict = chk_logic(self, test_loader, test_metrics)
            for metric in evaluation_dict:
                if metric not in metric_dict:
                    metric_dict[metric] = {}
                metric_mean = np.mean(evaluation_dict[metric])
                metric_std_err_val = np.std(evaluation_dict[metric]) / np.sqrt(len(evaluation_dict[metric]))
                if "mean" not in metric_dict[metric]:
                    metric_dict[metric]["mean"] = metric_mean
                    metric_dict[metric]["std"] = metric_std_err_val
                else:
                    metric_dict[metric]["mean"] += metric_mean
                    metric_dict[metric]["std"] += metric_std_err_val

        for metric in metric_dict:
            logger.info('%s: %.3f (%.4f)', metric, metric_dict[metric]["mean"] / n_times,
                        metric_dict[metric]["std"] / n_times)