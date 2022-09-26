
import logging
import bottleneck as bn
import numpy as np

__all__ = ['EvalMetrics']

logger = logging.getLogger(__name__)

class EvalMetrics(object):
    @staticmethod
    def calculation(pred_scores, ground_truth, metrics_list):
        hashmap_res = {}
        for metric in metrics_list:

            try:
                if "@" in metric:
                    met, k = metric.split("@")
                    met_foo = getattr(EvalMetrics, "%s_at_k" % met.lower())
                    hashmap_res[metric] = met_foo(pred_scores, ground_truth, int(k))
                else:
                    hashmap_res[metric] = getattr(EvalMetrics, metric)(pred_scores, ground_truth)
            except AttributeError:
                logger.warning("Skipped unknown metric '%s'.", metric)
        return hashmap_res

    @staticmethod
    def mae_at_k(pred_scores, ground_truth,k = 100):
        assert pred_scores.shape == ground_truth.shape, \
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)
        index = bn.argpartition(-pred_scores, k - 1, axis=1)
        pred_bin = np.zeros_like(pred_scores, dtype=bool)
        pred_bin[np.arange(pred_scores.shape[0])[:, np.newaxis], index[:, :k]] = True
        return np.mean(np.sum(ground_truth - pred_bin))

    @staticmethod
    def mse_at_k(p_sco, g_tru, k = 100):
        assert p_sco.shape == g_tru.shape, \
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(p_sco.shape[1], k)
        index = bn.argpartition(-p_sco, k - 1, axis=1)
        pred_bin = np.zeros_like(p_sco, dtype=bool)
        pred_bin[np.arange(p_sco.shape[0])[:, np.newaxis], index[:, :k]] = True
        return np.mean(np.sum((g_tru - pred_bin) ** 2)) ** 0.5

    @staticmethod
    def precision_at_k(p_sco, g_tru, k=100):
        assert p_sco.shape == g_tru.shape, \
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(p_sco.shape[1], k)
        index = bn.argpartition(-p_sco, k - 1, axis=1)
        pred_bin = np.zeros_like(p_sco, dtype=bool)
        pred_bin[np.arange(p_sco.shape[0])[:, np.newaxis], index[:, :k]] = True
        X_true_binary = (g_tru > 0)
        x_f_binary = (pred_bin > 0)
        num = (np.logical_and(X_true_binary, pred_bin).sum(axis=1)).astype(np.float32)
        deno = (np.logical_and(np.logical_not(x_f_binary), pred_bin).sum(axis=1)).astype(np.float32)
        precision = num / np.minimum(k,x_f_binary.sum(axis=1))

        return precision

    @staticmethod
    def ndcg_at_k(p_sco, g_tru, k=100):
        assert p_sco.shape == g_tru.shape,\
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(p_sco.shape[1], k)
        n_users = p_sco.shape[0]
        ind_top = bn.argpartition(-p_sco, k - 1, axis=1)
        topk_part = p_sco[np.arange(n_users)[:, np.newaxis], ind_top[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = ind_top[np.arange(n_users)[:, np.newaxis], idx_part]
        tp = 1. / np.log2(np.arange(2, k + 2))
        DCG = (g_tru[np.arange(n_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(int(n), k)]).sum() for n in g_tru.sum(axis=1)])
        return DCG / IDCG

    @staticmethod
    def recall_at_k(pred_scores, ground_truth, k=100):
        assert pred_scores.shape == ground_truth.shape,\
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(pred_scores.shape[1], k)
        idx = bn.argpartition(-pred_scores, k-1, axis=1)
        pred_scores_binary = np.zeros_like(pred_scores, dtype=bool)
        pred_scores_binary[np.arange(pred_scores.shape[0])[:, np.newaxis], idx[:, :k]] = True
        X_true_binary = (ground_truth > 0)
        num = (np.logical_and(X_true_binary, pred_scores_binary).sum(axis=1)).astype(np.float32)
        recall = num / np.minimum(k, X_true_binary.sum(axis=1))
        return recall

    @staticmethod
    def hit_at_k(p_sco, g_tru, k=100):
        assert p_sco.shape == g_tru.shape,\
            "'pred_scores' and 'ground_truth' must have the same shape."
        k = min(p_sco.shape[1], k)
        idx = bn.argpartition(-p_sco, k - 1, axis=1)
        p_bin = np.zeros_like(p_sco, dtype=bool)
        p_bin[np.arange(p_sco.shape[0])[:, np.newaxis], idx[:, :k]] = True
        X_true_binary = (g_tru > 0)
        num = (np.logical_and(X_true_binary, p_bin).sum(axis=1)).astype(np.float32)
        return num > 0
