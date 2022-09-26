import torch
from functools import partial
import inspect
import numpy as np
from .EvalMetrics import EvalMetrics
import random

__all__ = ['check_valid', 'chk_logic', 'res_eval', 'add_one']

class check_valid(object):
    def __init__(self, func, **kwargs):
        self.func_name = func.__name__
        self.function = partial(func, **kwargs)

        args = inspect.getfullargspec(self.function).args
        assert args == ["model", "test_loader", "metric_list"],\
            "A (partial) validation function must have the following kwargs: model, test_loader and\
            metric_list"

    def __call__(self, model, test_loader, metric):
        return self.function(model, test_loader, [metric])[metric]

    def __str__(self):
        kwdefargs = inspect.getfullargspec(self.function).kwonlydefaults
        return "ValidFunc(fun='%s', params=%s)" %(self.func_name, kwdefargs)

    def __repr__(self):
        return str(self)


def chk_logic(model, test_loader, metric_list):
    output_res = {m:[] for m in metric_list}
    for b_index, b_data in enumerate(test_loader):
        predic_pos, predic_neg = model.predict(b_data)
        predic_pos = predic_pos.view(predic_pos.size(0), 1)
        expec_scores = torch.cat((predic_pos, predic_neg), dim=1)

        g_tru = np.zeros(expec_scores.size())
        g_tru[:, 0] = 1  # the positive item is always in the first column of pred_scores, as we said before
        expec_scores = expec_scores.cpu().numpy()
        res = EvalMetrics.calculation(expec_scores, g_tru, metric_list)
        for m in res:
            output_res[m].append(res[m])

    for m in output_res:
        output_res[m] = np.concatenate(output_res[m])
    return output_res

def res_eval(model, test_loader, metric_list):
    results = {m:[] for m in metric_list}
    for _, (data_tr, heldout) in enumerate(test_loader):
        dat_ten = data_tr.view(data_tr.shape[0], -1)
        rec_b = model.predict(dat_ten)[0].cpu().numpy()
        heldout = heldout.view(heldout.shape[0], -1).cpu().numpy()
        res = EvalMetrics.calculation(rec_b, heldout, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results


def add_one(model, test_loader, metric_list, r=1000):
    results = {m:[] for m in metric_list}
    for _, (data_tr, heldout) in enumerate(test_loader):
        tot = set(range(heldout.shape[1]))
        d_ten = data_tr.view(data_tr.shape[0], -1)
        recon_batch = model.predict(d_ten)[0].cpu().numpy()
        heldout = heldout.view(heldout.shape[0], -1).cpu().numpy()

        users, items = heldout.nonzero()
        rows = []
        for u, i in zip(users, items):
            rnd = random.sample(tot - set(list(heldout[u].nonzero()[0])), r)
            rows.append(list(recon_batch[u][[i] + list(rnd)]))

        expec = np.array(rows)
        g_t = np.zeros_like(expec)
        g_t[:, 0] = 1
        res = EvalMetrics.calculation(expec, g_t, metric_list)
        for m in res:
            results[m].append(res[m])

    for m in results:
        results[m] = np.concatenate(results[m])
    return results