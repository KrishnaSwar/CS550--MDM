import argparse
from neural_CR.dataSet import DataGenerator
from neural_CR.dataloader import DataSampler
from neural_CR.nets import NCR
from neural_CR.models import NCRTrainer
from neural_CR.testeval import check_valid, chk_logic
import torch
import pandas as pd
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set pytorch device for computation

def main():

    arg_par = argparse.ArgumentParser(description='Experiment')
    arg_par.add_argument('--threshold', type=int, default=4,
                             )
    arg_par.add_argument('--order', type=bool, default=True,
                             )
    arg_par.add_argument('--leave_n', type=int, default=1,
                             )
    arg_par.add_argument('--keep_n', type=int, default=5,
                             )
    arg_par.add_argument('--max_history_length', type=int, default=5,
                             )
    arg_par.add_argument('--n_neg_train', type=int, default=1,
                             )
    arg_par.add_argument('--n_neg_val_test', type=int, default=100,
                             )
    arg_par.add_argument('--training_batch_size', type=int, default=128,
                             )
    arg_par.add_argument('--val_test_batch_size', type=int, default=128 * 2,
                             )
    arg_par.add_argument('--seed', type=int, default=2022,
                             )
    arg_par.add_argument('--emb_size', type=int, default=64,
                             )
    arg_par.add_argument('--dropout', type=float, default=0.0,
                             )
    arg_par.add_argument('--lr', type=float, default=0.001,
                             )
    arg_par.add_argument('--l2', type=float, default=0.0001,
                             )
    arg_par.add_argument('--r_weight', type=float, default=0.1,
                             )
    arg_par.add_argument('--val_metric', type=str, default='ndcg@5',
                             )
    arg_par.add_argument('--test_metrics', type=list, default=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'],
                             )
    arg_par.add_argument('--n_epochs', type=int, default=100,
                             )
    arg_par.add_argument('--early_stop', type=int, default=5,
                             )
    arg_par.add_argument('--at_least', type=int, default=20,
                             )
    arg_par.add_argument('--save_load_path', type=str, default="saved-models/best_model.json",
                             )
    arg_par.add_argument('--n_times', type=int, default=10,
                             )
    arg_par.add_argument('--dataset', type=str, default="movielens_100k",
                             )
    arg_par.add_argument('--test_only', type=bool, default=False,
                             )
    arg_par.add_argument('--premise_threshold', type=int, default=0,
                             )
    arg_par.add_argument('--remove_double_not', type=bool, default=False,
                             )
    init_args, init_extras = arg_par.parse_known_args()


    if init_args.dataset == "movielens_100k":
        raw_dataset = pd.read_csv("datasets/movielens-100k/movielens_100k.csv")



    dataset = DataGenerator(raw_dataset)
    dataset.manupulate_data(threshold=init_args.threshold, order=init_args.order, leave_n=init_args.leave_n,
                            keep_n=init_args.keep_n, max_history_length=init_args.max_history_length,
                            premise_threshold=init_args.premise_threshold)

    if init_args.test_only:
        pass
    else:
        t_load = DataSampler(dataset.t_set, dataset.user_item_matrix, n_neg_samples=init_args.n_neg_train,
                                   batch_size=init_args.training_batch_size, shuffle=True, seed=init_args.seed,
                                   device=device)
        v_loa = DataSampler(dataset.v_set, dataset.user_item_matrix,
                                 n_neg_samples=init_args.n_neg_val_test, batch_size=init_args.val_test_batch_size,
                                 shuffle=False, seed=init_args.seed, device=device)

    t_loa = DataSampler(dataset.test_set, dataset.user_item_matrix, n_neg_samples=init_args.n_neg_val_test,
                              batch_size=init_args.val_test_batch_size, shuffle=False, seed=init_args.seed,
                              device=device)

    ncr_net = NCR(dataset.all_users, dataset.all_items, emb_size=init_args.emb_size, dropout=init_args.dropout,
                  seed=init_args.seed, remove_double_not=init_args.remove_double_not).to(device)

    ncr_model = NCRTrainer(ncr_net, learning_rate=init_args.lr, l2_weight=init_args.l2,
                           logic_reg_weight=init_args.r_weight)

    if init_args.test_only:
        pass
    else:
        ncr_model.train(t_load, valid_data=v_loa, valid_metric=init_args.val_metric,
                        valid_func=check_valid(chk_logic), num_epochs=init_args.n_epochs,
                        at_least=init_args.at_least, e_stop=init_args.early_stop,
                        save_path=init_args.save_load_path, verbose=1)

    ncr_model.m_load(init_args.save_load_path)

    ncr_model.test(t_loa, test_metrics=init_args.test_metrics, n_times=init_args.n_times)

if __name__ == '__main__':
    main()