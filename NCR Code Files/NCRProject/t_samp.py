from neural_CR.dataSet import DataGenerator
from neural_CR.dataloader import DataSampler
from neural_CR.nets import NCR
import pandas as pd
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set pytorch device for computation

if __name__ == '__main__':
    raw_dataset = pd.read_csv("datasets/movielens-100k/movielens_100k.csv")

    dataset = DataGenerator(raw_dataset)

    dataset.manupulate_data(premise_threshold=0, max_history_length=5)

    train_loader = DataSampler(dataset.t_set, dataset.user_item_matrix, n_neg_samples=1, batch_size=128,
                               shuffle=True, seed=2022, device=device)

    ncr = NCR(dataset.all_users, dataset.all_items, emb_size=64, dropout=0.0, seed=2022).to(device)

    for batch_idx, batch_data in enumerate(train_loader):
        pos, neg, const = ncr(batch_data)