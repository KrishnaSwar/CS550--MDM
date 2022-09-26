
import pandas as pd
import json

def prepare_movielens_100k(path_to_file):
    d_set = pd.read_csv(path_to_file, sep='\t')
    d_set.columns = ["userID", "itemID", "rating", "timestamp"]

    d_set['userID'] -= 1
    d_set['itemID'] -= 1

    return d_set

def prepare_amazon(path_to_file):
    hash_map = {"userID": [], "itemID": [], "rating": [], "timestamp": []}
    user_to_id = {}
    item_to_id = {}
    item_id, user_id = 0, 0
    with open(path_to_file, 'r') as f:
        line = f.readline()
        while line:
            json_line = json.loads(line)

            if json_line['reviewerID'] not in user_to_id:
                user_to_id[json_line['reviewerID']] = user_id
                user_id += 1
            if json_line['asin'] not in item_to_id:
                item_to_id[json_line['asin']] = item_id
                item_id += 1
            hash_map['userID'].append(user_to_id[json_line['reviewerID']])
            hash_map['itemID'].append(item_to_id[json_line['asin']])
            hash_map['rating'].append(int(json_line['overall']))
            hash_map['timestamp'].append(int(json_line['unixReviewTime']))
            line = f.readline()

    dataset = pd.DataFrame.from_dict(hash_map)
    return dataset