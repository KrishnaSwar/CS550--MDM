from neural_CR.dataSet import DataGenerator
import pandas as pd

if __name__ == '__main__':
    raw_dataset = pd.read_csv("datasets/toy-dataset/fake2.csv")
    dataset = DataGenerator(raw_dataset)

    dataset.manupulate_data(threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5, premise_threshold=0)

