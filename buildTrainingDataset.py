import numpy as np


def build_training_dataset(input_ds):
    input_ds.reset_index()
    data = input_ds.filter(items=['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * .95))
    return data, dataset, training_data_len
