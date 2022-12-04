import torch
from torch.utils.data import DataLoader
import sys
import os

sys.path.append('./Sequential_Model')

from preprocess.dataset import chairrise_dataset, dataset_dir
# from configs.config import configs


def create_train_val_loader():
    train_set = chairrise_dataset(dataset_dir, 'train')

    # train_dl = DataLoader(train_set, batch_size = 32, shuffle = True, pin_memory = True, num_workers = 4, drop_last = True)
    train_dl = DataLoader(train_set, batch_size = 8, shuffle = True, pin_memory = True, num_workers = 4)

    validation_set = chairrise_dataset(dataset_dir, 'val')
    validation_dl = DataLoader(validation_set, 8 * 2, shuffle = True, pin_memory = True, num_workers = 4)
    return train_dl, validation_dl

def create_test_loader():

    test_set = chairrise_dataset(dataset_dir, 'test')
    test_dl = DataLoader(test_set, batch_size = 8 * 2, pin_memory = True, num_workers = 4)

    return test_dl

if __name__ == '__main__':
    import numpy as np
    train_dl = create_train_val_loader()[0]
    for data, label in train_dl:
        print(np.array(data).shape)
        print(label)
        break
