
import sys
import os
import pandas as pd

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset


dataset_dir = 'C://Users/bboom/Capstone_Design/Sequential_Model/dataset/'
num_features = 13




class chairrise_dataset(Dataset):

    def __init__(self, dataset_dir, mode):
        self.dataset_dir = dataset_dir
        self.mode = mode
        df = pd.read_csv(os.path.join(self.dataset_dir, f'chairrise_{self.mode}.csv')).drop(columns = 'Unnamed: 0')
        
        self.annos_info = []

        if mode == 'train' or mode == 'val':
            for _, input_seq in df.iterrows():
                input_sequence = input_seq[:-1].tolist()

                class_id = input_seq.label
                self.annos_info.append(
                    [torch.tensor([input_sequence[x:x + num_features] for x in range(0, len(input_sequence), num_features)]),
                    torch.tensor(class_id)])  

        if mode == 'test':
            for _, input_seq in df.iterrows():
                input_sequence = input_seq[1:-1].tolist()

                class_id = input_seq.label
                # sbj_id = input_seq.Subject

                self.annos_info.append(
                    [torch.tensor([input_sequence[x:x + num_features] for x in range(0, len(input_sequence), num_features)]), 
                    torch.tensor(class_id)])  

    def __len__(self):
        return len(self.annos_info)

    def __getitem__(self, idx):
        # if self.mode == 'test':
        #     return self.subs[idx], self.x[idx], self.y[idx]
        
        # return self.x[idx], self.y[idx]

        return self.annos_info[idx]



if __name__ == '__main__':

    df = pd.read_csv(os.path.join(dataset_dir, 'chairrise_train.csv')).drop(columns = 'Unnamed: 0')

    print(df.columns.tolist())