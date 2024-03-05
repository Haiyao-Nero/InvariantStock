import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import random
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, df,index, num_stock, sequence_length):
        self.data = df[df.columns.drop(["LABEL0"])].astype(float).values
        self.label = df["LABEL0"].astype(float).values
        # self.df.set_index(["datetime","instrument"],inplace=True)
        self.num_stock = num_stock
        self.sequence_length = sequence_length
        # self.date = list(self.df.index.get_level_values("datetime").unique())
        self.index = index.astype(int)

    def __len__(self):
        return len(self.index)  # subtract 30 to account for accumulation of 30 days of data

    def __getitem__(self, idx):

        input_data = []
        label_data = []
        # while True:
        #     index = self.index[idx]
        #     date, stock = self.df.index[index]
        #     if date>self.date[-self.sequence_length]:
        #         continue
        #     date_list = range(self.date.index(date),self.date.index(date)+20)
        #     idx_list = [(self.date[i],stock) for i in date_list]
        #     if not all(i in self.df.index for i in idx_list):
        #         continue
            # idx_list = [self.num_stock*i + idx for i in range(self.sequence_length) if self.num_stock*i + idx < len(self.df)]
        idx_list = self.index[idx]
        data = self.data[idx_list] #(seq_len, character)
        label = self.label[idx_list] #(seq_len, 1)
        
        input_data.append(data)
        input_data = np.concatenate(input_data, axis=0)
        label_data.append(label)
        # label 도출 값은 반드시 잘라서 써야함.
        return (input_data, label)

class DynamicBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_sizes):
        self.dataset = dataset
        self.total_batches = batch_sizes.shape[0]
        self.batch_sizes = batch_sizes
        self.iter_batch_sizes = iter(self.batch_sizes)
        

    def __iter__(self):
        batch_size = next(self.iter_batch_sizes, None)
        sampler = SequentialSampler(self.dataset)
        batch = []

        for idx in sampler:
            batch.append(idx)
            if len(batch) == batch_size:
                yield batch
                batch = []
                batch_size = next(self.iter_batch_sizes, None)
                if batch_size is None:
                    self.iter_batch_sizes = iter(self.batch_sizes)
                    break

        if batch:
            # Yield the last batch
            yield batch

    def __len__(self):
        return self.total_batches
