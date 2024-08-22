import torch
from torch.utils.data import BatchSampler, SequentialSampler

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, df,index):
        self.data = df[df.columns.drop(["label"])].values
        self.label = df["label"].values
        self.index = index

    def __len__(self):
        return len(self.index)  

    def __getitem__(self, idx):

        idx_list = self.index[idx]
        data = self.data[idx_list]
        label = self.label[idx_list] 
        return (data, label)

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
