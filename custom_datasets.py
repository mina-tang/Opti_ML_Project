from torch.utils.data import Dataset
import torch
import numpy as np

class WineDataset(Dataset):
    def __init__(self, data, data_classes):
        self.data = data
        self.data_classes = data_classes.to_numpy()

    def __getitem__(self, index):
        x = torch.from_numpy(self.data.iloc[index][:-1].to_numpy())
        labels = torch.from_numpy(self.data.iloc[index][-1:].to_numpy())
        i = labels.type(torch.int32)
        one_hot_base = np.eye(len(set(self.data_classes)))
        return x, one_hot_base[i - 3]

    def __len__(self):
        return len(self.data)
