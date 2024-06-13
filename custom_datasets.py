from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
import datasets


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


class LSTMDataset(Dataset):
    def __init__(self,
                 dataset: datasets.arrow_dataset.Dataset,
                 max_seq_length: int, ):
        self.train_data = self.prepare_dataset(dataset)
        self.max_seq_length = max_seq_length + 2  # as <start> and <stop> will be added
        self.dataset_vocab = self.get_vocabulary(dataset)
        self.token2idx = {element: index for index, element in enumerate(self.dataset_vocab)}
        self.idx2token = dict(enumerate(self.dataset_vocab))
        self.pad_idx = self.token2idx["<pad>"]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # Get a list of tokens of the given sequence. Represent each token with its index in `self.token2idx`.
        token_list = self.train_data[idx].split()
        # having a fallback to <unk> token if an unseen word is encoded.
        token_ids = [self.token2idx.get(t, self.token2idx['<unk>']) for t in token_list]

        # Add padding token to the sequence to reach the max_seq_length.
        token_ids += [self.token2idx['<pad>']] * (self.max_seq_length - len(token_ids))

        return torch.tensor(token_ids)

    def get_vocabulary(self, dataset: datasets.arrow_dataset.Dataset):
        vocab = set()
        print("Getting dataset's vocabulary")
        for sample in tqdm(dataset):
            vocab.update(set(sample["text"].split()))
        vocab.update(set(["<start>", "<stop>", "<pad>"]))
        vocab = sorted(vocab)
        return vocab

    @staticmethod
    def prepare_dataset(target_dataset: datasets.arrow_dataset.Dataset):
        """
        Encapsulate sequences between <start> and <stop>.

        :param: target_dataset: the target dataset to extract samples
        return: a list of encapsulated samples.
        """
        prepared_dataset = []
        for sample in target_dataset:
            prepared_dataset.append(f"<start> {sample['text']} <stop>")
        return prepared_dataset
