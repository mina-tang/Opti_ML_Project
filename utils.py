import torch
from torch.utils.data import DataLoader
from collections import Counter


def get_LSTM_dataloaders(lstm_dataset, test_ratio=0.1):
    """
    split dataset into train and test sets with provided ratio
    :param lstm_dataset: dataset to split
    :param test_ratio: ratio of test set
    :return: 2 dataloader objects for both train and test sets
    """
    # split train/test dataset.
    lstm_train_dataset, lstm_test_dataset = torch.utils.data.random_split(lstm_dataset, [1 - test_ratio, test_ratio])
    # get pytorch DataLoader
    train_dataloader = DataLoader(lstm_train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(lstm_test_dataset, batch_size=8, shuffle=False)
    return train_dataloader, test_dataloader


def isEnglish(sample):
    """
    is false when sample isn't made of ascii characters (approximation of english)
    true otherwise
    :param sample: sample to evaluate
    :return: boolean
    """
    try:
        sample.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


def lowerCase(sample):
    """
    put sample to lowercase
    :param sample:
    :return: sample with text to lowercase
    """
    return {"text": sample["text"].lower()}


def count_tokens(dataset):
    """Counts the frequency of each token in the dataset.
    return a dict with token as keys, frequency as values."""

    token_freq_dict = Counter(" ".join((x['text'] for x in dataset)).split())
    return token_freq_dict


def replace_rare_tokens(sample, rare_tokens, unk_token):
    """
    put unk_token in place of rare tokens
    :param sample: sample to evaluate
    :param rare_tokens: list of rare tokens
    :param unk_token: unknown token
    :return: sample with replaced tokens
    """
    text = sample["text"]
    modified_tokens = [(token if token not in rare_tokens else unk_token)
                       for token in text.split()]
    return {"text": " ".join(modified_tokens)}


def is_unknown_sequence(sample, unk_token, unk_threshold=0.1):
    """
    returns true if sample contains a proportion of unk token greater than the threshold
    :param sample: sample to evaluate
    :param unk_token: unknown token
    :param unk_threshold: threshold for unknown token
    :return: boolean
    """
    sample_tokens = sample["text"].split()
    if sample_tokens.count(unk_token) / len(sample_tokens) > unk_threshold:
        return True
    else:
        return False


def build_vocabulary(dataset, min_freq=5, unk_token='<unk>'):
    """Builds a vocabulary dict for the given dataset."""
    # Get unique tokens and their frequencies.
    token_freq_dict = count_tokens(dataset)

    # Find a set of rare tokens with frequency lower than `min_freq` and replace them with `unk_token`.
    rare_tokens_set = set()
    low_freq = [x[0] for x in token_freq_dict.items() if x[1] <= min_freq]
    rare_tokens_set.update(low_freq)
    dataset = dataset.map(replace_rare_tokens, fn_kwargs={"rare_tokens": rare_tokens_set,
                                                          "unk_token": unk_token})

    # Filter out sequences with more than 15% rare tokens.
    dataset = dataset.filter(lambda x: not is_unknown_sequence(x, unk_token, unk_threshold=0.15))

    # Recompute the token frequency to get final vocabulary dict.
    token_freq_dict = count_tokens(dataset)
    return dataset, token_freq_dict
