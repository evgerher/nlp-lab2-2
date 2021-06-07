import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchtext.data.utils import RandomShuffler
from torchtext.legacy.data import BucketIterator
from torchtext.legacy.data.dataset import check_split_ratio

from collections import namedtuple

from utils.data import *

enru = namedtuple('enru', ['en', 'ru'])

def create_field(tokenizer):
  return Field(
    use_vocab=False,
    tokenize=tokenizer.tokenize,
    init_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    unk_token=UNK_TOKEN
  )


class TranslationDataset(Dataset):
  def __init__(self, en, ru): # expected 1st to be en
    self.en = en
    self.ru = ru

  def split(self, split_ratio: list):
    assert (sum(split_ratio) - 1.0) < 1e-4
    N = len(self.en)
    splits = []
    position = 0
    for ratio in split_ratio:
      next_position = position + int(N * ratio)
      en = self.en[position:next_position]
      ru = self.ru[position:next_position]
      position = next_position
      ds = TranslationDataset(en, ru)
      splits.append(ds)

    return splits

  @classmethod
  def from_file(cls, path, sep='\t'):
    with open(path, 'r') as fo:
      contents = fo.read().split('\n')
      en, ru = [], []
      for line in contents:
        if len(line) > 0:
          left, right = line.split(sep)
          en.append(left)
          ru.append(right)
    return cls(en, ru)

  def __len__(self):
    return len(self.en)

  def __getitem__(self, idx):
    return (self.en[idx], self.ru[idx])


def build_collator(enc_tokenizer, dec_tokenizer, device):
  def collate_fn(examples):
    ens, rus = list(zip(*examples))
    ens, rus = list(ens), list(rus)

    en_input = enc_tokenizer(ens, return_tensors='pt', padding=True).to(device)
    ru_input = dec_tokenizer(rus).to(device)

    return enru(en_input, ru_input)
  return collate_fn


def prepare_iterators(train_data, valid_data, test_data, BATCH_SIZE, device, collate_fn, num_workers):
  train_iterator, valid_iterator, test_iterator = [
    DataLoader(dataset, BATCH_SIZE, collate_fn=collate_fn, num_workers=num_workers)
    for dataset in [train_data, valid_data, test_data]
  ]

  return train_iterator, valid_iterator, test_iterator

def labels_from_target(trg):
  labels = trg['input_ids']
  return labels[:, 1:].reshape(-1) # batch_size, seq_len

def get_text(x, tokenizer):
  return tokenizer.decode(x)
