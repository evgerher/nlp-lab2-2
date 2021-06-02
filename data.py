import logging
import os
from typing import Optional, Tuple, List
import numpy as np
import torch
import torchtext
from torch import nn
from torchtext.vocab import FastText, GloVe
from torchtext.legacy.data import Field, TabularDataset, Example
from nltk.tokenize import WordPunctTokenizer
import datasets

logger = logging.getLogger('runner')
tokenizer = WordPunctTokenizer()
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN]
VOCAB = List[str]


SAMPLES = [
  'What does it mean to be a god?',
  'The prices of oil suddenly risen',
  'I am not a doctor, silly you!'
]

def tokenization(x):
  return tokenizer.tokenize(x.lower())

EN_field = Field(
    tokenize=tokenization,
    init_token = BOS_TOKEN,
    eos_token = EOS_TOKEN,
    pad_token=PAD_TOKEN,
    # fix_length=5,
    lower=True
)

RU_field = Field(
  tokenize=tokenization,
  init_token = BOS_TOKEN,
  eos_token = EOS_TOKEN,
  pad_token=PAD_TOKEN,
  # lower = True,
)

def build_vocab(field, preprocessed_text, vectors=None):
  field.build_vocab(
      preprocessed_text,
      vectors=vectors
  )
  # get the vocab instance
  vocab = field.vocab
  return vocab

def build_vocab_en(preprocessed_text):
  return build_vocab(EN_field, preprocessed_text, 'fasttext.simple.300d')

def load_dataset_local(path: str):

  dataset = TabularDataset(
    path=path,
    format='tsv',
    fields=[('en', EN_field), ('ru', RU_field)]
  )
  train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
  return train_data, valid_data, test_data

class TabularDataset_From_List(torchtext.legacy.data.Dataset):

  def __init__(self, input_list, format, fields, skip_header=False, **kwargs):
    make_example = {
      'json': Example.fromJSON, 'dict': Example.fromdict, 'csv': Example.fromCSV}[format.lower()]

    examples = [make_example(item, fields) for item in input_list]

    if make_example in (Example.fromdict, Example.fromJSON):
      fields, field_dict = [], fields
      for field in field_dict.values():
        if isinstance(field, list):
          fields.extend(field)
        else:
          fields.append(field)

    super(TabularDataset_From_List, self).__init__(examples, fields, **kwargs)

  @classmethod
  def splits(cls, path=None, root='.data', train=None, validation=None,
             test=None, **kwargs):
    if path is None:
      path = cls.download(root)
    train_data = None if train is None else cls(
      train, **kwargs)
    val_data = None if validation is None else cls(
      validation, **kwargs)
    test_data = None if test is None else cls(
      test, **kwargs)
    return tuple(d for d in (train_data, val_data, test_data) if d is not None)


def load_dataset_opus():
  dataset = datasets.load_dataset("opus100", "en-ru")
  train_ds = dataset['train']['translation']
  val_ds = dataset['validation']['translation']
  test_ds = dataset['test']['translation']

  train_ds = [torchtext.legacy.data.Example.fromlist(
    [y['en'], y['ru']],
    [('en', EN_field), ('ru', RU_field)])
   for y in train_ds]
  val_ds = [torchtext.legacy.data.Example.fromlist(
    [y['en'], y['ru']],
    [('en', EN_field), ('ru', RU_field)])
  for y in val_ds]
  test_ds = [torchtext.legacy.data.Example.fromlist(
    [y['en'], y['ru']],
    [('en', EN_field), ('ru', RU_field)])
  for y in test_ds]

  # train_ds, val_ds, test_ds = [[torchtext.legacy.data.Example.fromlist(
  #   [y['en'], y['ru']],
  #   [('en', EN_field), ('ru', RU_field)])
  # ] for x in [train_ds, val_ds, test_ds] for y in x]

  train_ds, val_ds, test_ds = [
    torchtext.legacy.data.Dataset(
      ds,
      [('en', EN_field), ('ru', RU_field)]
    ) for ds in [train_ds, val_ds, test_ds]
  ]

  return train_ds, val_ds, test_ds


def flatten(l):
  return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr):
  tokens_to_remove = set(SPECIAL_TOKENS)
  return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
  text = [TRG_vocab.itos[token] for token in x]
  try:
    end_idx = text.index(EOS_TOKEN)
    text = text[:end_idx]
  except ValueError:
    pass
  text = remove_tech_tokens(text)
  if len(text) < 1:
    text = []
  return text


def generate_translation(src, trg, model, TRG_vocab):
  with torch.no_grad():
    model.eval()

    output = model(src, trg, 0)  # turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:, 0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)

    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()

def translate(model, sentences: List[str], max_len=128):
  with torch.no_grad():
    outs = []
    model.eval()
    for sentence in sentences:
      en_tokens = EN_field.process([sentence], model.device)
      ru_tokens = model.translate(en_tokens, max_len=max_len)
      ru_text = ' '.join(get_text(ru_tokens, RU_field.vocab))
      outs.append(ru_text)
    return outs

# ru
# http://wikipedia2vec.s3.amazonaws.com/models/ru/2018-04-20/ruwiki_20180420_300d.txt.bz2
