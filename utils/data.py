import logging
import os
from typing import List
import torch
import torchtext
from torchtext.legacy.data import Field, TabularDataset, Example
from nltk.tokenize import WordPunctTokenizer
import datasets

logger = logging.getLogger('runner')
tokenizer = WordPunctTokenizer()
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
SEP_TOKEN = '<SEP>'
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, SEP_TOKEN]
VOCAB = List[str]


SAMPLES = [
  'What does it mean to be a god?',
  'The prices of oil suddenly risen',
  'I am not a doctor, silly you!'
]

def tokenization(x):
  return tokenizer.tokenize(x.lower())


def build_vocab(field, preprocessed_text, vectors=None):
  field.build_vocab(
      preprocessed_text,
      vectors=vectors
  )
  # get the vocab instance
  vocab = field.vocab
  return vocab

def build_vocab_en(EN_field, preprocessed_text):
  return build_vocab(EN_field, preprocessed_text, 'fasttext.simple.300d')

def load_dataset_local(EN_field, RU_field, path: str):

  dataset = TabularDataset(
    path=path,
    format='tsv',
    fields=[('en', EN_field), ('ru', RU_field)]
  )
  train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
  return train_data, valid_data, test_data

class TabularDataset_From_List(torchtext.legacy.data.Dataset):

  def __init__(self, input_list, format, fields, skip_header=False, **kwargs):
    examples = [Example.fromlist(item, fields) for item in input_list]

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


def load_dataset_opus(EN_field, RU_field):
  fname = 'saved_dataset'
  if not os.path.exists(fname):
    dataset = datasets.load_dataset("opus100", "en-ru")
    train_ds = dataset['train']
    val_ds = dataset['validation']
    test_ds = dataset['test']

    build_vocab_en([x['en'] for x in train_ds['translation']])
    build_vocab(RU_field, [x['ru'] for x in train_ds['translation']])

    dataset = dataset.map(lambda x: {
      'en': EN_field.process(x['en']),
      'ru': RU_field.process(x['ru'])
    }, input_columns='translation')
    dataset.set_format('torch', columns=['en', 'ru'])

    dataset.save_to_disk(fname)
  else:
    dataset = datasets.load_from_disk(fname)
    train_ds = dataset['train']
    build_vocab_en([x['en'] for x in train_ds['translation']])
    build_vocab(RU_field, [x['ru'] for x in train_ds['translation']])
    dataset.set_format('torch', columns=['en', 'ru'])


  tsv_path = 'opus100-train.txt'
  if not os.path.exists(tsv_path):
    examples = dataset['train']['translation']
    with open(tsv_path, 'w') as fw:
      for ex in examples[:-1]:
        a, b = ex['en'], ex['ru']
        if len(a) == 0 or len(b) == 0:
          continue
        fw.write(ex['en'])
        fw.write('\t')
        fw.write(ex['ru'])
        fw.write('\n')
      fw.write(examples[-1]['en'])
      fw.write('\t')
      fw.write(examples[-1]['ru'])
  dataset = TabularDataset(
    path=tsv_path,
    format='tsv',
    fields=[('en', EN_field), ('ru', RU_field)]
  )
  train_data, valid_data, test_data = dataset.split(split_ratio=[0.8, 0.15, 0.05])
  return train_data, valid_data, test_data

  #
  # train_ds = dataset['train']
  # val_ds = dataset['validation']
  # test_ds = dataset['test']

  return train_ds, val_ds, test_ds

  # train_ds = [torchtext.legacy.data.Example.fromlist(
  #   [y['en'], y['ru']],
  #   [('en', EN_field), ('ru', RU_field)])
  #  for y in train_ds]
  # val_ds = [torchtext.legacy.data.Example.fromlist(
  #   [y['en'], y['ru']],
  #   [('en', EN_field), ('ru', RU_field)])
  # for y in val_ds]
  # test_ds = [torchtext.legacy.data.Example.fromlist(
  #   [y['en'], y['ru']],
  #   [('en', EN_field), ('ru', RU_field)])
  # for y in test_ds]

  # train_ds, val_ds, test_ds = [[torchtext.legacy.data.Example.fromlist(
  #   [y['en'], y['ru']],
  #   [('en', EN_field), ('ru', RU_field)])
  # ] for x in [train_ds, val_ds, test_ds] for y in x]

  # train_ds, val_ds, test_ds = [
  #   torchtext.legacy.data.Dataset(
  #     ds,
  #     [('en', EN_field), ('ru', RU_field)]
  #   ) for ds in [train_ds, val_ds, test_ds]
  # ]

  # def conversion(examples):
  #   d = {'en': [], 'ru': []}
  #   for item in examples:
  #     d['en'].append(item['en'])
  #     d['ru'].append(item['ru'])
  #   return d
  #
  # train_ds, val_ds, test_ds = [
  #   TabularDataset_From_List(
  #     conversion(x),
  #     'dict',
  #     [('en', EN_field), ('ru', RU_field)]
  #   ) for x in [train_ds, val_ds, test_ds]
  # ]

  # return train_ds, val_ds, test_ds


def flatten(l):
  return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr):
  tokens_to_remove = set(SPECIAL_TOKENS)
  return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, token_to_word):
  text = [token_to_word(token) for token in x]
  try:
    end_idx = text.index(EOS_TOKEN)
    text = text[:end_idx]
  except ValueError:
    pass
  text = remove_tech_tokens(text)
  if len(text) < 1:
    text = []
  return text


def generate_translation(src, trg, model, token_to_word):
  with torch.no_grad():
    model.eval()

    output = model(src, trg, 0)  # turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:, 0].cpu().numpy()), token_to_word)
    generated = get_text(list(output[1:, 0]), token_to_word)

    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()

def translate(model, sentences: List[str], encode, token_to_word, max_len=128):
  with torch.no_grad():
    outs = []
    model.eval()
    for sentence in sentences:
      en_tokens = encode([sentence], model.device)
      ru_tokens = model.translate(en_tokens, max_len=max_len)
      ru_text = ' '.join(get_text(ru_tokens, token_to_word))
      outs.append(ru_text)
    return outs

# ru
# http://wikipedia2vec.s3.amazonaws.com/models/ru/2018-04-20/ruwiki_20180420_300d.txt.bz2
