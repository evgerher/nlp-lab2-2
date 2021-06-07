from torchtext.legacy.data import BucketIterator

from utils.data import *


def numericalize(self, arr, device=None):
  unk_token = self.vocab.stoi[UNK_TOKEN]
  if self.include_lengths and not isinstance(arr, tuple):
    raise ValueError("Field has include_lengths set to True, but "
                     "input data is not a tuple of "
                     "(data batch, batch lengths).")
  if isinstance(arr, tuple):
    arr, lengths = arr
    lengths = torch.tensor(lengths, dtype=self.dtype, device=device)

  if self.use_vocab:
    if self.sequential:
      arr = [[self.vocab.stoi.get(x, unk_token) for x in ex] for ex in arr]
    else:
      arr = [self.vocab.stoi.get(x, unk_token) for x in arr]

    if self.postprocessing is not None:
      arr = self.postprocessing(arr, self.vocab)
  else:
    if self.dtype not in self.dtypes:
      raise ValueError(
        "Specified Field dtype {} can not be used with "
        "use_vocab=False because we do not know how to numericalize it. "
        "Please raise an issue at "
        "https://github.com/pytorch/text/issues".format(self.dtype))
    numericalization_func = self.dtypes[self.dtype]
    # It doesn't make sense to explicitly coerce to a numeric type if
    # the data is sequential, since it's unclear how to coerce padding tokens
    # to a numeric type.
    if not self.sequential:
      arr = [numericalization_func(x) if isinstance(x, str)
             else x for x in arr]
    if self.postprocessing is not None:
      arr = self.postprocessing(arr, None)

  var = torch.tensor(arr, dtype=self.dtype, device=device)

  if self.sequential and not self.batch_first:
    var.t_()
  if self.sequential:
    var = var.contiguous()

  if self.include_lengths:
    return var, lengths
  return var

EN_field = Field(
    tokenize=tokenization,
    init_token = BOS_TOKEN,
    eos_token = EOS_TOKEN,
    pad_token=PAD_TOKEN,
    unk_token=UNK_TOKEN,
    # fix_length=5,
    lower=True
)

RU_field = Field(
  tokenize=tokenization,
  init_token = BOS_TOKEN,
  eos_token = EOS_TOKEN,
  pad_token=PAD_TOKEN,
  unk_token=UNK_TOKEN,

  # lower = True,
)

EN_field.numericalize = lambda *args, **kwargs: numericalize(EN_field, *args, **kwargs)
RU_field.numericalize = lambda *args, **kwargs: numericalize(RU_field, *args, **kwargs)

def _len_sort_key(x):
  return len(x.en)

def prepare_iterators(train_data, valid_data, test_data, BATCH_SIZE, device):
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=_len_sort_key
  )

  return train_iterator, valid_iterator, test_iterator

def labels_from_target(trg):
  trg = trg[1:].view(-1)
  return trg


def get_text(x, id_to_word):
  text = [id_to_word(token) for token in x]
  try:
    end_idx = text.index(EOS_TOKEN)
    text = text[:end_idx]
  except ValueError:
    pass
  text = remove_tech_tokens(text)
  if len(text) < 1:
    text = []
  return text
