from utils.data import *

def create_field(tokenizer):
  return Field(
    use_vocab=False,
    tokenize=tokenizer.encode,
    init_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    unk_token=UNK_TOKEN
  )
