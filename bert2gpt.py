import random
import json

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

from transformers import BertTokenizer, BertModel, GPT2Config, GPT2TokenizerFast, AutoTokenizer, GPT2LMHeadModel
from utils.bert2gpt_utils import *
from utils.logger import setup_logger
from utils.train import prepare, train_epochs, bleu_score

logger = logging.getLogger('runner')

class BERT2GPT(nn.Module):
  def __init__(self,
               encoder,
               encoder_tokenizer,
               decoder,
               decoder_tokenizer,
               device,
               model_name):
    super().__init__()
    self.encoder = encoder
    self.encoder_tokenizer = encoder_tokenizer
    self.decoder = decoder
    self.decoder_tokenizer = decoder_tokenizer
    self.device = device
    self.name = model_name

    self.encoder.eval() # freezy encoder
    for param in self.encoder.parameters():
      param.requires_grad = False

  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    batch_size = trg['input_ids'].shape[0]
    max_len = trg['input_ids'].shape[1]
    trg_vocab_size = len(self.decoder_tokenizer.get_vocab())

    # tensor to store decoder outputs
    outputs = torch.zeros((max_len - 1, batch_size, trg_vocab_size), device=self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_out = self.encoder(**src)  # encoder_hidden can be pair

    trg['encoder_hidden_states'] = encoder_out['last_hidden_state']
    trg['encoder_attention_mask'] = src['attention_mask']

    idx = 0
    new_trg = {
      'input_ids': trg['input_ids'][:, [idx]], # [batch, seq_len]
      'attention_mask': trg['attention_mask'][:, [idx]],
      'past_key_values': None,
      'use_cache': True,
      'encoder_attention_mask': src['attention_mask'],
      'encoder_hidden_states': encoder_out['last_hidden_state']
    }

    for t in range(1, max_len):
      idx += 1
      decoder_out = self.decoder(**new_trg)
      logits = decoder_out['logits'].squeeze(1)
      outputs[t - 1] = logits
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = logits.max(1)[1]

      if teacher_force:
        new_trg['input_ids'] = trg['input_ids'][:, [idx]]
      else:
        new_trg['input_ids'] = top1.unsqueeze(1)
      new_trg['attention_mask'] = trg['attention_mask'][:, [idx]]
      new_trg['past_key_values'] = decoder_out['past_key_values']

    return outputs # todo: softmax here?



  def translate(self, en_tokens, max_len: int):
    ru_tokens = []
    batch_size = en_tokens['input_ids'].shape[0]
    encoder_out = self.encoder(**en_tokens)

    CLS_TOKEN_ID = self.decoder_tokenizer.cls_token_id
    SEP_TOKEN_ID = self.decoder_tokenizer.sep_token_id

    input_ids = torch.tensor([[CLS_TOKEN_ID] * batch_size], dtype=torch.long, device=self.device)
    attn_mask = torch.ones([batch_size, 1], dtype=torch.long, device=self.device)
    trg = {
      'input_ids': input_ids,
      'attention_mask': attn_mask,
      'past_key_values': None,
      'use_cache': True,
      'encoder_attention_mask': en_tokens['attention_mask'],
      'encoder_hidden_states': encoder_out['last_hidden_state']
    }

    for t in range(1, max_len):
      decoder_out = self.decoder(**trg)
      logits = decoder_out['logits'].squeeze(1)
      input = logits.max(1)[1]
      trg['input_ids'] = input.unsqueeze(1)
      token = input.item()
      ru_tokens.append(token)
      if token == SEP_TOKEN_ID:
        break
    return ru_tokens


def create_tokenizer_ru(files: List[str], fout_path='tokenizer_gpt_ru'):
  if not os.path.exists(fout_path):
    bpe = BPE(unk_token=UNK_TOKEN)
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN, SEP_TOKEN])
    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
      single="[BOS] $A [EOS]",
      pair="[BOS] $A [SEP] $B:1 [BOS]:1",
      special_tokens=[
          ("[BOS]", tokenizer.token_to_id(BOS_TOKEN)),
          ("[EOS]", tokenizer.token_to_id(EOS_TOKEN)),
          ("[SEP]", tokenizer.token_to_id(SEP_TOKEN)),
      ],
    )
    tokenizer.enable_padding(pad_token=PAD_TOKEN, pad_id=tokenizer.token_to_id(PAD_TOKEN))
    tokenizer.save(fout_path)

    with open(fout_path, encoding='utf8') as f:
      content = json.load(f)
      vocab = content['model']['vocab']
      merges = content['model']['merges']

      with open(f'vocab-{fout_path}', 'w', encoding='utf8') as f:
        json.dump(vocab, f)

      with open(f'merges-{fout_path}', 'w', encoding='utf8') as f:
        json.dump(merges, f)

  tokenizer = GPT2TokenizerFast(f'vocab-{fout_path}',
                                f'merges-{fout_path}',
                                unk_token=UNK_TOKEN,
                                bos_token=BOS_TOKEN,
                                eos_token=EOS_TOKEN,
                                model_max_length=1024)
  return tokenizer



def init_arguments():
  encoder_setup = {
    'name': 'bert-base-uncased'
  }

  enc_tokenizer = BertTokenizer.from_pretrained(encoder_setup['name'])
  enc_model = BertModel.from_pretrained(encoder_setup['name'])



  # text = "Replace me by any text you'd like."
  # encoded_input = tokenizer(text, return_tensors='pt')
  # output = enc_model(**encoded_input)



  # dec_tokenizer = create_tokenizer_ru(['data.txt.ru'])
  dec_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
  dec_tokenizer.add_special_tokens({'bos_token': BOS_TOKEN, 'eos_token': EOS_TOKEN})
  decoder_setup = {
    'n_ctx': 768,
    'n_embd': 768, # self.head_dim * self.num_heads != self.embed_dim
    'n_layer': 8,
    'n_head': 8,
    'bos_token_id': dec_tokenizer.bos_token_id,
    'eos_token_id': dec_tokenizer.eos_token_id,
    'vocab_size': len(dec_tokenizer.get_vocab()),
    'add_cross_attention': True
  }

  decoder_config = GPT2Config(**decoder_setup)
  dec_model = GPT2LMHeadModel(decoder_config)

  train_params = {
    'lr': 0.001,
    'epochs': 15,
    'batch_size': 16
  }

  return (enc_tokenizer, enc_model), (dec_tokenizer, dec_model), train_params



def build_seq2seq(encoderr, decoderr, model_name):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  (enc_tokenizer, enc_model), (dec_tokenizer, dec_model) = encoderr, decoderr
  seq2seq = BERT2GPT(enc_model, enc_tokenizer, dec_model, dec_tokenizer, device, model_name).to(device)
  return seq2seq, device


if __name__ == '__main__':
  setup_logger()
  model_name = 'bert2gpt'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_bert2gpt')
  (enc_tokenizer, enc_model), (dec_tokenizer, dec_model), train_params = init_arguments()
  datasets = TranslationDataset.from_file('data.txt', '\t').split([0.8, 0.15, 0.05])
  seq2seq, device = build_seq2seq((enc_tokenizer, enc_model), (dec_tokenizer, dec_model), model_name)

  pad_idx = -100 # gpt2 does not use pad
  closured_collate = build_collator(enc_tokenizer, dec_tokenizer, device)
  optimizer, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params,
                                                                                  seq2seq,
                                                                                  datasets,
                                                                                  device,
                                                                                  pad_idx,
                                                                                  prepare_iterators,
                                                                                  num_workers=0,
                                                                                  collate_fn=closured_collate)

  train_epochs( # todo: тренировка gpt2 - на каждом шаге менять attention mask
    seq2seq,
    train_iterator,
    valid_iterator,
    optimizer,
    criterion,
    train_params['epochs'],
    writer,
    lambda x, device: enc_tokenizer(x, return_tensors='pt', padding=True).to(device),
    lambda token_id: dec_tokenizer.decode(token_id),
    labels_from_target
  )

  score = bleu_score(seq2seq, test_iterator, dec_tokenizer.token_to_id)
