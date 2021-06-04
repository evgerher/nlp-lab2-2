import random

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

from transformers import BertTokenizer, BertModel, GPT2Config, GPT2Model
from utils.bert2gpt_utils import *
from utils.logger import setup_logger
from utils.train import prepare, train_epochs, bleu_score

logger = logging.getLogger('runner')

def resolve_rnn(input_size: int, cell_name: str, model_setup: dict) -> nn.Module:
  if cell_name == 'GRU':
    constructor = nn.GRU
  elif cell_name == 'LSTM':
    constructor = nn.LSTM
  elif cell_name == 'RNN':
    constructor = nn.RNN
  else:
    raise NotImplementedError()

  return constructor(
    input_size=input_size,
    hidden_size=model_setup['hidden_size'],
    dropout=model_setup['dropout'],
    num_layers=model_setup['layers'],
    bidirectional=model_setup['bidirectional']
  )


class RNN_ModelEncoder(nn.Module):
  def __init__(self, cell_name: str, model_setup, embedding):
    super(RNN_ModelEncoder, self).__init__()
    self.input_dim = embedding.embedding_dim
    self.hid_dim = model_setup['hidden_size']
    self.nlayers = model_setup['layers']
    self.bidirectional = model_setup['bidirectional']
    self.rnn = resolve_rnn(embedding.embedding_dim, cell_name, model_setup)
    self.embedding = embedding
    self.cell_type = cell_name

  def forward(self, input, hidden):
    embeds = self.embedding(input)
    if hidden is None and self.cell_type == 'LSTM':
      hidden = (None, None)
    output, new_hidden = self.rnn(embeds, hidden)
    return output, new_hidden


class RNN_ModelDecoder(nn.Module):
  def __init__(self, cell_name: str, model_setup: dict, embedding: nn.Embedding, attention):
    super().__init__()
    self.out_classes = embedding.num_embeddings
    self.attention = attention
    self.input_dim = embedding.embedding_dim
    self.hid_dim = model_setup['hidden_size']
    self.nlayers = model_setup['layers']
    self.rnn = resolve_rnn(embedding.embedding_dim, cell_name, model_setup)
    self.embedding = embedding
    self.dropout = nn.Dropout(model_setup['other_dropout'])
    self.out = nn.Linear(model_setup['hidden_size'], self.out_classes)

  def forward(self, input, hidden, encoder_outputs):
    embeds = self.embedding(input).unsqueeze(0)
    embeds = self.dropout(embeds)
    output, hidden = self.rnn(embeds, hidden)
    attn_weights = None
    if self.attention:
      output, attn_weights = self.attention(output, hidden, encoder_outputs)
    output = self.out(output)
    return output, hidden, attn_weights

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

  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    # src = src.T
    # trg = trg.T
    # src = [src sent len, batch size]
    # trg = [trg sent len, batch size]
    # teacher_forcing_ratio is probability to use teacher forcing
    # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

    # Again, now batch is the first dimention instead of zero
    batch_size = trg.shape[1]
    max_len = trg.shape[0]  # todo: look precisely here
    trg_vocab_size = self.decoder.embedding.num_embeddings  # todo: what ?

    # tensor to store decoder outputs
    outputs = torch.zeros((max_len - 1, batch_size, trg_vocab_size), device=self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_output_states, encoder_hidden = self.encoder(src, None)  # encoder_hidden can be pair

    # first input to the decoder is the <bos> tokens
    input = trg[0, :]
    decoder_hidden = encoder_hidden[-2:]
    for t in range(1, max_len):
      # todo: i am not expecting softmax or log_softmax
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, encoder_output_states)
      outputs[t - 1] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.max(1)[1]
      input = (trg[t] if teacher_force else top1)

    return outputs

  def translate(self, en_tokens, max_len: int):
    ru_tokens = []
    encoder_output_states, encoder_hidden = self.encoder(en_tokens, None)
    input = torch.tensor([RU_field.vocab.stoi[BOS_TOKEN]], dtype=torch.long, device=self.device)
    EOS_TOKEN_ID = RU_field.vocab.stoi[EOS_TOKEN]
    decoder_hidden = encoder_hidden[-2:]
    for t in range(1, max_len):
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, encoder_output_states)
      input = output.max(1)[1]
      token = input.item()
      ru_tokens.append(token)
      if token == EOS_TOKEN_ID:
        break
    return ru_tokens


def create_tokenizer_ru(files: List[str], fout_path='tokenizer_gpt_ru'):
  if not os.path.exists(fout_path):
    bpe = BPE(unk_token=UNK_TOKEN)
    tokenizer = Tokenizer(bpe)
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(special_tokens=[BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, PAD_TOKEN])
    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
      single="[BOS] $A [EOS]",
      pair="[BOS] $A [SEP] $B:1 [BOS]:1",
      special_tokens=[
          ("[BOS]", tokenizer.token_to_id(BOS_TOKEN)),
          ("[EOS]", tokenizer.token_to_id(EOS_TOKEN)),
      ],
    )
    tokenizer.enable_padding(pad_token=PAD_TOKEN, pad_id=tokenizer.token_to_id(PAD_TOKEN))
    tokenizer.save(fout_path)
  else:
    tokenizer = Tokenizer.from_file(fout_path)

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



  dec_tokenizer = create_tokenizer_ru(['data.txt.ru'])
  decoder_setup = {
    'n_ctx': 768,
    'n_embd': 512,
    'n_layer': 12,
    'n_head': 12,
    'bos_token_id': dec_tokenizer.token_to_id(BOS_TOKEN),
    'eos_token_id': dec_tokenizer.token_to_id(EOS_TOKEN),
    'vocab_size': None, # todo: vocab_size from dec_tokenizer
  }

  decoder_config = GPT2Config(**decoder_setup)
  dec_model = GPT2Model(**decoder_config)

  RU_field = create_field(dec_tokenizer)
  EN_field = create_field(enc_tokenizer)


  train_params = {
    'lr': 0.001,
    'epochs': 15,
    'batch_size': 128
  }

  return (enc_tokenizer, enc_model), (dec_tokenizer, dec_model), (EN_field, RU_field), train_params



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
  encoderr, decoderr, (EN_field, RU_field), train_params = init_arguments()
  dataset = load_dataset_local(EN_field, RU_field, 'data.txt')
  seq2seq, device = build_seq2seq(encoderr, decoderr, model_name)

  pad_idx = seq2seq.decoder_tokenizer.token_to_id(PAD_TOKEN)
  optimizer, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params, seq2seq, dataset, device, pad_idx)
  train_epochs(seq2seq, train_iterator, valid_iterator, optimizer, criterion, train_params['epochs'], writer, EN_field, RU_field)

  score = bleu_score(seq2seq, test_iterator)
