import random

from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F

from utils.attention import LuongAttention
from utils.rnn_utils import *
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


class CNN(nn.Module):
  def __init__(self, model_setup, embedding, device):
    super(CNN, self).__init__()

    hid_dim = model_setup['hidden_size']
    emb_dim = model_setup['input_size']
    kernel_size = model_setup['kernel_size']
    n_layers = model_setup['n_layers']
    dropout = model_setup['dropout']

    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
    self.device = device
    self.tok_embedding = embedding
    self.pos_embedding = nn.Embedding(model_setup['max_length'], emb_dim)
    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                          out_channels=2 * hid_dim,
                                          kernel_size=kernel_size,
                                          padding=(kernel_size - 1) // 2)
                                for _ in range(n_layers)])

    self.dropout = nn.Dropout(dropout)

  def forward(self, src):
    batch_size = src.shape[0]
    src_len = src.shape[1]
    pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    tok_embedded = self.tok_embedding(src)
    pos_embedded = self.pos_embedding(pos)
    embedded = self.dropout(tok_embedded + pos_embedded)
    conv_input = self.emb2hid(embedded)
    conv_input = conv_input.permute(0, 2, 1)
    for i, conv in enumerate(self.convs):
      conved = conv(self.dropout(conv_input))
      conved = F.glu(conved, dim=1)
      conved = (conved + conv_input) * self.scale
      conv_input = conved
    conved = self.hid2emb(conved.permute(0, 2, 1))
    combined = (conved + embedded) * self.scale
    return conved, combined


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

class CNN2RNN(nn.Module):
  def __init__(self,
               decoder_cell: str,
               encoder_setup: dict,
               decoder_setup: dict,
               encoder_embedding: nn.Embedding,
               decoder_embedding: nn.Embedding,
               attention,
               device,
               model_name):
    super().__init__()
    self.encoder = CNN(encoder_setup, encoder_embedding, device)
    self.decoder = RNN_ModelDecoder(decoder_cell, decoder_setup, decoder_embedding, attention)
    self.adapter = nn.Linear(encoder_setup['input_size'], decoder_setup['hidden_size'])
    self.device = device
    self.name = model_name

  def forward(self, src, trg, teacher_forcing_ratio = 0.5):
    batch_size = trg.shape[1]
    max_len = trg.shape[0]  # todo: look precisely here
    trg_vocab_size = self.decoder.embedding.num_embeddings  # todo: what ?

    # tensor to store decoder outputs
    outputs = torch.zeros((max_len - 1, batch_size, trg_vocab_size), device=self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    conved, combined = self.encoder(src)  # encoder_hidden can be pair
    combined_new = self.adapter(combined)
    # first input to the decoder is the <bos> tokens
    input = trg[0, :]
    decoder_hidden = None
    for t in range(1, max_len):
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, combined_new)
      outputs[t - 1] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.max(1)[1]
      input = (trg[t] if teacher_force else top1)

    return outputs # todo: softmax here?

  def translate(self, en_tokens, max_len: int):
    ru_tokens = []
    conved, combined = self.encoder(en_tokens.T)
    combined_new = self.adapter(combined)
    input = torch.tensor([RU_field.vocab.stoi[BOS_TOKEN]], dtype=torch.long, device=self.device)
    EOS_TOKEN_ID = RU_field.vocab.stoi[EOS_TOKEN]
    decoder_hidden = None
    for t in range(1, max_len):
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, combined_new)
      input = output.max(1)[1]
      token = input.item()
      ru_tokens.append(token)
      if token == EOS_TOKEN_ID:
        break
    return ru_tokens


def init_arguments():
  encoder_setup = {
    'max_length': 128,
    'input_size': 300,
    'kernel_size': 3,
    'n_layers': 10,
    'hidden_size': 256,
    'dropout': 0.3
  }

  decoder_setup = {
    'hidden_size': 256,
    'input_size': 200,
    'bidirectional': False,
    'dropout': 0,
    'other_dropout': 0.3,
    'layers': 1
  }

  dec_emb_setup = {
    'embedding_size': 200,
    'max_length': 128
  }

  train_params = {
    'lr': 0.001,
    'epochs': 20,
    'batch_size': 128
  }

  return encoder_setup, decoder_setup, dec_emb_setup, train_params


def init_embeds(encoder_setup, decoder_setup, dec_emb_setup, train_params):
  train_data, valid_data, test_data = load_dataset_local(EN_field, RU_field, 'data.txt')
  en_vocab = build_vocab_en(EN_field, train_data)
  ru_vocab = build_vocab(RU_field, train_data)


  weights = EN_field.vocab.vectors
  mask = (weights[:, 0] == 0.0)
  mean, std = weights[~mask].mean(), weights[~mask].std()
  weights[mask] = torch.normal(mean, std, weights[mask].size())

  n_tokens = len(ru_vocab.stoi)
  encoder_embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=en_vocab.stoi[PAD_TOKEN])
  decoder_embedding = nn.Embedding(n_tokens, dec_emb_setup['embedding_size'], padding_idx=ru_vocab.stoi[PAD_TOKEN])

  attention = LuongAttention(decoder_setup['hidden_size'], False, n_tokens)
  dataset = (train_data, valid_data, test_data)
  embeds = (encoder_embedding, decoder_embedding)
  vocabs = (en_vocab, ru_vocab)
  setups = (encoder_setup, decoder_setup)
  logger.info('Initialized params: loaded dataset, vocabs, embeds')
  return train_params, setups, vocabs, embeds, attention, dataset


def build_seq2seq(setups, embeds, attention, model_name):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  encoder_setup, decoder_setup = setups

  en_embed, ru_embed = embeds
  seq2seq = CNN2RNN('GRU', encoder_setup, decoder_setup, en_embed, ru_embed, attention, device, model_name).to(device)
  logger.info('Initialized model')
  return seq2seq, device

if __name__ == '__main__':
  setup_logger()
  model_name = 'CNN2RNN'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_CNN2RNN')
  encoder_setup, decoder_setup, dec_emb_setup, train_params = init_arguments()
  train_params, setups, vocabs, embeds, attention, datasets = init_embeds(encoder_setup, decoder_setup, dec_emb_setup, train_params)
  (en_vocab, ru_vocab) = vocabs
  seq2seq, device = build_seq2seq(setups, embeds, attention, model_name)

  pad_idx = ru_vocab.stoi[PAD_TOKEN]
  optimizer, scheduler, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params,
                                                                                  seq2seq,
                                                                                  datasets,
                                                                                  device,
                                                                                  pad_idx,
                                                                                  prepare_iterators)
  convert_text = lambda x: get_text(x, lambda token: RU_field.vocab.itos[token])
  train_epochs(
    seq2seq,
    train_iterator,
    valid_iterator,
    optimizer,
    scheduler,
    criterion,
    train_params['epochs'],
    writer,
    lambda x, device: EN_field.process(x, device),
    convert_text,
    labels_from_target
  )

  score = bleu_score(seq2seq, test_iterator, convert_text)
