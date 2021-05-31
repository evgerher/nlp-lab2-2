import random

import torch
from torch import nn
import torch.nn.functional as F

from data import *
from logger import setup_logger
from train import prepare, train, train_epochs, bleu_score


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
    self.bidirectional = model_setup['bidirectional']
    self.rnn = resolve_rnn(embedding.embedding_dim, cell_name, model_setup)
    self.embedding = embedding
    self.dropout = nn.Dropout(model_setup['other_dropout'])
    self.out = nn.Linear(model_setup['hidden_size'], self.out_classes)

  def forward(self, input, hidden, encoder_outputs):
    embeds = self.embedding(input).unsqueeze(0)
    embeds = self.dropout(embeds)
    if self.attention:
      rnn_input = self.attention(embeds, hidden, encoder_outputs)
    else:
      rnn_input = embeds
    rnn_input = F.relu(rnn_input)
    output, hidden = self.rnn(rnn_input, hidden)
    output = self.out(output)[0]
    return output, hidden

class RNN2RNN(nn.Module):
  def __init__(self,
               encoder_cell: str,
               decoder_cell: str,
               encoder_setup: dict,
               decoder_setup: dict,
               encoder_embedding: nn.Embedding,
               decoder_embedding: nn.Embedding,
               attention,
               device):
    super().__init__()
    self.encoder = RNN_ModelEncoder(encoder_cell, encoder_setup, encoder_embedding)
    self.decoder = RNN_ModelDecoder(decoder_cell, decoder_setup, decoder_embedding, attention)
    self.device = device

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
    decoder_hidden = encoder_hidden[:2]
    for t in range(1, max_len):
      # todo: i am not expecting softmax or log_softmax
      output, decoder_hidden = self.decoder(input, decoder_hidden, encoder_output_states)
      outputs[t - 1] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.max(1)[1]
      input = (trg[t] if teacher_force else top1)

    return outputs


def init_arguments():
  encoder_setup = {
    'hidden_size': 256,
    'input_size': 300,
    'bidirectional': True,
    'dropout': 0.4,
    'layers': 2
  }

  decoder_setup = {
    'hidden_size': 256,
    'input_size': 200,
    'bidirectional': False,
    'dropout': 0.4,
    'other_dropout': 0.3,
    'layers': 2
  }

  dec_emb_setup = {
    'embedding_size': 200,
    'max_length': 128
  }

  train_params = {
    'lr': 0.001,
    'epochs': 30,
    'batch_size': 128
  }

  train_data, valid_data, test_data = load_dataset('data.txt')
  en_vocab = build_vocab_en(train_data)
  ru_vocab = build_vocab(RU_field, train_data)

  n_tokens = len(ru_vocab.stoi)
  encoder_embedding = nn.Embedding.from_pretrained(EN_field.vocab.vectors, padding_idx=en_vocab.stoi[PAD_TOKEN])
  decoder_embedding = nn.Embedding(n_tokens, dec_emb_setup['embedding_size'], padding_idx=ru_vocab.stoi[PAD_TOKEN])

  attention = None
  dataset = (train_data, valid_data, test_data)
  embeds = (encoder_embedding, decoder_embedding)
  vocabs = (en_vocab, ru_vocab)
  setups = (encoder_setup, decoder_setup)
  return train_params, setups, vocabs, embeds, attention, dataset


def build_seq2seq(setups, embeds, attention):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  encoder_setup, decoder_setup = setups

  en_embed, ru_embed = embeds
  seq2seq = RNN2RNN('GRU', 'GRU', encoder_setup, decoder_setup, en_embed, ru_embed, attention, device).to(device)
  return seq2seq, device

if __name__ == '__main__':
  setup_logger()
  train_params, setups, vocabs, embeds, attention, dataset = init_arguments()
  (en_vocab, ru_vocab) = vocabs
  seq2seq, device = build_seq2seq(setups, embeds, attention)

  pad_idx = ru_vocab.stoi[PAD_TOKEN]
  optimizer, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params, seq2seq, dataset, device, pad_idx)
  train_epochs(seq2seq, train_iterator, valid_iterator, optimizer, criterion, epochs=train_params['epochs'])

  score = bleu_score(seq2seq, test_iterator)
  print('Model bleu', score)
