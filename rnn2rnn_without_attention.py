import random

from datasets import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from nltk.translate.bleu_score import corpus_bleu

from utils.attention import LuongAttention
from utils.rnn_utils import *
from utils.logger import setup_logger
from utils.train import prepare, train_epochs
import torch

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

class RNN2RNN(nn.Module):
  def __init__(self,
               encoder_cell: str,
               decoder_cell: str,
               encoder_setup: dict,
               decoder_setup: dict,
               encoder_embedding: nn.Embedding,
               decoder_embedding: nn.Embedding,
               attention,
               device,
               model_name):
    super().__init__()
    self.encoder = RNN_ModelEncoder(encoder_cell, encoder_setup, encoder_embedding)
    self.decoder = RNN_ModelDecoder(decoder_cell, decoder_setup, decoder_embedding, attention)
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
    trg_vocab_size = self.decoder.embedding.num_embeddings  

    # tensor to store decoder outputs
    outputs = torch.zeros((max_len - 1, batch_size, trg_vocab_size), device=self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_output_states, encoder_hidden = self.encoder(src, None)  # encoder_hidden can be pair

    # first input to the decoder is the <bos> tokens
    input = trg[0, :]
    decoder_hidden = encoder_hidden[-2:]
    for t in range(1, max_len):
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, encoder_output_states)
      outputs[t - 1] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.max(1)[1]
      input = (trg[t] if teacher_force else top1)

    return outputs # todo: softmax here?

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


def init_arguments():
  encoder_setup = {
    'hidden_size': 512,
    'input_size': 256,
    'bidirectional': True,
    'dropout': 0.25,
    'layers': 2
  }

  decoder_setup = {
    'hidden_size': 512,
    'input_size': 256,
    'bidirectional': False,
    'dropout': 0.25,
    'other_dropout': 0.25,
    'layers': 2
  }

  dec_emb_setup = {
    'embedding_size': 256,
    'max_length': 128
  }

  train_params = {
    'lr': 1e-3,
    'epochs': 20,
    'batch_size': 128
  }

  return encoder_setup, decoder_setup, dec_emb_setup, train_params


def init_embeds(encoder_setup, decoder_setup, dec_emb_setup, train_params):
  train_data, valid_data, test_data = load_dataset_local(EN_field, RU_field, 'data.txt')
  en_vocab = build_vocab_en(EN_field, train_data)
  ru_vocab = build_vocab(RU_field, train_data)
  # train_data, valid_data, test_data = load_dataset_opus(EN_field, RU_field)
  # en_vocab = EN_field.vocab
  # ru_vocab = RU_field.vocab



  # weights = EN_field.vocab.vectors
  # mask = (weights[:, 0] == 0.0)
  # mean, std = weights[~mask].mean(), weights[~mask].std()
  # weights[mask] = torch.normal(mean, std, weights[mask].size())

  n_tokens = len(ru_vocab.stoi)
  # encoder_embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=en_vocab.stoi[PAD_TOKEN])
  encoder_embedding = nn.Embedding(len(en_vocab.stoi), encoder_setup['input_size'], padding_idx=en_vocab.stoi[PAD_TOKEN])
  decoder_embedding = nn.Embedding(n_tokens, dec_emb_setup['embedding_size'], padding_idx=ru_vocab.stoi[PAD_TOKEN])

  attention = None
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
  seq2seq = RNN2RNN('GRU', 'GRU', encoder_setup, decoder_setup, en_embed, ru_embed, attention, device, model_name).to(device)
  logger.info('Initialized model')
  return seq2seq, device


def bleu_score(model, iterator_test, get_text):
  logger.info('Start BLEU scoring')
  original_text = []
  generated_text = []
  model.eval()
  with torch.no_grad():
    for i, batch in tqdm(enumerate(iterator_test)):
      src = batch.en
      trg = batch.ru

      output = model(src, trg, 0)  # turn off teacher forcing

      # trg = [trg sent len, batch size]
      # output = [trg sent len, batch size, output dim]

      output = output.argmax(dim=-1)
      original = [get_text(x) for x in trg.cpu().numpy().T]
      generated = [get_text(x) for x in output.detach().cpu().numpy().T]
      original_text.extend(original)
      generated_text.extend(generated)
  score = corpus_bleu([[text] for text in original_text], generated_text) * 100
  logger.info('Finished BLEU scoring')
  logger.info('BLEU score: %.2f', score)

  return score


if __name__ == '__main__':
  setup_logger()
  model_name = 'RNN2RNN'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_RNN2RNN')
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
