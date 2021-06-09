from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu

from utils.rnn_utils import *
from utils.logger import setup_logger
from utils.train import prepare, train_epochs
from utils.cnn2rnn_utils import labels_from_target

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


class DecoderCNN(nn.Module):
  def __init__(self,
               model_setup,
               embedding,
               pad_idx,
               device,
               max_length=500):
    super().__init__()
    hid_dim = model_setup['hidden_size']
    emb_dim = model_setup['input_size']
    kernel_size = model_setup['kernel_size']
    n_layers = model_setup['n_layers']
    dropout = model_setup['dropout']

    self.kernel_size = kernel_size
    self.trg_pad_idx = pad_idx
    self.device = device

    self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

    self.tok_embedding = embedding
    self.pos_embedding = nn.Embedding(max_length, emb_dim)

    self.emb2hid = nn.Linear(emb_dim, hid_dim)
    self.hid2emb = nn.Linear(hid_dim, emb_dim)

    self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
    self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

    self.fc_out = nn.Linear(emb_dim, embedding.num_embeddings)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=hid_dim,
                                          out_channels=2 * hid_dim,
                                          kernel_size=kernel_size)
                                for _ in range(n_layers)])

    self.dropout = nn.Dropout(dropout)

  def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
    conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
    combined = (conved_emb + embedded) * self.scale
    energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
    attention = F.softmax(energy, dim=2)
    attended_encoding = torch.matmul(attention, encoder_combined)
    attended_encoding = self.attn_emb2hid(attended_encoding)
    attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
    return attention, attended_combined

  def forward(self, trg, encoder_conved, encoder_combined):
    batch_size = trg.shape[0]
    trg_len = trg.shape[1]
    pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    tok_embedded = self.tok_embedding(trg)
    pos_embedded = self.pos_embedding(pos)
    embedded = self.dropout(tok_embedded + pos_embedded)
    conv_input = self.emb2hid(embedded)
    conv_input = conv_input.permute(0, 2, 1)
    batch_size = conv_input.shape[0]
    hid_dim = conv_input.shape[1]
    for i, conv in enumerate(self.convs):
      conv_input = self.dropout(conv_input)
      padding = torch.zeros(batch_size,
                            hid_dim,
                            self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
      padded_conv_input = torch.cat((padding, conv_input), dim=2)
      conved = conv(padded_conv_input)
      conved = F.glu(conved, dim=1)
      attention, conved = self.calculate_attention(embedded,
                                                   conved,
                                                   encoder_conved,
                                                   encoder_combined)
      conved = (conved + conv_input) * self.scale
      conv_input = conved
    conved = self.hid2emb(conved.permute(0, 2, 1))
    output = self.fc_out(self.dropout(conved))
    return output, attention

class CNN2CNN(nn.Module):
  def __init__(self,
               encoder_setup: dict,
               decoder_setup: dict,
               encoder_embedding: nn.Embedding,
               decoder_embedding: nn.Embedding,
               en_vocab,
               ru_vocab,
               dec_pad_idx,
               device,
               model_name):
    super().__init__()
    self.encoder = CNN(encoder_setup, encoder_embedding, device)
    self.decoder = DecoderCNN(decoder_setup, decoder_embedding, dec_pad_idx, device)
    self.device = device
    self.name = model_name
    self.en_vocab = en_vocab
    self.ru_vocab = ru_vocab

  def forward(self, src, trg, *args):
    src = src.T
    trg = trg.T
    encoder_conved, encoder_combined = self.encoder(src)
    output, attention = self.decoder(trg, encoder_conved, encoder_combined)
    return output

  def translate(self, en_tokens, max_len=50):
    en_tokens = en_tokens.T
    batch_size = en_tokens.shape[0]
    encoder_conved, encoder_combined = self.encoder(en_tokens)

    trg_indexes = torch.tensor([[self.ru_vocab.stoi[BOS_TOKEN]] * batch_size], dtype=torch.long, device=device).T
    for i in range(max_len):
      with torch.no_grad():
        output, attention = self.decoder(trg_indexes, encoder_conved, encoder_combined)
      pred_token = output.argmax(2)[:, [-1]]
      trg_indexes = torch.hstack([trg_indexes, pred_token])
      # if pred_token == self.ru_vocab.stoi[EOS_TOKEN]:
      #   break
    return trg_indexes[:, 1:]

    # EOS_TOKEN_ID = self.ru_vocab.stoi[EOS_TOKEN]
    # trg_tokens = []
    # for row in trg_indexes:
    #   words = []
    #   for token in row:
    #     if token == EOS_TOKEN_ID:
    #       break
    #     word = self.ru_vocab.itos[token]
    #     words.append(word)
    #   trg_tokens.append(words)
    # return trg_tokens

    # ru_tokens = []
    # conved, combined = self.encoder(en_tokens)
    # input = torch.tensor([RU_field.vocab.stoi[BOS_TOKEN]], dtype=torch.long, device=self.device)
    # EOS_TOKEN_ID = RU_field.vocab.stoi[EOS_TOKEN]
    # decoder_hidden = None
    # for t in range(1, max_len):
    #   output, decoder_hidden, _ = self.decoder(input, decoder_hidden, combined_new)
    #   input = output.max(1)[1]
    #   token = input.item()
    #   ru_tokens.append(token)
    #   if token == EOS_TOKEN_ID:
    #     break
    # return ru_tokens


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
    'max_length': 128,
    'input_size': 300,
    'kernel_size': 3,
    'n_layers': 10,
    'hidden_size': 256,
    'dropout': 0.3
  }

  dec_emb_setup = {
    'embedding_size': 300,
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

  dataset = (train_data, valid_data, test_data)
  embeds = (encoder_embedding, decoder_embedding)
  vocabs = (en_vocab, ru_vocab)
  setups = (encoder_setup, decoder_setup)
  logger.info('Initialized params: loaded dataset, vocabs, embeds')
  return train_params, setups, vocabs, embeds, dataset


def build_seq2seq(setups, embeds, model_name, dec_pad_idx, en_vocab, ru_vocab):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  encoder_setup, decoder_setup = setups

  en_embed, ru_embed = embeds
  seq2seq = CNN2CNN(encoder_setup, decoder_setup, en_embed, ru_embed,
                    en_vocab, ru_vocab, dec_pad_idx,
                    device, model_name).to(device)
  logger.info('Initialized model')
  return seq2seq, device


def bleu_score(model, iterator_test, get_text):
  logger.info('Start BLEU scoring')
  original_text = []
  generated_text = []
  model.eval()
  BOS_TOKEN_ID = model.ru_vocab.stoi[EOS_TOKEN]
  with torch.no_grad():
    for i, batch in tqdm(enumerate(iterator_test)):
      src = batch.en
      trg = batch.ru

      output = model.translate(src, max(50, trg.shape[0]))
      trg = trg.cpu().numpy().T
      output = output.detach().cpu().numpy()
      original = [get_text(x) for x in trg]
      generated = [get_text(x) for x in output]
      original_text.extend(original)
      generated_text.extend(generated)
  score = corpus_bleu([[text] for text in original_text], generated_text) * 100
  logger.info('Finished BLEU scoring')
  logger.info('BLEU score: %.2f', score)

  return score

if __name__ == '__main__':
  setup_logger()
  model_name = 'CNN2CNN'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_CNN2CNN')
  encoder_setup, decoder_setup, dec_emb_setup, train_params = init_arguments()
  train_params, setups, vocabs, embeds, datasets = init_embeds(encoder_setup, decoder_setup, dec_emb_setup, train_params)
  (en_vocab, ru_vocab) = vocabs
  pad_idx = ru_vocab.stoi[PAD_TOKEN]
  seq2seq, device = build_seq2seq(setups, embeds, model_name, pad_idx, en_vocab, ru_vocab)

  optimizer, scheduler, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params,
                                                                                  seq2seq,
                                                                                  datasets,
                                                                                  device,
                                                                                  pad_idx,
                                                                                  prepare_iterators)
  convert_text = lambda x: get_text(x, lambda y: ru_vocab.itos[y])
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
