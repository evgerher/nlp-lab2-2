import random
from torch import nn
from torch.utils.tensorboard import SummaryWriter


from transformers import BertTokenizer, BertModel

from utils.attention import LuongAttention
from utils.bert2gpt_utils import *
from utils.logger import setup_logger
from utils.train import prepare, train_epochs, bleu_score
from utils.rnn_utils import numericalize
from rnn_model_attention import resolve_rnn

logger = logging.getLogger('runner')


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

class BERT2RNN(nn.Module):
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
    batch_size = trg.shape[1]
    max_len = trg.shape[0]
    trg_vocab_size = self.decoder.out_classes

    # tensor to store decoder outputs
    outputs = torch.zeros((max_len - 1, batch_size, trg_vocab_size), device=self.device)

    # last hidden state of the encoder is used as the initial hidden state of the decoder
    encoder_out = self.encoder(**src)  # encoder_hidden can be pair
    encoder_out_states = encoder_out['last_hidden_state']

    idx = 0
    input = trg[0, :]
    decoder_hidden = None
    for t in range(1, max_len):
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, encoder_out_states)
      outputs[t - 1] = output
      teacher_force = random.random() < teacher_forcing_ratio
      top1 = output.max(1)[1]
      input = (trg[t] if teacher_force else top1)

    return outputs



  def translate(self, en_tokens, max_len: int):
    ru_tokens = []
    batch_size = en_tokens['input_ids'].shape[0]
    encoder_out = self.encoder(**en_tokens)
    encoder_output_states = encoder_out['last_hidden_state']


    input = torch.tensor([self.decoder_tokenizer.vocab.stoi[BOS_TOKEN]], dtype=torch.long, device=self.device)
    EOS_TOKEN_ID = self.decoder_tokenizer.vocab.stoi[EOS_TOKEN]
    decoder_hidden = None
    for t in range(1, max_len):
      output, decoder_hidden, _ = self.decoder(input, decoder_hidden, encoder_output_states)
      input = output.max(1)[1]
      token = input.item()
      ru_tokens.append(token)
      if token == EOS_TOKEN_ID:
        break
    return ru_tokens


def create_tokenizer_ru(file: str):
  ru_field = Field(
    tokenize=tokenization,
    init_token=BOS_TOKEN,
    eos_token=EOS_TOKEN,
    pad_token=PAD_TOKEN,
    unk_token=UNK_TOKEN
  )
  with open(file, 'r') as f:
    contents = f.read().split('\n')[:-1]
    ru_lines = [c.split('\t')[1] for c in contents]
    ru_field.build_vocab(ru_lines)

  ru_field.numericalize = lambda *args, **kwargs: numericalize(ru_field, *args, **kwargs)
  return ru_field


def init_arguments():
  encoder_setup = {
    'name': 'bert-base-uncased'
  }
  enc_tokenizer = BertTokenizer.from_pretrained(encoder_setup['name'])
  enc_model = BertModel.from_pretrained(encoder_setup['name'])

  ru_field = create_tokenizer_ru('data.txt')
  hidden_size = 768
  decoder_setup = {
    'hidden_size': hidden_size,
    'input_size': 128,
    'bidirectional': False,
    'dropout': 0.3,
    'other_dropout': 0.2,
    'layers': 2
  }
  n_tokens = len(ru_field.vocab.stoi)
  pad_idx = ru_field.vocab.stoi[PAD_TOKEN]
  dec_embedding = nn.Embedding(n_tokens, decoder_setup['input_size'], padding_idx=pad_idx)
  attention = LuongAttention(hidden_size, False, n_tokens)
  dec_model = RNN_ModelDecoder('GRU', decoder_setup, dec_embedding, attention)


  train_params = {
    'lr': 0.001,
    'epochs': 10,
    'batch_size': 12
  }

  return (enc_tokenizer, enc_model), (ru_field, dec_model), train_params


def build_seq2seq(encoderr, decoderr, model_name):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  (enc_tokenizer, enc_model), (dec_tokenizer, dec_model) = encoderr, decoderr
  seq2seq = BERT2RNN(enc_model, enc_tokenizer, dec_model, dec_tokenizer, device, model_name).to(device)
  return seq2seq, device


if __name__ == '__main__':
  setup_logger()
  model_name = 'bert2rnn'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_bert2rnn')
  (enc_tokenizer, enc_model), (dec_tokenizer, dec_model), train_params = init_arguments()
  datasets = TranslationDataset.from_file('data.txt', '\t').split([0.8, 0.15, 0.05])
  seq2seq, device = build_seq2seq((enc_tokenizer, enc_model), (dec_tokenizer, dec_model), model_name)

  pad_idx = dec_tokenizer.vocab.stoi[PAD_TOKEN]
  closured_collate = build_collator(enc_tokenizer, dec_tokenizer, device)
  optimizer, scheduler, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params,
                                                                                  seq2seq,
                                                                                  datasets,
                                                                                  device,
                                                                                  pad_idx,
                                                                                  prepare_iterators,
                                                                                  num_workers=0,
                                                                                  collate_fn=closured_collate)
  convert_text = lambda x: get_text(x, dec_tokenizer)
  train_epochs( # todo: тренировка gpt2 - на каждом шаге менять attention mask
    seq2seq,
    train_iterator,
    valid_iterator,
    optimizer,
    scheduler,
    criterion,
    train_params['epochs'],
    writer,
    lambda x, device: enc_tokenizer(x, return_tensors='pt', padding=True).to(device),
    convert_text,
    labels_from_target
  )

  score = bleu_score(seq2seq, test_iterator, dec_tokenizer.token_to_id)
