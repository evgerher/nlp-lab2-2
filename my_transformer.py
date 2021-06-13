from nltk.translate.bleu_score import corpus_bleu
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.logger import setup_logger
from utils.train import prepare, train_epochs, estimate_batch_time_simple, compute_parameters_number
from utils.transformer_utils import *
from utils.rnn_utils import prepare_iterators, get_text


def init_arguments(en_vocab, ru_vocab):
  encoder_setup = {
    'input_size': len(en_vocab),
    'hidden_size': 256,
    'nlayer': 3,
    'dropout': 0.15,
    'pf_size': 512,
    'nheads': 8,
  }

  decoder_setup = {
    'output_size': len(ru_vocab),
    'hidden_size': 256,
    'nlayer': 3,
    'dropout': 0.15,
    'pf_size': 512,
    'nheads': 8,
  }

  train_params = {
    'lr': 0.0003,
    'epochs': 15,
    'batch_size': 128
  }
  
  return encoder_setup, decoder_setup, train_params


def init_dataset():
  dataset, train_data, valid_data, test_data = load_dataset_local(EN_field, RU_field, 'data.txt')
  en_vocab = build_vocab(EN_field, dataset)
  ru_vocab = build_vocab(RU_field, dataset)

  dataset = (train_data, valid_data, test_data)
  vocabs = (en_vocab, ru_vocab)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info('Initialized params: loaded dataset, vocabs')
  return dataset, vocabs, device

def init_model(enc_setup, dec_setup, en_vocab, ru_vocab, device, model_name, init_weights=True):
  encoder = Encoder(enc_setup, device)
  decoder = Decoder(dec_setup, device)
  src_pad_idx = en_vocab.stoi[PAD_TOKEN]
  trg_pad_idx = ru_vocab.stoi[PAD_TOKEN]
  model = Seq2Seq(encoder, decoder, src_pad_idx, trg_pad_idx, device, model_name).to(device)

  if init_weights:
    model.apply(initialize_weights)

  return model


def bleu_score(model, iterator_test, get_text):
  logger.info('Start BLEU scoring')
  original_text = []
  generated_text = []
  model.eval()
  with torch.no_grad():
    for i, batch in tqdm(enumerate(iterator_test)):
      src = batch.en
      trg = batch.ru

      max_len = trg.shape[1]


      for en_tokens, ru_tokens_expected in zip(src, trg):
        ru_tokens = model.translate(en_tokens.unsqueeze(0), max_len + 10, add_fields=False, tensorize=False, convert_tokens=False)

        original = ' '.join(get_text(ru_tokens_expected))
        generated = ' '.join(get_text(ru_tokens))

        original_text.append(original)
        generated_text.append(generated)
  score = corpus_bleu([[text] for text in original_text], generated_text) * 100
  logger.info('Finished BLEU scoring')
  logger.info('BLEU score: %.2f', score)

  return score


if __name__ == '__main__':
  setup_logger()
  model_name = 'WORD_TRANSFORMER'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_WORD_TRANSFORMER')

  datasets, (en_vocab, ru_vocab), device = init_dataset()
  enc_setup, dec_setup, train_params = init_arguments(en_vocab, ru_vocab)
  seq2seq = init_model(enc_setup, dec_setup, en_vocab, ru_vocab, device, model_name)
  RU_SEQ_LEN = 50
  EN_SEQ_LEN = 45
  BATCH_SIZE = 32
  estimated_time = estimate_batch_time_simple(seq2seq, model_name, BATCH_SIZE, EN_SEQ_LEN, RU_SEQ_LEN, device, 100, True)
  nparams = compute_parameters_number(seq2seq, model_name)

  pad_idx = ru_vocab.stoi[PAD_TOKEN]
  optimizer, scheduler, criterion, (train_iterator, valid_iterator, test_iterator) = prepare(train_params,
                                                                                             seq2seq,
                                                                                             datasets,
                                                                                             device,
                                                                                             pad_idx,
                                                                                             prepare_iterators)
  convert_text = lambda x: get_text(x, lambda token: RU_field.vocab.itos[token])
  # train_epochs(
  #   seq2seq,
  #   train_iterator,
  #   valid_iterator,
  #   optimizer,
  #   scheduler,
  #   criterion,
  #   train_params['epochs'],
  #   writer,
  #   lambda x, device: EN_field.tokenize(x[0].lower()),
  #   convert_text,
  #   labels_from_target
  # )
  #
  # best_state = torch.load(f"{model_name}_best.pt", mapping=device)
  # seq2seq.load_state_dict(best_state, strict=False)
  score = bleu_score(seq2seq, test_iterator, convert_text)
