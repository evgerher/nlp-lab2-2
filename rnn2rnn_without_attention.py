from rnn2rnn import *
from utils.train import compute_parameters_number


def init_embeds(encoder_setup, decoder_setup, dec_emb_setup, train_params):
  dataset, train_data, valid_data, test_data = load_dataset_local(EN_field, RU_field, 'data.txt')
  en_vocab = build_vocab(EN_field, dataset)
  ru_vocab = build_vocab(RU_field, dataset)

  n_tokens = len(ru_vocab.stoi)
  encoder_embedding = nn.Embedding(len(en_vocab.stoi), encoder_setup['input_size'], padding_idx=en_vocab.stoi[PAD_TOKEN])
  decoder_embedding = nn.Embedding(n_tokens, dec_emb_setup['embedding_size'], padding_idx=ru_vocab.stoi[PAD_TOKEN])

  attention = None # update is only here
  dataset = (train_data, valid_data, test_data)
  embeds = (encoder_embedding, decoder_embedding)
  vocabs = (en_vocab, ru_vocab)
  setups = (encoder_setup, decoder_setup)
  logger.info('Initialized params: loaded dataset, vocabs, embeds')
  return train_params, setups, vocabs, embeds, attention, dataset

if __name__ == '__main__':
  setup_logger()
  model_name = 'RNN2RNN_wo_attention'
  logger.info(f'Model {model_name}')
  writer = SummaryWriter('exp_RNN2RNN_wo_attention')
  encoder_setup, decoder_setup, dec_emb_setup, train_params = init_arguments()
  train_params, setups, vocabs, embeds, attention, datasets = init_embeds(encoder_setup, decoder_setup, dec_emb_setup, train_params)
  (en_vocab, ru_vocab) = vocabs
  seq2seq, device = build_seq2seq(setups, embeds, attention, model_name, en_vocab, ru_vocab)

  RU_SEQ_LEN = 50
  EN_SEQ_LEN = 45
  BATCH_SIZE = 32
  estimated_time = estimate_batch_time_simple(seq2seq, model_name, BATCH_SIZE, EN_SEQ_LEN, RU_SEQ_LEN, device, 100)
  nparams = compute_parameters_number(seq2seq, model_name)

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
