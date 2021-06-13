from torch.utils.tensorboard import SummaryWriter
import time
import torch
from torch import optim, nn
from tqdm import tqdm, trange
import logging

from utils.data import translate, SAMPLES

CLIP = 0.1
logger = logging.getLogger('runner')

def prepare(train_params, model, dataset, device, pad_idx, prepare_iterators, **kwargs):
  BATCH_SIZE = train_params['batch_size']
  train_data, valid_data, test_data = dataset
  train_iterator, valid_iterator, test_iterator = prepare_iterators(train_data, valid_data, test_data, BATCH_SIZE, device, **kwargs)

  optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])
  criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
  return optimizer, scheduler, criterion, (train_iterator, valid_iterator, test_iterator)


def train_epoch(model, iterator, optimizer, criterion, labels_from_target):
  model.train()
  epoch_loss = 0
  for i, batch in enumerate(iterator):
    src = batch.en
    trg = batch.ru

    optimizer.zero_grad()
    if 'cnn' in model.name.lower():
      tt = trg.T[:-1] # [seq_len - 1, batch_size]
    elif 'word_transformer' in model.name.lower():
      tt = trg[:, :-1]
    elif 'rnn' in model.name.lower():
      tt = trg
    output = model(src, tt)

    # trg = [trg sent len, batch size]
    # output = [trg sent len, batch size, output dim]
    output = output.contiguous().view(-1, output.shape[-1])
    expected_labels = labels_from_target(trg)

    # trg = [(trg sent len - 1) * batch size]
    # output = [(trg sent len - 1) * batch size, output dim]

    loss = criterion(output, expected_labels)
    loss.backward()
    # Let's clip the gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    optimizer.step()
    epoch_loss += loss.item()
  return epoch_loss / len(iterator)

def evaluate_epoch(model, iterator, criterion, labels_from_target):
  model.eval()
  epoch_loss = 0
  with torch.no_grad():
    for i, batch in enumerate(iterator):
      src = batch.en
      trg = batch.ru

      if 'cnn' in model.name.lower():
        tt = trg[:-1]
      elif 'word_transformer' in model.name.lower():
        tt = trg[:, :-1]
      else:
        tt = trg
      output = model(src, tt)
      # trg = [trg sent len, batch size]
      # output = [trg sent len, batch size, output dim]
      output = output.view(-1, output.shape[-1])
      expected_labels = labels_from_target(trg)

      # trg = [(trg sent len - 1) * batch size]
      # output = [(trg sent len - 1) * batch size, output dim]
      loss = criterion(output, expected_labels)
      epoch_loss += loss
  return epoch_loss / len(iterator)


def train_epochs(model,
                 iterator_train,
                 iterator_val,
                 optimizer,
                 scheduler,
                 criterion,
                 epochs,
                 writer: SummaryWriter,
                 encode_en,
                 get_text,
                 labels_from_target):
  logger.info('Start training')
  best_loss = float('inf')
  train_losses = []
  val_losses = []
  for epoch in trange(1, epochs + 1):
    train_epoch_loss = train_epoch(model, iterator_train, optimizer, criterion, labels_from_target) # todo: bert2gpt - tokens repeat - think b' past_key_values!
    val_epoch_loss = evaluate_epoch(model, iterator_val, criterion, labels_from_target)
    scheduler.step(val_epoch_loss)
    val_epoch_loss = val_epoch_loss.item()

    translated_samples = translate(model, SAMPLES, encode_en, get_text)

    train_losses.append(train_epoch_loss)
    val_losses.append(val_epoch_loss)
    logger.info('Epoch [%d] Train loss:\t%.3f', epoch, train_epoch_loss)
    logger.info('Epoch [%d] Val loss:\t%.3f', epoch, val_epoch_loss)
    writer.add_scalar('train_loss', train_epoch_loss, epoch)
    writer.add_scalar('val_loss', val_epoch_loss, epoch)

    for en_sample, ru_sample in zip(SAMPLES, translated_samples):
      msg = 'Translation [{}] === [{}]'.format(en_sample, ru_sample)
      logger.info(msg)
      writer.add_text('translation', msg, epoch)

    if val_epoch_loss < best_loss:
      best_loss = val_epoch_loss
      state_dict = model.state_dict()
      logger.info('New best model')
      torch.save(state_dict, f'{model.name}_best.pt')

    if epoch % 5 == 0:
      state_dict = model.state_dict()
      torch.save(state_dict, f'{model.name}_{epoch}.pt')

  logger.info('Finish training')
  return train_losses, val_losses


def estimate_batch_time_simple(model,
                        model_name: str,
                        batch_size: int,
                        seq_len_en: int,
                        seq_len_ru: int,
                        device,
                        trials: int,
                        batch_first: bool = False):
  assert trials > 0
  with torch.no_grad():
    model.eval()
    en = torch.zeros((seq_len_en, batch_size), dtype=torch.long, device=device)
    ru = torch.zeros((seq_len_ru, batch_size), dtype=torch.long, device=device)
    if batch_first:
      en = en.T
      ru = ru.T
    timer = 0

    for _ in range(trials):
      start = time.time()
      _ = model(en, ru)
      end = time.time()
      timer += (end - start)
    model.train()
    avg_time = timer / trials
    logger.info("Model %s computes sequence [batch_size: %d, en_seq_lenth: %d, ru_seq_length: %d] within %.5f seconds.",
                model_name, batch_size, seq_len_en, seq_len_ru, avg_time)
    return avg_time

def compute_parameters_number(model, model_name: str):
  nparams = sum(p.numel() for p in model.parameters())
  logger.info('Model %s has [%d] parameters', model_name, nparams)
  return nparams
