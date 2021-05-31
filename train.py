from typing import List

from torch.utils.tensorboard import SummaryWriter
from torchtext.legacy.data import BucketIterator
import torch
from torch import optim, nn
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm, trange
import logging

from data import get_text, RU_field, translate, SAMPLES

CLIP = 1
logger = logging.getLogger('runner')


def _len_sort_key(x):
  return len(x.en)

def prepare(train_params, model, dataset, device, pad_idx):
  BATCH_SIZE = train_params['batch_size']
  train_data, valid_data, test_data = dataset
  train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=_len_sort_key
  )

  optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])
  criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
  return optimizer, criterion, (train_iterator, valid_iterator, test_iterator)


def train(model, iterator, optimizer, criterion):
  model.train()

  epoch_loss = 0
  history = []
  for i, batch in enumerate(iterator):

    src = batch.en
    trg = batch.ru

    optimizer.zero_grad()

    output = model(src, trg)

    # trg = [trg sent len, batch size]
    # output = [trg sent len, batch size, output dim]

    output = output.view(-1, output.shape[-1])
    trg = trg[1:].view(-1)

    # trg = [(trg sent len - 1) * batch size]
    # output = [(trg sent len - 1) * batch size, output dim]

    loss = criterion(output, trg)

    loss.backward()

    # Let's clip the gradient
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

    optimizer.step()

    epoch_loss += loss.item()
    history.append(loss.item())
  return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):

  model.eval()

  epoch_loss = 0

  history = []

  with torch.no_grad():
    for i, batch in enumerate(iterator):
      src = batch.en
      trg = batch.ru

      output = model(src, trg, 0)  # turn off teacher forcing

      # trg = [trg sent len, batch size]
      # output = [trg sent len, batch size, output dim]

      output = output.view(-1, output.shape[-1])
      trg = trg[1:].view(-1)

      # trg = [(trg sent len - 1) * batch size]
      # output = [(trg sent len - 1) * batch size, output dim]

      loss = criterion(output, trg)
      epoch_loss += loss.item()

  return epoch_loss / len(iterator)


def train_epochs(model,
                 iterator_train,
                 iterator_val,
                 optimizer,
                 criterion,
                 epochs,
                 writer: SummaryWriter):
  logger.info('Start training')
  train_losses = []
  val_losses = []
  for epoch in trange(1, epochs + 1):
    train_epoch_loss = train(model, iterator_train, optimizer, criterion)
    val_epoch_loss = evaluate(model, iterator_val, criterion)

    translated_samples = translate(model, SAMPLES)

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

  logger.info('Finish training')
  return train_losses, val_losses

def bleu_score(model, iterator_test):
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

      original_text.extend([get_text(x, RU_field.vocab) for x in trg.cpu().numpy().T])
      generated_text.extend([get_text(x, RU_field.vocab) for x in output[1:].detach().cpu().numpy().T])
  logger.info('Finished BLEU scoring')
  score = corpus_bleu([[text] for text in original_text], generated_text) * 100
  logger.info('BLEU score: %.2f', score)

  return score
