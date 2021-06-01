import torch
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):
  pass

class LuongAttention(Attention):
  # Luong model https://arxiv.org/abs/1508.04025
  # https://github.com/kevinlu1211/pytorch-batch-luong-attention/blob/master/models/luong_attention/luong_attention.py
  def __init__(self, hidden_size: int, bidirectional_encoder: bool, n_out: int):
    super().__init__()
    self.bidirectional_encoder = bidirectional_encoder
    self.hidden_size = hidden_size

    context_size = self.hidden_size * 2 if self.bidirectional_encoder else 1
    self.attn = nn.Linear(self.hidden_size, context_size)
    self.concat = nn.Linear(self.hidden_size + context_size, self.hidden_size)

  def forward(self, rnn_output, hidden, encoder_outputs):
    attn_energies = torch.bmm(self.attn(hidden).transpose(1, 0), encoder_outputs.permute(1, 2, 0))
    attn_weights = F.softmax(attn_energies, dim=-1)
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

    concat_input = torch.cat((rnn_output, context[:, -1]), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    return concat_output, attn_weights
