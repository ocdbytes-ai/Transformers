from dataclasses import dataclass

import torch
from torch import nn


def masked_softmax(x: torch.Tensor, valid_lengths: torch.Tensor | None) -> torch.Tensor:
    # x : shape (batch_size, num_queries, num_keys/values)
    # num_keys : each query can attend to
    # valid_lengths : A tensor description which tells valid lengths for per batch or per query
    # - if valid_lengths.dim == 1 : describes per batch length
    # - if valid_lengths.dim == 2 : describes per batch, per query lengths

    if valid_lengths is None:
        return nn.functional.softmax(x, dim=-1)
    else:
        shape = x.shape
        if valid_lengths.dim() == 1:
            valid_lengths = torch.repeat_interleave(valid_lengths, shape[1])
        else:
            valid_lengths = valid_lengths.reshape(-1)

        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lengths, -1e6)
        return nn.functional.softmax(x.reshape(shape), dim=-1)


def sequence_mask(
    x: torch.Tensor, valid_lengths: torch.Tensor, value: float = 0.0
) -> torch.Tensor:
    # get the number of queries
    max_length = x.size(1)
    # Add a dimension in front
    positions = torch.arange(max_length, device=x.device).unsqueeze(0)
    lengths = valid_lengths.unsqueeze(1)
    mask = positions < lengths
    return x.masked_fill(~mask, value)


@dataclass
class PositionalEncodingInputs:
    num_hiddens: int
    dropout: float
    max_len: int = 1000


class PositionalEncoding(nn.Module):
    def __init__(self, inputs: PositionalEncodingInputs):
        super().__init__()
        self.dropout = nn.Dropout(inputs.dropout)
        self.P = torch.zeros((1, inputs.max_len, inputs.num_hiddens))
        X = torch.arange(inputs.max_len, dtype=torch.float32).reshape(
            -1, 1
        ) / torch.pow(
            10000,
            torch.arange(0, inputs.num_hiddens, 2, dtype=torch.float32)
            / inputs.num_hiddens,
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return self.dropout(X + self.P[:, offset : offset + X.shape[1], :].to(X.device))
