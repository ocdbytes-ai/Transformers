import math
from dataclasses import dataclass

import torch
from torch import nn

from .ops import masked_softmax


@dataclass
class DotProductAttentionInputs:
    dropout: float


class DotProductAttention(nn.Module):
    def __init__(self, dot_product_attention_inputs: DotProductAttentionInputs) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dot_product_attention_inputs.dropout)
        self.attention_weights = torch.empty(0)

    def forward(
        self,
        # (batch_size, num_queries, len_query)
        queries: torch.Tensor,
        # (batch_size, num_key_val_pairs, len_key)
        keys: torch.Tensor,
        # (batch_size, num_key_val_pairs, len_value)
        values: torch.Tensor,
        # (batch_size) || (batch_size, num_queries)
        valid_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        len_query = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        # For shape matching
        # (batch_size, num_key_val_pairs, len_query/key) -> (batch_size, len_query/keys, num_key_val_pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(len_query)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


@dataclass
class MultiheadAttentionInputs:
    num_hiddens: int
    num_heads: int
    dropout: float
    bias: bool


class MultiheadAttention(nn.Module):
    def __init__(self, inputs: MultiheadAttentionInputs):
        super().__init__()
        self.num_heads = inputs.num_heads
        self.attention = DotProductAttention(DotProductAttentionInputs(inputs.dropout))
        self.W_q = nn.LazyLinear(inputs.num_hiddens, bias=inputs.bias)
        self.W_k = nn.LazyLinear(inputs.num_hiddens, bias=inputs.bias)
        self.W_v = nn.LazyLinear(inputs.num_hiddens, bias=inputs.bias)
        self.W_o = nn.LazyLinear(inputs.num_hiddens, bias=inputs.bias)

    def forward(
        self,
        # (batch_size, num_queries, num_hiddens)
        queries: torch.Tensor,
        # (batch_size, num_key_val_pairs, num_hiddens)
        keys: torch.Tensor,
        # (batch_size, num_key_val_pairs, num_hiddens)
        values: torch.Tensor,
        # (batch_size) / (batch_size, num_queries)
        valid_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0
            )

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat)

    def transpose_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """Transposition for parallel computation of multiple attention heads."""
        # input x : (batch_size, num_queries/ num_key_val_pairs, num_hiddens)
        #
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    def transpose_output(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse the operation of transpose_qkv."""
        x = x.reshape(-1, self.num_heads, x.shape[1], x.shape[2])
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)
