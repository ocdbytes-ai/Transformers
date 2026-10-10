import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from typing_extensions import override

from src.attention import MultiheadAttention, MultiheadAttentionInputs
from src.ops import PositionalEncoding, PositionalEncodingInputs
from src.seq2seq import AttentionDecoder


@dataclass
class PositionwiseFFNInput:
    num_hiddens: int
    num_outputs: int


class PositionwiseFFN(nn.Module):
    """
    PositionwiseFFN is a small MLP to apply non linear
    transformation to the tokens. FFN layers act like
    key-value memories: rows of W1 match input patterns,
    and rows of W2 write out associated output distributions.
    Source: "Transformer Feed-Forward Layers Are Key-Value Memories".
    """

    def __init__(self, inputs: PositionwiseFFNInput) -> None:
        super().__init__()
        self.dense1 = nn.LazyLinear(inputs.num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(inputs.num_outputs)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(X)))


@dataclass
class AddNormInput:
    norm_shape: int
    dropout: float


class AddNorm(nn.Module):
    """
    A Normalization layer added after each pass through either attention or FFN
    We define the layer as a input sanitation layer which keeps the inputs (features) across
    the blocks consistent in scale across the layers.

    We basically take the sublayer output to its input and then normalize each
    token vector of that sum and thus producing a single tensor of information.

    input -------|
                 | ---> AddNorm (input, output)
    output ------|

    input : input to the sublayer like a FFN or Attention
    output : output from that sublayer
    """

    def __init__(self, inputs: AddNormInput):
        super().__init__()
        self.dropout = nn.Dropout(inputs.dropout)
        self.ln = nn.LayerNorm(inputs.norm_shape)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.ln(self.dropout(Y) + X)


@dataclass
class TransformerEncoderBlockInput:
    num_hiddens: int
    num_hiddens_ffn: int
    num_heads: int
    dropout: float
    use_bias: bool = False


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder BLock

    Each block has its own multi head attention and a FFN
    After each layer the features are normalized using the
    Add Norm layer.
    """

    def __init__(self, inputs: TransformerEncoderBlockInput):
        super().__init__()
        self.attention = MultiheadAttention(
            MultiheadAttentionInputs(
                num_heads=inputs.num_heads,
                num_hiddens=inputs.num_hiddens,
                dropout=inputs.dropout,
                bias=inputs.use_bias,
            )
        )
        self.add_norm1 = AddNorm(
            AddNormInput(norm_shape=inputs.num_hiddens, dropout=inputs.dropout)
        )
        self.ffn = PositionwiseFFN(
            PositionwiseFFNInput(
                num_hiddens=inputs.num_hiddens_ffn, num_outputs=inputs.num_hiddens
            )
        )
        self.add_norm2 = AddNorm(
            AddNormInput(norm_shape=inputs.num_hiddens, dropout=inputs.dropout)
        )

    def forward(self, X: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
        Y = self.add_norm1(X, self.attention(X, X, X, valid_lens))
        return self.add_norm2(Y, self.ffn(Y))


@dataclass
class TransformerEncoderInput:
    vocab_size: int
    num_hiddens: int
    num_hiddens_ffn: int
    num_heads: int
    num_blocks: int
    dropout: float
    use_bias: bool = False


class TransformerEncoder(nn.Module):
    def __init__(self, inputs: TransformerEncoderInput) -> None:
        super().__init__()
        self.num_hiddens = inputs.num_hiddens
        self.embedding = nn.Embedding(inputs.vocab_size, inputs.num_hiddens)
        self.pos_encoding = PositionalEncoding(
            PositionalEncodingInputs(
                num_hiddens=inputs.num_hiddens, dropout=inputs.dropout
            )
        )
        self.attention_weights: list[torch.Tensor] = []
        self.blocks = nn.Sequential()
        for i in range(inputs.num_blocks):
            self.blocks.add_module(
                "block" + str(i),
                TransformerEncoderBlock(
                    TransformerEncoderBlockInput(
                        num_hiddens=inputs.num_hiddens,
                        num_hiddens_ffn=inputs.num_hiddens_ffn,
                        num_heads=inputs.num_heads,
                        dropout=inputs.dropout,
                        use_bias=inputs.use_bias,
                    )
                ),
            )

    @override
    def forward(self, x: torch.Tensor, valid_lens: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        self.attention_weights = []
        for block in self.blocks:
            assert isinstance(block, TransformerEncoderBlock)
            x = block(x, valid_lens)
            self.attention_weights.append(block.attention.attention.attention_weights)
        return x


@dataclass
class TransformerDecoderState:
    """State shared by all decoder blocks across decoding steps.

    Attributes:
        enc_outputs: Encoder output, shape (batch_size, num_src_steps, num_hiddens).
        enc_valid_lens: Valid source lengths, shape (batch_size,), or None for no masking.
        key_values: One entry per decoder block. Each entry holds every input that
            block has received so far, shape (batch_size, num_steps_so_far, num_hiddens),
            or None before the first step.
    """

    enc_outputs: torch.Tensor
    enc_valid_lens: torch.Tensor | None
    key_values: list[torch.Tensor | None]


# Per-block attention weights: (decoder self-attention, encoder-decoder attention).
TransformerDecoderAttentionWeights = tuple[list[torch.Tensor], list[torch.Tensor]]


@dataclass
class TransformerDecoderBlockInput:
    num_hiddens: int
    num_hiddens_ffn: int
    num_heads: int
    dropout: float
    block_index: int
    use_bias: bool


class TransformerDecoderBlock(nn.Module):
    """
    Decoder block for transformer

    the block has attention --> add norm --> attention --> add norm
    --> ffn --> add norm
    """

    def __init__(self, inputs: TransformerDecoderBlockInput):
        super().__init__()
        self.index = inputs.block_index
        self.attention_1 = MultiheadAttention(
            MultiheadAttentionInputs(
                num_hiddens=inputs.num_hiddens,
                num_heads=inputs.num_heads,
                dropout=inputs.dropout,
                bias=inputs.use_bias,
            )
        )
        self.add_norm1 = AddNorm(
            AddNormInput(norm_shape=inputs.num_hiddens, dropout=inputs.dropout)
        )
        self.attention_2 = MultiheadAttention(
            MultiheadAttentionInputs(
                num_hiddens=inputs.num_hiddens,
                num_heads=inputs.num_heads,
                dropout=inputs.dropout,
                bias=inputs.use_bias,
            )
        )
        self.add_norm2 = AddNorm(
            AddNormInput(norm_shape=inputs.num_hiddens, dropout=inputs.dropout)
        )
        self.ffn = PositionwiseFFN(
            PositionwiseFFNInput(
                num_hiddens=inputs.num_hiddens_ffn, num_outputs=inputs.num_hiddens
            )
        )
        self.add_norm3 = AddNorm(
            AddNormInput(norm_shape=inputs.num_hiddens, dropout=inputs.dropout)
        )

    @override
    def forward(
        self, X: torch.Tensor, state: TransformerDecoderState
    ) -> tuple[torch.Tensor, TransformerDecoderState]:
        enc_outputs, enc_valid_lens = state.enc_outputs, state.enc_valid_lens
        cached = state.key_values[self.index]
        key_values = X if cached is None else torch.cat((cached, X), dim=1)
        state.key_values[self.index] = key_values

        # Causal mask in every mode: query i attends to all cached tokens plus the
        # new tokens up to and including i.
        batch_size, num_steps, _ = X.shape
        past = 0 if cached is None else cached.shape[1]
        dec_valid_lens = torch.arange(
            past + 1, past + num_steps + 1, device=X.device
        ).repeat(batch_size, 1)

        # Self Attention
        X2 = self.attention_1(X, key_values, key_values, dec_valid_lens)
        Y = self.add_norm1(X, X2)
        # Encoder Decoder attention
        Y2 = self.attention_2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.add_norm2(Y, Y2)
        return self.add_norm3(Z, self.ffn(Z)), state


@dataclass
class TransformerDecoderInput:
    vocab_size: int
    num_hiddens: int
    num_hiddens_ffn: int
    num_heads: int
    num_blocks: int
    dropout: float
    use_bias: bool = False


class TransformerDecoder(
    AttentionDecoder[
        torch.Tensor, TransformerDecoderState, TransformerDecoderAttentionWeights
    ]
):
    def __init__(self, inputs: TransformerDecoderInput) -> None:
        super().__init__()
        self.num_hiddens = inputs.num_hiddens
        self.num_blocks = inputs.num_blocks
        self.embedding = nn.Embedding(inputs.vocab_size, inputs.num_hiddens)
        self.pos_encoding = PositionalEncoding(
            PositionalEncodingInputs(
                num_hiddens=inputs.num_hiddens, dropout=inputs.dropout
            )
        )
        self.blocks = nn.Sequential()
        for i in range(inputs.num_blocks):
            self.blocks.add_module(
                "block" + str(i),
                TransformerDecoderBlock(
                    TransformerDecoderBlockInput(
                        num_hiddens=inputs.num_hiddens,
                        num_hiddens_ffn=inputs.num_hiddens_ffn,
                        num_heads=inputs.num_heads,
                        dropout=inputs.dropout,
                        block_index=i,
                        use_bias=inputs.use_bias,
                    )
                ),
            )
        self.dense = nn.LazyLinear(inputs.vocab_size)
        self._attention_weights: TransformerDecoderAttentionWeights = ([], [])

    @override
    def init_state(
        self, enc_outputs: torch.Tensor, enc_valid_lens: torch.Tensor | None
    ) -> TransformerDecoderState:
        return TransformerDecoderState(
            enc_outputs=enc_outputs,
            enc_valid_lens=enc_valid_lens,
            key_values=[None] * self.num_blocks,
        )

    @override
    def forward(
        self, x: torch.Tensor, state: TransformerDecoderState
    ) -> tuple[torch.Tensor, TransformerDecoderState]:
        # Blocks write to key_values, so copy it; the caller's state stays unchanged
        # and several beams can continue from one state.
        state = replace(state, key_values=list(state.key_values))
        # In step-by-step decoding, new tokens continue after the cached ones.
        cached = state.key_values[0]
        offset = 0 if cached is None else cached.shape[1]
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens), offset)
        self_attention: list[torch.Tensor] = []
        enc_dec_attention: list[torch.Tensor] = []
        for block in self.blocks:
            assert isinstance(block, TransformerDecoderBlock)
            x, state = block(x, state)
            self_attention.append(block.attention_1.attention.attention_weights)
            enc_dec_attention.append(block.attention_2.attention.attention_weights)
        self._attention_weights = (self_attention, enc_dec_attention)
        return self.dense(x), state

    @property
    @override
    def attention_weights(self) -> TransformerDecoderAttentionWeights:
        return self._attention_weights
