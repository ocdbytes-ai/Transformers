from typing import Generic, TypeVar

import torch
from torch import nn
from typing_extensions import override

EncoderState = tuple[torch.Tensor, torch.Tensor]
DecoderState = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

EncoderOutputT = TypeVar("EncoderOutputT")
DecoderStateT = TypeVar("DecoderStateT")
AttentionWeightsT = TypeVar("AttentionWeightsT")


class Encoder(nn.Module):
    @override
    def forward(
        self, x: torch.Tensor, valid_lengths: torch.Tensor | None = None
    ) -> EncoderState:
        raise NotImplementedError


class Decoder(nn.Module, Generic[EncoderOutputT, DecoderStateT]):
    """Base decoder, generic over the encoder output and decoder state types.

    An RNN decoder would use `Decoder[EncoderState, DecoderState]`; the
    Transformer decoder uses `Decoder[torch.Tensor, TransformerDecoderState]`.
    """

    def init_state(
        self, _enc_outputs: EncoderOutputT, _enc_valid_lens: torch.Tensor | None, /
    ) -> DecoderStateT:
        raise NotImplementedError

    @override
    def forward(
        self, x: torch.Tensor, state: DecoderStateT
    ) -> tuple[torch.Tensor, DecoderStateT]:
        raise NotImplementedError


class AttentionDecoder(
    Decoder[EncoderOutputT, DecoderStateT],
    Generic[EncoderOutputT, DecoderStateT, AttentionWeightsT],
):
    """Decoder that also exposes the attention weights of its last forward pass."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def attention_weights(self) -> AttentionWeightsT:
        raise NotImplementedError
