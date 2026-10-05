import torch

from src.transformer import (
    TransformerDecoder,
    TransformerDecoderInput,
    TransformerEncoder,
    TransformerEncoderInput,
)


def test_full_sequence_matches_cached_steps_in_eval() -> None:
    torch.manual_seed(0)
    encoder = TransformerEncoder(TransformerEncoderInput(50, 32, 64, 4, 2, 0.0))
    decoder = TransformerDecoder(TransformerDecoderInput(60, 32, 64, 4, 2, 0.0))
    source = torch.randint(0, 50, (2, 7))
    source_lengths = torch.tensor([7, 5])
    target = torch.randint(0, 60, (2, 6))
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        state = decoder.init_state(encoder(source, source_lengths), source_lengths)
        full, _ = decoder(target, state)
        steps = []
        for t in range(target.shape[1]):
            logits, state = decoder(target[:, t : t + 1], state)
            steps.append(logits)

    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


if __name__ == "__main__":
    test_full_sequence_matches_cached_steps_in_eval()
