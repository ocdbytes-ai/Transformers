# Transformer

A from-scratch PyTorch implementation of the Transformer encoder-decoder from [Attention Is All You Need](https://arxiv.org/abs/1706.03762), trained on English-to-French translation. The code follows the structure of [Dive into Deep Learning, §11.7](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html), with type annotations, dataclass inputs and fixes for step-by-step decoding.

## Results

English → French, 192,416 training and 48,105 validation sentence pairs, 15 epochs on an Apple-silicon GPU (MPS).

| Model | Best validation loss | Validation BLEU-4 (greedy) | Training time |
| --- | ---: | ---: | ---: |
| Transformer (this project) | **1.1095** (epoch 14) | **40.71** | 25.9 min |
| GRU + additive attention ([Attention](https://github.com/ocdbytes-ai/Attention)) | 1.2158 (epoch 12) | 38.60 | 34.7 min |

- Both models use the same data split (seed 42, 80/20), vocabularies, loss and BLEU code.
- Validation loss is token-level cross-entropy with teacher forcing.
- BLEU is corpus BLEU-4 with add-one smoothing, computed on token ids from this project's tokenizer. It is useful for comparing the two models here, but not comparable to published WMT scores.
- BLEU was measured on the weights after epoch 15. For the Transformer, validation loss stopped improving after epoch 12 (1.1106 → 1.1128 → 1.1095 → 1.1097), so 12–15 epochs is enough with the current settings.

## Project structure

```
src/
  ops.py            masked softmax, sequence mask, sinusoidal positional encoding
  attention.py      scaled dot-product attention, multi-head attention
  transformer.py    position-wise FFN, add & norm, encoder and decoder (blocks + stacks)
  seq2seq.py        generic Encoder / Decoder / AttentionDecoder base classes
  mt/data.py        dataset download, preprocessing, vocabulary, padded data loaders
notebooks/
  ops.ipynb                    walkthrough of splitting Q/K/V into heads and attention scores
  positional_encoding.ipynb    plot of the sinusoidal encoding columns
  machine_translation.ipynb    training, beam-search translation, attention heatmaps, BLEU
tests/
  test_mt_data.py              data pipeline on a two-sentence corpus
  test_transformer.py          full-sequence decoding matches step-by-step decoding
```

## Model

| Setting | Value |
| --- | --- |
| Parameters | 16.4M |
| Hidden size / FFN size | 256 / 1024 |
| Heads / blocks (encoder and decoder) | 4 / 2 |
| Dropout | 0.1 |
| Optimizer | Adam, learning rate 1e-3, gradient clipping at 1.0 |
| Batch size / max sequence length | 256 / 20 tokens |

How it works:

- **Normalization**: post-LN blocks, as in the original paper: `LayerNorm(x + Sublayer(x))`.
- **Masking**: padding in the source is masked in encoder self-attention and in encoder-decoder attention. The decoder uses a causal mask in both training and evaluation, so a full target sequence gives the same output as decoding one token at a time. `tests/test_transformer.py` checks this.
- **Decoder state**: `TransformerDecoderState` holds the encoder output, the source lengths and, for each decoder block, every input it has seen so far.
  - The decoder copies this cache on every call, so several beams can continue from one state without affecting each other.
  - New tokens get positional encodings that start after the cached tokens.
- **Beam search**: the beam size defaults to 4, and scores are normalized with the length penalty `((5 + length) / 6) ** 0.6` from [GNMT](https://arxiv.org/abs/1609.08144).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

The notebooks need a Jupyter kernel. To run them with this project's environment, add `ipykernel` once and select `.venv` as the kernel in your editor:

```bash
uv add --dev ipykernel
```

## Usage

Open `notebooks/machine_translation.ipynb` and run the cells in order:

1. **Data.** The first run downloads the corpus (about 8 MB) to `notebooks/data/`. Later runs use the cached copy.
2. **Model.** Builds the encoder and decoder, then runs one batch through them so the `LazyLinear` layers get their input sizes before the optimizer is created.
3. **Training.** A live loss chart updates in place. Progress lines (loss, perplexity, gradient norm, tokens/s, ETA, device memory) go to the notebook and to `logs/training_<timestamp>.log`.
4. **Saving.** The weights are saved to `translation_model.pt`.
5. **Translation.** Translates a list of sentences with greedy search and beam search.
6. **Attention.** For each translation, draws heatmaps of encoder self-attention, decoder self-attention and encoder-decoder attention for every block and head.
7. **Evaluation.** Computes BLEU-4 on the validation split.

To change the model or training settings, edit the constants at the top of the model cell (`NUM_HIDDENS`, `NUM_HEADS`, `EPOCHS`, `LOG_EVERY`, …).

## Development

Run the tests:

```bash
uv run python -m tests.test_mt_data
uv run python -m tests.test_transformer
```

Type-check with [basedpyright](https://docs.basedpyright.com/). It currently reports 0 errors:

```bash
basedpyright --pythonpath .venv/bin/python src tests
```

## Known limitations

- **Training schedule**: there is no learning-rate warmup or decay, no label smoothing and no early stopping. The original paper uses all three except early stopping.
- **Sentence length**: inputs and outputs are cut to 20 tokens.
- **Positional encoding**: the table `PositionalEncoding.P` is a plain attribute, not a registered buffer. It is copied to the device on every forward pass and is not included in `state_dict()`.

## Data

The English-French sentence pairs come from the [Tatoeba Project](https://tatoeba.org), packaged by [manythings.org/anki](https://www.manythings.org/anki/), and are released under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). Attribution: www.manythings.org/anki and tatoeba.org.

The data, model checkpoints and logs are not committed (see `.gitignore`).

## References

- Vaswani et al., 2017. [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Zhang et al. [Dive into Deep Learning, §11.7 The Transformer Architecture](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html).
- Wu et al., 2016. [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) (length penalty for beam search).
