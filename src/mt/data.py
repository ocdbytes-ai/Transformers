import collections
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfileobj
from typing import cast, overload
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader, TensorDataset


class Vocab:
    def __init__(
        self,
        tokens: list[list[str]] | list[str] | None = None,
        min_freq: int = 0,
        reserved_tokens: list[str] | None = None,
    ) -> None:
        tokens = tokens or []
        flat_tokens = (
            [token for line in cast(list[list[str]], tokens) for token in line]
            if tokens and isinstance(tokens[0], list)
            else cast(list[str], tokens)
        )
        counter = collections.Counter(flat_tokens)
        self.token_freqs = sorted(counter.items(), key=lambda item: item[1], reverse=True)
        self.index_to_token: list[str] = sorted(
            {"<unk>", *(reserved_tokens or []), *(
                token for token, frequency in self.token_freqs if frequency >= min_freq
            )}
        )
        self.token_to_index = {
            token: index for index, token in enumerate(self.index_to_token)
        }

    def __len__(self) -> int:
        return len(self.index_to_token)

    @overload
    def __getitem__(self, tokens: str) -> int: ...

    @overload
    def __getitem__(self, tokens: list[str] | tuple[str, ...]) -> list[int]: ...

    def __getitem__(self, tokens: str | list[str] | tuple[str, ...]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.token_to_index.get(tokens, self.unk)
        return [self.token_to_index.get(token, self.unk) for token in tokens]

    def to_tokens(self, indices: int | list[int]) -> str | list[str]:
        if isinstance(indices, int):
            return self.index_to_token[indices]
        return [self.index_to_token[index] for index in indices]

    @property
    def unk(self) -> int:
        return self.token_to_index["<unk>"]


@dataclass
class MTDataInput:
    data_url: str
    zip_data: bool
    max_steps: int
    data_dir: Path = Path("data")


@dataclass
class DataLoaderConfig:
    train_percentage: int
    validation_percentage: int
    batch_size: int


class MTData:
    def __init__(self, inputs: MTDataInput) -> None:
        self.inputs = inputs
        self.file_path = inputs.data_dir
        self.src_vocab = Vocab()
        self.tgt_vocab = Vocab()

    def download_and_extract(self) -> Path:
        self.inputs.data_dir.mkdir(parents=True, exist_ok=True)
        file_name = Path(urlparse(self.inputs.data_url).path).name
        if not file_name:
            raise ValueError("data_url must contain a file name")

        file_path = self.inputs.data_dir / file_name
        if not file_path.exists():
            temporary_path = file_path.with_suffix(file_path.suffix + ".part")
            request = Request(self.inputs.data_url, headers={"User-Agent": "curl/8.0"})
            with urlopen(request) as response, temporary_path.open("wb") as output:
                copyfileobj(response, output)
            temporary_path.replace(file_path)

        if not self.inputs.zip_data:
            self.file_path = file_path
            return file_path

        extract_dir = file_path.with_suffix("")
        extract_dir.mkdir(exist_ok=True)
        root = extract_dir.resolve()
        with ZipFile(file_path) as archive:
            if any(
                not (root / member.filename).resolve().is_relative_to(root)
                for member in archive.infolist()
            ):
                raise ValueError("Archive contains an unsafe path")
            archive.extractall(extract_dir)
        self.file_path = extract_dir
        return extract_dir

    def read_data(self, file_name: str) -> str:
        return (self.file_path / file_name).read_text(encoding="utf-8")

    @staticmethod
    def preprocess(text: str) -> str:
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        output: list[str] = []
        for index, char in enumerate(text.lower()):
            if index and char in ",.!?" and text[index - 1] != " ":
                output.append(" " + char)
            else:
                output.append(char)
        return "".join(output)

    @staticmethod
    def tokenize(text: str) -> tuple[list[list[str]], list[list[str]]]:
        pairs = [line.split("\t", 2)[:2] for line in text.splitlines()]
        source = [(pair[0] + " <eos>").split() for pair in pairs]
        target = [(pair[1] + " <eos>").split() for pair in pairs]
        return source, target

    def build_dataloaders(
        self,
        source: list[list[str]],
        target: list[list[str]],
        config: DataLoaderConfig,
    ) -> tuple[
        DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]
    ]:
        if len(source) != len(target) or not source:
            raise ValueError("Source and target corpora must be non-empty and equal in size")
        if (
            config.train_percentage <= 0
            or config.validation_percentage <= 0
            or config.train_percentage + config.validation_percentage != 100
            or config.batch_size <= 0
        ):
            raise ValueError("Percentages must be positive and total 100; batch size must be positive")

        def build_data(
            sentences: list[list[str]], vocab: Vocab | None, is_target: bool
        ) -> tuple[torch.Tensor, Vocab, torch.Tensor]:
            if is_target:
                sentences = [["<bos>", *sentence] for sentence in sentences]
            vocab = vocab or Vocab(
                sentences, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"]
            )
            sentences = [
                sentence[: self.inputs.max_steps]
                + ["<pad>"] * max(0, self.inputs.max_steps - len(sentence))
                for sentence in sentences
            ]
            data = torch.tensor(
                [vocab[sentence] for sentence in sentences], dtype=torch.long
            )
            valid_lengths = (data != vocab["<pad>"]).to(torch.int32).sum(1)
            return data, vocab, valid_lengths

        indices = cast(list[int], torch.randperm(len(source)).tolist())
        train_size = len(source) * config.train_percentage // 100
        if train_size == 0 or train_size == len(source):
            raise ValueError("Corpus is too small for the requested split")
        train_indices, validation_indices = indices[:train_size], indices[train_size:]

        def select(sentences: list[list[str]], selected: list[int]) -> list[list[str]]:
            return [sentences[index] for index in selected]

        train_source, self.src_vocab, train_source_lengths = build_data(
            select(source, train_indices), None, False
        )
        train_target, self.tgt_vocab, train_target_lengths = build_data(
            select(target, train_indices), None, True
        )
        validation_source, _, validation_source_lengths = build_data(
            select(source, validation_indices), self.src_vocab, False
        )
        validation_target, _, validation_target_lengths = build_data(
            select(target, validation_indices), self.tgt_vocab, True
        )

        train_data = TensorDataset(
            train_source, train_source_lengths, train_target, train_target_lengths
        )
        validation_data = TensorDataset(
            validation_source,
            validation_source_lengths,
            validation_target,
            validation_target_lengths,
        )
        return (
            DataLoader(train_data, config.batch_size, shuffle=True),
            DataLoader(validation_data, config.batch_size),
        )
