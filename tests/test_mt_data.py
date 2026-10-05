import torch

from src.mt.data import DataLoaderConfig, MTData, MTDataInput


def test_local_translation_data_pipeline() -> None:
    torch.manual_seed(0)
    data = MTData(MTDataInput("unused", zip_data=False, max_steps=5))
    source, target = data.tokenize(data.preprocess("Go!\tVa!\nWait.\tAttends."))
    train, validation = data.build_dataloaders(
        source, target, DataLoaderConfig(50, 50, batch_size=1)
    )

    assert next(iter(train))[0].shape == (1, 5)
    assert next(iter(validation))[0].shape == (1, 5)
    assert data.tgt_vocab.to_tokens(data.tgt_vocab["<bos>"]) == "<bos>"


if __name__ == "__main__":
    test_local_translation_data_pipeline()
