from torchtext.legacy import data, datasets


def get_imdb_dataloaders(vocab_size, batch_size, device, root_dir=".data") -> None:
    text_field = data.Field(lower=True, include_lengths=True, batch_first=True)
    label_field = data.Field(sequential=False)

    tdata, _ = datasets.IMDB.splits(text_field, label_field, root=root_dir)
    train, test = tdata.split(0.8)
    text_field.build_vocab(train, max_size=vocab_size - 2)
    label_field.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits(
        (train, test), batch_size=batch_size, device=device
    )
    return train_iter, test_iter


if __name__=="__main__":
    train, test = get_imdb_dataloaders(10000, 32, "cpu")

