import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

from .preprocessing_lstm import yield_tokens, tokenizer
from .model_config import BATCH_SIZE


class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_batch(batch, vocab):
    label_list, text_list = [], []
    for _text, _label in batch:
        label_list.append(_label)
        processed_text = torch.tensor([vocab[token] for token in tokenizer(_text)], dtype=torch.int64)
        text_list.append(processed_text)
    return pad_sequence(text_list, padding_value=vocab["<PAD>"], batch_first=True), torch.stack(label_list)


def get_vocab(X_train, y_train):
    vocab = build_vocab_from_iterator(yield_tokens(zip(X_train, y_train)), specials=["<PAD>", "<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])
    return vocab


def setup_loaders(X_train, y_train, X_val, y_val):
    # Get vocab and initialize datasets
    vocab = get_vocab(X_train, y_train)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_dataset = SpamDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_batch(b, vocab))

    val_dataset = SpamDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_batch(b, vocab))

    return vocab, train_loader, val_loader
