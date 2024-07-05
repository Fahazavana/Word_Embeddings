from typing import List, Union, Generator
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class SkipGram(nn.Module):
    def __init__(self, num_words, emb_dim):
        super(SkipGram, self).__init__()
        self.num_words = num_words
        self.emb_dim = emb_dim
        self.center = nn.Embedding(num_words, emb_dim)
        self.context = nn.Linear(emb_dim, num_words, bias=False)

    def forward(self, input_):
        center_emb = self.center(input_)
        output = self.context(center_emb)
        return output

    def get_similarity(self, idx):
        with torch.no_grad():
            center_emb = self.center.weight[idx]
            similarities = torch.cosine_similarity(
                center_emb.unsqueeze(0), self.center.weight, dim=1
            )
        return similarities

    def get_cosine_distance(self, idx):
        similarities = self.get_similarity(idx)
        return 1 - similarities


class CorpusData(Dataset):
    def __init__(self, corpus):
        self.data = torch.from_numpy(corpus.data)
        self.V = corpus.vocab_size

    def __getitem__(self, idx):
        c, o = self.data[idx]
        return c, o

    def __len__(self):
        return len(self.data)


class Corpus:
    def __init__(self, file_name, window_size=1):

        self.file_name = file_name
        self.window_size = window_size
        self.word2id = {}
        self.id2word = {}
        self.vocab_size = 0
        self.data: Union[List, np.ndarray] = []
        self.__build_data()

    def __iter__(self) -> Generator:
        with open(self.file_name, "r") as file:
            yield from file

    def __update_map(self, text):
        for word in text:
            if word not in self.word2id.keys():
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1

    def __write_pairs(self, text):
        num_words = len(text)
        for i, word in enumerate(text):
            center = self.word2id[word]
            num_words = len(text)
            context = (
                text[max(0, i - self.window_size) : i]
                + text[i + 1 : min(num_words, i + self.window_size + 1)]
            )
            self.data.extend(((center, self.word2id[cnt]) for cnt in context))  # type: ignore

    def __build_data(self) -> None:
        for line in self:
            text = line.strip().split()
            text = [word.strip() for word in text if len(word) > 2]
            self.__update_map(text)
            self.__write_pairs(text)
        self.data = np.array(list(set(self.data)), dtype="int32")
        
	


def train(model, dataloader, num_epochs, criterion, optimizer, device):
    N = len(dataloader.dataset)
    log = []
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}")
        for center, context in pbar:
            output = model(center.to(device))
            loss = criterion(output, context.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.size(0)
            pbar.set_postfix(loss=f"{total_loss/N:.2f}")
        log.append(total_loss / N)
    model.eval()
    return model, log
