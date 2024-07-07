from typing import List, Union, Generator, Dict
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import itertools
from torch import optim


# COPRUS
class Corpus:
    def __init__(self, file_name: str, window_size: int = 1):
        self.file_name = file_name
        self.window_size = window_size
        self.word2id: Dict[str, int] = {}
        self.id2word: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.data: Union[List, torch.Tensor] = []
        self.__build_data()

    def __iter__(self) -> Generator[str, None, None]:
        with open(self.file_name, "r") as file:
            for line in file:
                yield line.strip()

    def __update_map(self, text: List[str]) -> None:
        for word in text:
            if word not in self.word2id:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1

    def __write_pairs(self, text: List[str]) -> None:
        for i, word in enumerate(text):
            center = self.word2id[word]
            start = max(0, i - self.window_size)
            end = min(len(text), i + self.window_size + 1)
            context = itertools.chain(text[start:i], text[i + 1 : end])
            self.data.extend(  # type: ignore
                (center, self.word2id[cnt]) for cnt in context if cnt in self.word2id
            )

    def __build_data(self) -> None:
        for line in self:
            text = [word.strip() for word in line.split()]
            self.__update_map(text)
            self.__write_pairs(text)
        self.data = torch.tensor(list(set(self.data)), dtype=torch.long)


# DATASET
class CorpusData(Dataset):
    def __init__(self, corpus):
        self.data = corpus.data
        self.V = corpus.vocab_size

    def __getitem__(self, idx):
        c, o = self.data[idx]
        return c, o

    def __len__(self):
        return len(self.data)


# MODEL
class SkipGram(nn.Module):
    def __init__(self, num_words, emb_dim):
        super(SkipGram, self).__init__()
        self.num_words = num_words
        self.emb_dim = emb_dim
        self.vEmbedding = nn.Embedding(num_words, emb_dim)
        self.uEmbedding = nn.Linear(emb_dim, num_words, bias=False)

    def forward(self, v):
        v = self.vEmbedding(v)
        output = self.uEmbedding(v)
        return output

    def get_metrics(self, idx):
        with torch.no_grad():
            v = self.vEmbedding.weight[idx]
            similarities = torch.cosine_similarity(
                v.unsqueeze(0), self.uEmbedding.weight, dim=1
            )
        return similarities, 1 - similarities

    # MODEL TRAINER
    def train_model(self, dataloader, num_epochs, lr, device):
        optimizer = optim.NAdam(self.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        N = len(dataloader.dataset)
        log = []
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
            for center, context in pbar:
                output = self(center.to(device))
                loss = criterion(output, context.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * center.size(0)
                pbar.set_postfix(loss=f"{total_loss/N:.2f}")
            log.append(total_loss / N)
        self.eval()
        return self, log