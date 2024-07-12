import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from typing import List, Dict
from tqdm import tqdm
from torch import optim
import itertools


class Corpus:
    def __init__(self, file_name: str, window: int = 1, k=3):
        self.file_name = file_name
        self.word2id: Dict[str, int] = {"<pad>":0, "<unk>":1}
        self.id2word: Dict[int, str] = {0:"<pad>", 1: "<unk>"}
        self.word_count = {}
        self.total = 0
        self.k = k
        self.vocab_size = 0
        self.window_size = window
        self.pairs = []
        self.noise_dist = np.array([])
        self.__init_data()
        self.__make_pairs()
        self.__sample_negative()

    def __init_data(self):
        with open(self.file_name, "r") as file:
            data = file.read()
        data = data.split()
        self.word_count = Counter(data)
        self.word_count["<pad>"] = 0
        self.word_count["<unk>"] = 0
        self.total = sum(list(self.word_count.values()))+2
        self.vocab_size = len(self.word_count) + 1
        sorted_count = sorted(self.word_count, key=self.word_count.get, reverse=True)  # type: ignore
        for idx, word in enumerate(sorted_count):
            self.word2id[word] = idx + 1
            self.id2word[idx + 1] = word

    def __get_pairs(self, text: List[str]):
        num_words = len(text)
        for i, word in enumerate(text):
            center = self.word2id[word]
            start = max(0, i - self.window_size)
            end = min(num_words, i + self.window_size + 1)
            context = itertools.chain(text[start:i], text[i + 1 : end])
            self.pairs.extend(  # type: ignore
                (center, self.word2id[cnt]) for cnt in context if cnt in self.word2id
            )

    def __make_pairs(self):
        with open(self.file_name) as data:
            for text in data:
                self.__get_pairs(text.strip().split())
        _tmp = set(self.pairs)
        _tmp = np.array(list(_tmp))
        self.pairs = np.array(_tmp)

    def __sample_negative(
        self,
    ):
        freq = {}
        for word in self.word2id.keys():
            freq[word] = self.word_count[word] / self.total
        unigram = np.array(list(freq.values())) ** (3 / 4)
        self.noise_dist = unigram / unigram.sum()
        self.noise_dist = torch.from_numpy(self.noise_dist)
        _neg = []
        for pair in self.pairs:
            _, b = pair
            _tmp = self.noise_dist.clone()
            _tmp[b] = 0.0
            _neg.append(
                torch.multinomial(input=_tmp, num_samples=self.k, replacement=True)
            )
        self.neg = np.asarray(_neg)
        i, j = self.pairs.shape  # type: ignore
        self.pairs = self.pairs.reshape(i, j, 1)  # type: ignore


class CorpusData(Dataset):
    def __init__(self, corpus):
        self.pairs = torch.from_numpy(corpus.pairs)
        self.negative = torch.from_numpy(corpus.neg)
        self.V = corpus.vocab_size

    def __getitem__(self, idx):
        c_positive, o_positive = self.pairs[idx]
        o_negative = self.negative[idx]
        return c_positive, o_positive, o_negative.unsqueeze(1)

    def __len__(self):
        return len(self.pairs)


class SGNS(nn.Module):
    def __init__(self, vocab_size, emb_dim) -> None:
        super(SGNS, self).__init__()
        self.vocab_size = vocab_size  # N
        self.emb_dim = emb_dim  # H
        self.vEmbedding = nn.Embedding(self.vocab_size, self.emb_dim)
        self.uEmbedding = nn.Embedding(self.vocab_size, self.emb_dim)

    def forward(self, c, o, neg):
        vv = self.vEmbedding(c)  # BxH
        uu = self.uEmbedding(o)  # BxH
        ng = self.uEmbedding(neg)  # BxKxH
        pos = torch.sigmoid(torch.einsum("bki,bik->bk", uu, vv.mT))  # Bx1
        neg = torch.sigmoid(torch.einsum("bjkl,blk->bjk", ng, vv.mT).neg())  # BxKx1
        return pos, neg

    def get_metrics(self, idx):
        with torch.no_grad():
            v = self.vEmbedding(idx)
            similarities = torch.cosine_similarity(
                self.vEmbedding.weight, v.unsqueeze(1).mT, dim=1
            )
        return similarities, 1 - similarities

    # MODEL TRAINER
    def train_model(self, dataloader, num_epochs, lr, device):
        optimizer = optim.NAdam(self.parameters(), lr)
        criterion = SGNSLoss()
        N = len(dataloader.dataset)
        log = []
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            pbar = tqdm(
                dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            for c, o, n in pbar:
                c, o, n = c.to(device), o.to(device), n.to(device)
                pos, neg = self(c, o, n)
                optimizer.zero_grad()
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * c.size(0)
                pbar.set_postfix(loss=f"{total_loss/N:.2f}")
            log.append(total_loss / N)
        self.eval()
        return self, np.array(log)


class SGNSLoss(nn.Module):
    def __init__(self):
        super(SGNSLoss, self).__init__()

    def forward(self, positive, negatives):
        a = torch.log(positive).neg()
        b = torch.log(negatives).neg().sum(1)
        return torch.mean(a + b)
