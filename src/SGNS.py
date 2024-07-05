import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class SkipGram(nn.Module):
    def __init__(self, num_words, emb_dim):
        super(SkipGram, self).__init__()
        self.num_words = num_words
        self.emb_dim = emb_dim
        self.center = nn.Linear(num_words, emb_dim)
        self.context = nn.Linear(emb_dim, num_words)

    def forward(self, input_):
        center_emb = self.center(input_)
        output = self.context(center_emb)
        return torch.relu(output)

    def get_similarity(self, idx):
        with torch.no_grad():
            center_emb = self.center.weight.T[idx]
            similarities = torch.cosine_similarity(
                center_emb, self.center.weight.T, dim=1
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
        center_id, y = self.data[idx]
        center_onehot = torch.zeros((self.V,))
        center_onehot[center_id] = 1
        context_onehot = torch.zeros((self.V,))
        context_onehot[y] = 1
        return center_onehot, context_onehot

    def __len__(self):
        return len(self.data)
