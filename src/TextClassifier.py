import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np


class TextClassifier(nn.Module):
    def __init__(self, nbrclass, embedding, train_embeddings=False) -> None:
        super(TextClassifier, self).__init__()
        self.embedding = self.__embedding(embedding, train_embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding.num_embeddings, 64), 
            nn.Linear(64, nbrclass)
        )

    def forward(self, x):
        x = self.mean_embedding(x)
        return self.classifier(x)

    def __embedding(self, embedding, train_embeddings):
        embedding.requires_grad = train_embeddings
        return embedding

    def mean_embedding(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return x

    def train_classifier(self, dataloader, num_epochs, lr, device):
        optimizer = optim.NAdam(self.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        N = len(dataloader.dataset)
        log = []
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)

                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()

                pbar.set_postfix(
                    loss=f"{total_loss/N:.2f}", accuracy=f"{correct/N:.2f}"
                )

            avg_loss = total_loss / N
            accuracy = correct / N
            log.append((avg_loss, accuracy))

        self.eval()
        return self, log


class CorpusClassifier:
    def __init__(self, text, label, mapping):
        self.x_file = text
        self.y_file = label
        self.x = []
        self.y = []
        self.word2id = mapping
        self.__read_data()

    def __read_data(self):
        with open(self.y_file) as f:
            for y in f:
                self.y.append(int(y.strip()))

        with open(self.x_file) as file:
            for line in file:
                line = line.strip().split()
                text = [self.word2id[word] for word in line]
                self.x.append(text)


class ClassifierData(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        x = torch.tensor(self.data.x[idx], dtype=torch.long)
        y = torch.tensor(self.data.y[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data.y)


def collate_fn(batch):
    # Separate the inputs and targets
    input_, targets = zip(*batch)
    # Pad sequences
    input_ = pad_sequence(input_, batch_first=True, padding_value=0)
    # Stack targets
    targets = torch.stack(targets)
    return input_, targets


def dataLoader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
