import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import numpy as np


class TextClassifier(nn.Module):
    """
		Text Classifier models
    """
    def __init__(self, nbr_class, embedding, train_embeddings=False) -> None:
        super(TextClassifier, self).__init__()
        self.nbr_class = nbr_class
        self.embedding = self.__embedding(embedding, train_embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.nbr_class),
        )

    def forward(self, x):
        """
			Forward call
        """
        x = self.mean_embedding(x)
        return self.classifier(x)

    def __embedding(self, embedding, train_embeddings):
        """
			Freeze the embedding weights
        """
        embedding.requires_grad = train_embeddings
        return embedding

    def mean_embedding(self, x):
        """
			Compute the mean embeddings
        """
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return x

    def eval_classifier(self, dataloader, device):
        """
			Evaluate the classification model:
            * Compute accuracy
            * Compute loss
        """
        criterion = nn.CrossEntropyLoss()
        N = len(dataloader.dataset)
        log = []
        self.eval()
        total_loss = 0
        correct = 0
        conf_matrix = np.zeros((self.nbr_class, self.nbr_class), dtype="int32")
        pbar = tqdm(dataloader, total=len(dataloader), desc=f"Eval")
        with torch.no_grad():
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                output = self(x)
                loss = criterion(output, y)

                total_loss += loss.item() * x.size(0)
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()

                # Collect true and predicted labels for confusion matrix
                for i in range(y.size(0)):
                    conf_matrix[y[i].item(), predicted[i].item()] += 1  # type: ignore

                pbar.set_postfix(
                    loss=f"{total_loss/N:.2f}", accuracy=f"{correct/N:.2f}"
                )

        avg_loss = total_loss / N
        accuracy = correct / N

        return avg_loss, accuracy, conf_matrix

    def train_classifier(self, dataloader, num_epochs, lr, device):
        """
			Train the model
        """
        optimizer = optim.NAdam(self.parameters(), lr)
        criterion = nn.CrossEntropyLoss()
        N = len(dataloader.dataset)
        log = []
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}")
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)

                _, predicted = torch.max(output, 1)
                correct += (predicted == y).sum().item()

                pbar.set_postfix(
                    loss=f"{total_loss/N:.2f}", accuracy=f"{correct/N:.2f}"
                )

            avg_loss = total_loss / N
            accuracy = correct / N
            log.append((avg_loss, accuracy))

        self.eval()
        return self, np.array(log)


class CorpusClassifier:
    """
		Create an appropiate corpus structure for the classifier
    """
    def __init__(self, text, label, mapping):
        self.x_file = text
        self.y_file = label
        self.x = []
        self.y = []
        self.word2id = mapping
        self.__read_data()

    def __read_data(self):
        """
			Read the data from the files 
        """
        with open(self.y_file) as f:
            for y in f:
                self.y.append(int(y.strip()) - 1)

        with open(self.x_file) as file:
            """
				Create sentence target pairs.
            """
            for line in file:
                line = line.strip().split()
                text = [
                    self.word2id[word] if word in self.word2id else 1 for word in line
                ]
                self.x.append(text)


class ClassifierData(Dataset):
    """
		Custom dataset for the classifier 
    """
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        x = torch.tensor(self.data.x[idx], dtype=torch.long)
        y = torch.tensor(self.data.y[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.data.y)


def collate_fn(batch):
    """
     Pad sorter sequence with 0 (<pad>) to match the longest sequence
     to obtain a uniform bacht size.
    """
    input_, targets = zip(*batch)
    # Pad sequences
    input_ = pad_sequence(input_, batch_first=True, padding_value=0)
    # Stack targets
    targets = torch.stack(targets)
    return input_, targets


def dataLoader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
