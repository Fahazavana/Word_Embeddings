import torch
from torch import nn
from torch.utils.data import Dataset

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


class SGNSLoss(nn.Module):
	def __init__(self):
		super(SGNSLoss, self).__init__()

	def forward(self, positive, negatives):
		a = torch.log(positive).neg()
		b = torch.log(negatives).neg().sum(1)
		return torch.mean(a + b)


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