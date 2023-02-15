import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, args):
        self.path = args['path']
        self.corpus, self.tokens_to_id, self.idx_to_tokens = self.read_file()
        self.emb_size = args['ebm_size']
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        coded_word = [self.tokens_to_id[char] for char in self.corpus[index]] + \
                     [self.tokens_to_id[' ']] * max(self.emb_size - len(self.corpus[index]), 0)

        next_word = coded_word[1:] + [self.tokens_to_id[' ']]

        return (torch.Tensor(coded_word[: self.emb_size]).to(torch.int64).to(self.device),
                torch.Tensor(next_word[: self.emb_size]).to(torch.int64).to(self.device)
                )

    def read_file(self):
        data = pd.read_csv(self.path)
        corpus = []

        for idx, (name, text) in enumerate(data[['from', 'text']].values):
            corpus.append(' ' + ''.join(ch if ord(ch) < 2000 else '' for ch in
                                        text))
        tokens = list(sorted(set(char if ord(char) < 2000 else '' for char in ''.join(corpus))))
        tokens_to_id = {token: idx for idx, token in enumerate(tokens)}
        idx_to_tokens = {v: k for k, v in tokens_to_id.items()}

        return corpus, tokens_to_id, idx_to_tokens
