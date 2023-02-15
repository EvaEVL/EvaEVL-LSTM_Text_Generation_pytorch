import torch
from torch import nn

class Model(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layer=1):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layer
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, prev_state):

        x = self.embedding(x).squeeze(2)
        out, state = self.lstm(x, prev_state)
        x = self.fc(out)
        return x, state

    def init_state(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device))
