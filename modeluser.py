import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
# import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, model, batch_size, args, dataset, token_to_idx, idx_to_token, num_epoch=1, eval=False):
        self.model = model
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.args = args
        self.dataset = dataset
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
        self.eval = eval
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train(self, num_epoch=None, eval=None):
        if num_epoch:
            self.num_epoch = num_epoch

        if eval:
            self.eval = eval

        self.model.to(self.device)
        self.model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=5,
            verbose=True,
            factor=0.5
        )

        loss_global = []
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, drop_last=True)

        for epoch in range(self.num_epoch):
            state_h, state_c = self.model.init_state(self.batch_size)
            state_h = state_h.to(self.device)
            state_c = state_c.to(self.device)

            loss_avg = []

            for batch, (x, y) in enumerate(dataloader):

                optimizer.zero_grad()

                x = x.view(self.batch_size, self.args['ebm_size'], 1)
                y = y.view(self.batch_size, self.args['ebm_size'], 1)

                x = x.permute(1, 0, 2).to(self.device)
                y = y.permute(1, 0, 2).to(self.device)

                y_pred, (state_h, state_c) = self.model(x, (state_h, state_c))

                loss = criterion(y_pred.permute(1, 2, 0), y.squeeze(-1).permute(1, 0))


                state_h = state_h.detach()
                state_c = state_c.detach()

                loss.backward()
                optimizer.step()
                loss_avg.append(loss.item())

            loss_epoch = sum(loss_avg) / len(loss_avg)
            loss_global.append(loss_epoch)
            print({'epoch': epoch, 'epoch loss': loss.item()})
            if eval:
                print(self.evaluate())
                self.model.train()

            if (epoch + 1) % 100 == 0:
                torch.save(self.model.state_dict(), f'homeland{epoch + 1}')

    def evaluate(self, start_seq=' ', prediction_len=128, temp=0.3):

        self.model.eval()

        state_h, state_c = self.model.init_state()
        coded_word = [self.token_to_idx[char] for char in start_seq]  # +\

        predicted_text = start_seq

        coded_word = torch.Tensor(coded_word[: self.args['ebm_size']]).to(torch.int64).to(self.device)

        inp = coded_word[-1].view(-1, 1)

        for i in range(prediction_len - len(start_seq) - 1):
            y_pred, (state_h, state_c) = self.model(inp, (state_h, state_c))

            y_pred = y_pred.cpu().data.view(-1)
            p_next = F.softmax(y_pred / temp, dim=-1).detach().cpu().data.numpy()

            next_ix = np.random.choice(a=self.args['input_size'], p=p_next)

            inp = torch.LongTensor([next_ix]).view(-1, 1).to(self.device)

            predicted_text += self.idx_to_token[next_ix]

        return predicted_text
