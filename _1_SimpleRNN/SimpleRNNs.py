import torch
import torch.nn as nn
torch.manual_seed(1)

M = lambda n, m, r : nn.Parameter(torch.randn(n, m) * r)
V = lambda n : nn.Parameter(torch.zeros(n))

'''

Regression RNNs

'''

class LowLevelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(LowLevelRNN, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.Wx_X = M(input_size, hidden_size, 0.01)
        self.Wh_h = M(hidden_size, hidden_size, 0.01)
        self.Wh_y = M(hidden_size, output_size, 0.01)
        self.bh = V(hidden_size)
        self.by = V(output_size)

    def step(self, input, hidden):
        h = torch.tanh(torch.matmul(input, self.Wx_X) +
                       torch.matmul(hidden, self.Wh_h) +
                       self.bh)
        y = torch.matmul(h, self.Wh_y) + self.by
        return y, h

    def forward(self, x, h):
        output = torch.zeros(self.rollout, self.y_dim)
        for i in range(self.rollout):
            output[i], h = self.step(x[0, i], h)
        return output, h

    def init_hidden(self):
        return torch.zeros(self.h_dim)

    def break_connection(self, hidden):
        return hidden.detach()

    def __repr__(self):
        return "RNN(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)

class MidLevelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(MidLevelRNN, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.xh_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.hy = nn.Linear(hidden_size, output_size)
        self.h_dim = hidden_size

    def step(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        hidden = self.xh_h(combined)
        hidden = torch.tanh(hidden)
        output = self.hy(hidden)
        return output, hidden

    def forward(self, x, h):
        output = torch.zeros(self.rollout, self.y_dim)
        for i in range(self.rollout):
            output[i], h = self.step(x[0, i], h)
        return output, h

    def init_hidden(self):
        return torch.zeros(self.h_dim)

    def break_connection(self, hidden):
        return hidden.detach()

class HighLevelRNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, rollout):
        super(HighLevelRNN, self).__init__()
        self.h_dim, self.n_layers = h_dim, n_layers

        self.rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,# x, y have batch size as first dim. {batch, time_step, input_size}
        )
        self.out = nn.Linear(h_dim, out_dim)

    def forward(self, x, h_state):
        '''
        x:       {batch,    time_step, input_size}
        h_state: {n_layers, batch,     hidden_size}
        r_out:   {batch,    time_step, hidden_size}
        '''
        r_out, h_state = self.rnn(x, h_state)
        # r_out is the hidden states at each time step
        # h_state is the final hidden state
        y_hat = self.out(r_out)# make predictions y from h at each time step
        return y_hat, h_state

    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.h_dim)

    def break_connection(self, hidden):
        return hidden.detach()

'''

Classification RNNs

'''

class LowLevelRNN_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(LowLevelRNN_classifier, self).__init__()
        self.h_dim, self.n_layers, self.rollout = hidden_size, n_layers, rollout

        self.embedding = nn.Embedding(input_size, input_size)# Change this!

        self.rnn = LowLevelRNN(input_size, hidden_size, output_size, n_layers, rollout)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        embedding = torch.unsqueeze(torch.squeeze(embedding, 1), 0)
        output, hidden_state = self.rnn(embedding, hidden_state)
        return output, hidden_state.detach()

    def init_hidden(self, length=None):
        self.rnn.rollout = self.rollout if length==None else length
        return self.rnn.init_hidden()#torch.zeros(self.n_layers, 1, self.h_dim)

    def break_connection(self, hidden):
        return hidden.detach()

class HighLevelRNN_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(HighLevelRNN_classifier, self).__init__()
        self.h_dim, self.n_layers, self.rollout = hidden_size, n_layers, rollout

        self.embedding = nn.Embedding(input_size, input_size)

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,# x, y have batch size as first dim. {batch, time_step, input_size}
        )
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, hidden_state.detach()

    def init_hidden(self, length=None):
        if length==None:
            return torch.zeros(self.n_layers, self.rollout, self.h_dim)
        return torch.zeros(self.n_layers, length, self.h_dim)

    def break_connection(self, hidden):
        return hidden.detach()
