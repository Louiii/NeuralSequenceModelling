import torch
import torch.nn as nn
torch.manual_seed(1)

'''

Of course this is equivalent to using nn.RNN(.., num_layers=n, ..)
and nn.RNN can be replaced with nn.GRU or nn.LSTM for example.

'''

class StackedRNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, rollout):
        super(StackedRNN, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = in_dim, h_dim, out_dim
        self.n_layers = n_layers

        self.layers = [nn.RNN(in_dim, h_dim, num_layers=1, batch_first=True)]
        for _ in range(n_layers-1):
            self.layers += [nn.RNN(h_dim, h_dim, num_layers=1, batch_first=True)]

        self.out = nn.Linear(h_dim, out_dim)

    def forward(self, x, hidden_state):
        '''
        x:       {batch,    time_step, input_size}
        h_state: {n_layers, batch,     hidden_size}
        r_out:   {batch,    time_step, hidden_size}
        '''
        h_list = list(torch.split(hidden_state, [1]*self.n_layers))
        inp = x
        for i in range(self.n_layers):
            rnn = self.layers[i]
            inp, h_list[i] = rnn(inp, h_list[i])
        hidden_state = torch.cat(h_list, dim=0)
        last_h = h_list[-1]
        
        y_hat = self.out(inp)
        return y_hat, hidden_state

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.n_layers, batch_size, self.h_dim)

    def break_connection(self, hidden_state):
        return hidden_state.detach()

    def parameters(self):
        p = []
        for rnn in self.layers:
            p += rnn.parameters()
        return p + [self.out.weight, self.out.bias]