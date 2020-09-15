import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class DRNN(nn.Module):
    ''' Uses batch first encoding, uses stacked LSTMs but can easily be adapted to GRUs '''
    def __init__(self, in_dim, h_dim, n_layers, dropout=0):
        super(DRNN, self).__init__()

        self.h_dim = h_dim
        self.dilations = [2 ** i for i in range(n_layers)]

        self.cells = nn.ModuleList([nn.LSTM(in_dim, h_dim, dropout=dropout, batch_first=True)])
        for _ in range(n_layers-1):
            self.cells.append(nn.LSTM(h_dim, h_dim, dropout=dropout, batch_first=True))

    def forward(self, X, hidden=None):
        '''
        x:       {batch,    time_step, input_size}
        h_state: {n_layers, batch,     hidden_size}
        out:     {batch,    time_step, hidden_size}
        '''
        batch_size, n_steps, _ = X.size()

        h_states = X
        new_h = []
        for i, (rnn, dilation) in enumerate(zip(self.cells, self.dilations)):
            h_states = self.pad_input(h_states, n_steps, dilation)

            # Dilated sequence (dil=3): [0,...,6] -> [[0,3],[1,4],[2,5],[3,6]] 
            dilated_h = torch.cat([h_states[:, j::dilation, :] for j in range(dilation)], 0)

            if hidden is None:
                h = self.init_single_hidden(batch_size * dilation)
            else: 
                h = hidden[i]
            dilated_hs, h = rnn(dilated_h, h)
            new_h.append(h)

            splitted_hs = self.split_outputs(dilated_hs, dilation)
            h_states = splitted_hs[:, :n_steps, :]# get rid of the padding

        return h_states, new_h

    def split_outputs(self, dilated_hs, dilation):
        bs = dilated_hs.size(0) // dilation# original batch_size = the current batch size // dilation
        blocks = [dilated_hs[i*bs:(i+1)*bs,:,:] for i in range(dilation)]
        interleaved = torch.stack((blocks)).transpose(1, 2)#.contiguous()
        return interleaved.view(bs, dilated_hs.size(1)*dilation, dilated_hs.size(2))

    def pad_input(self, X, n_steps, dilation):
        if n_steps%dilation != 0:# pad
            dilated_steps = n_steps // dilation + 1
            z = torch.zeros(X.size(0), dilated_steps*dilation-X.size(1), X.size(2))
            # if use_cuda: z = z.cuda()
            return torch.cat((X, z), dim=1)
        return X# no padding needed

    def init_single_hidden(self, batch_size):
        a = torch.zeros(1, batch_size, self.h_dim)
        c = torch.zeros(1, batch_size, self.h_dim)
        if use_cuda: a, c = a.cuda(), c.cuda()
        return (a, c)

    def init_hidden(self, batch_size):
        hidden = []
        for dilation in self.dilations:
            hidden.append(self.init_single_hidden(batch_size * dilation))
        return hidden

class DilatedRNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, rollout, dropout=0):
        super(DilatedRNN, self).__init__()
        self.rnn = DRNN(in_dim, h_dim, n_layers, dropout)
        self.out = nn.Linear(h_dim, out_dim)

    def forward(self, X, h):
        hs, h = self.rnn(X, h)
        y_hat = self.out(hs)
        return y_hat, h

    def init_hidden(self, batch_size=1):
        return self.rnn.init_hidden(batch_size)

    def break_connection(self, h):
        lstm_break = lambda a, c: (a.detach(), c.detach())
        return [lstm_break(*hd) for hd in h]

    def parameters(self):
        return list(self.rnn.parameters()) + [self.out.bias, self.out.weight]
