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
        batch_size = X.size(0)
        if hidden is None: 
            hidden = self.init_hidden(len(self.dilations), batch_size, self.h_dim)

        h_states = X
        n_steps = h_states.size(1)
        outputs = []
        for i, (rnn, dilation) in enumerate(zip(self.cells, self.dilations)):
            h_states, _ = self.pad_input(h_states, n_steps, dilation)

            # Dilated sequence (dil=3): [0,...,6] -> [[0,3],[1,4],[2,5],[3,6]] 
            dilated_h = torch.cat([h_states[:, j::dilation, :] for j in range(dilation)], 0)

            dilated_hs, hidden[i] = rnn(dilated_h, hidden[i])

            print("\nhidden states size", dilated_hs.size(), "\nindividual h_"+str(i+1)+" size", hidden[i][0].size())
            
            splitted_hs = self.split_outputs(dilated_hs, dilation)
            h_states = splitted_hs[:, :n_steps, :]# get rid of the padding

            outputs.append(h_states[:,-dilation:,:])
        return h_states, outputs

    def split_outputs(self, dilated_hs, dilation):
        batchsize = dilated_hs.size(0) // dilation# original batch_size = the current batch size // dilation
        blocks = [dilated_hs[i * batchsize: (i + 1) * batchsize, :, :] for i in range(dilation)]
        interleaved = torch.stack((blocks)).transpose(1, 2).contiguous()
        return interleaved.view(batchsize, dilated_hs.size(1) * dilation, dilated_hs.size(2) )

    def pad_input(self, X, n_steps, dilation):
        is_even = (n_steps % dilation) == 0

        if not is_even:# pad
            dilated_steps = n_steps // dilation + 1
            zeros_ = torch.zeros(X.size(0),
                                 dilated_steps * dilation - X.size(1),
                                 X.size(2))
            if use_cuda: zeros_ = zeros_.cuda()
            return torch.cat((X, zeros_), dim=1), dilated_steps
        return X, n_steps // dilation# no padding needed

    def init_single_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(1, batch_size, hidden_dim)
        memory = torch.zeros(1, batch_size, hidden_dim)
        if use_cuda: hidden, memory = hidden.cuda(), memory.cuda()
        return (hidden, memory)

    def init_hidden(self, n_layers, batch_size, hidden_dim):
        hidden = []
        for dilation in self.dilations:
            hidden.append(self.init_single_hidden(batch_size * dilation, hidden_dim))
        return hidden

class DilatedRNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        super(DilatedRNN, self).__init__()
        self.rnn = DRNN(in_dim, h_dim, n_layers, dropout)
        self.out = nn.Linear(h_dim, out_dim)

    def forward(self, X, h):
        hs, h = self.rnn(X, h)
        y_hat = self.out(hs)
        return y_hat, h

    def parameters(self):
        return self.rnn.parameters() + [self.out.bias, self.out.weight]


# n_input = 9
# h_dim = 13
# n_layers = 5
# batch_size = 6
# seq_size = 11

# model = DRNN(n_input, h_dim, n_layers)
# h = None

# x1 = torch.randn(batch_size, seq_size, n_input)
# x2 = torch.randn(batch_size, seq_size, n_input)

# out, hidden = model(x1, h)

# print("Out size", out.size())