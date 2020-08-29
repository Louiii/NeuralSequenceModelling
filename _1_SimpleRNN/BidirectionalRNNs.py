import torch
import torch.nn as nn
torch.manual_seed(1)

'''

Bidirectional RNNs are used when we know the whole sequence as input
and try to predict a sequence or vector output, so bidirectional RNNs 
are used as encoders (e.g. seq2seq, seq2vec).

For this reason I only output the hidden states. The second part of 
the model must be implemented depending on the problem at hand.

'''

class BRNN(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers):
        super(BRNN, self).__init__()
        self.h_dim, self.n_layers = h_dim, n_layers

        self.frwd_rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,# x, y have batch size as first dim. {batch, time_step, input_size}
        )
        self.bkwd_rnn = nn.RNN(
            input_size=in_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,# x, y have batch size as first dim. {batch, time_step, input_size}
        )

    def core_forward(self, x, h_state):
        '''
        x:        {batch,    time_step, input_size}
        fh_state: {n_layers, batch,     hidden_size}
        f_out:    {batch,    time_step, hidden_size}
        '''
        f_h, b_h = torch.split(h_state, self.h_dim, dim=2)
        f_out, fh_state = self.frwd_rnn(x, f_h)

        xr = torch.flip(xr, dims=[1])
        b_out, bh_state = self.bkwd_rnn(xr, b_h)

        h_state = torch.cat((fh_state, bh_state), dim=2)
        r_out = torch.cat((f_out, b_out), dim=2)
        return r_out, h_state

    def forward(self, x, h_state):
        return self.core_forward(x, h_state)

    def init_hidden(self, n_batches=1):
        return torch.zeros(self.n_layers, n_batches, 2 * self.h_dim)

    def break_connection(self, hidden):
        return hidden.detach()

class BRNN_classifier(BRNN):
    def __init__(self, in_dim, h_dim, n_layers):
        super(BRNN_classifier, self).__init__(in_dim, h_dim, n_layers)
        self.embedding = nn.Embedding(in_dim, in_dim)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        return self.core_forward(embedding, hidden_state)
