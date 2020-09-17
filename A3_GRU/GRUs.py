import torch
import torch.nn as nn
torch.manual_seed(1)

M = lambda n, m, r : nn.Parameter(torch.randn(n, m) * r)
V = lambda n : nn.Parameter(torch.zeros(n))

'''

Regression GRUs

'''

class LowLevelGRU_simplified(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(LowLevelGRU_simplified, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.Wc = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wu = M(input_size+hidden_size, hidden_size, 0.01)
        self.bc = V(hidden_size)
        self.bu = V(hidden_size)

        # self.out = nn.Linear(hidden_size, output_size)
        self.Wh_y = M(hidden_size, output_size, 0.01)
        self.by = V(output_size)

    def step(self, input, hidden):
        '''
        implements the following update equation:

        c_{t} = L_u .* c^_{t} + (1 .- L_u) .* c_{t-1}

        where,

        c^_{t} = tanh( W_c * [c_{t-1}, x_{t}] + b_c )
        L_u = sig( W_u * [c_{t-1}, x_{t}] + b_u )
        '''
        xc = torch.cat((input, hidden), 0)
        Lu = torch.sigmoid(torch.matmul(xc, self.Wu) + self.bu)

        c_prev = hidden
        c_tilda = torch.tanh(torch.matmul(xc, self.Wc) + self.bc)
        hidden = torch.mul(c_tilda, Lu) + torch.mul(c_prev, 1-Lu)

        y = torch.matmul(hidden, self.Wh_y) + self.by
        return y, hidden

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
        return "(simple) GRU(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)


class LowLevelGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(LowLevelGRU, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.Wc = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wu = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wr = M(input_size+hidden_size, hidden_size, 0.01)
        self.bc = V(hidden_size)
        self.bu = V(hidden_size)
        self.br = V(hidden_size)

        # self.out = nn.Linear(hidden_size, output_size)
        self.Wh_y = M(hidden_size, output_size, 0.01)
        self.by = V(output_size)

    def step(self, input, hidden):
        '''
        implements the following update equation:

        c_{t} = L_u .* c^_{t} + (1 .- L_u) .* c_{t-1}

        where,

        c^_{t} = tanh( W_c * [L_r .* c_{t-1}, x_{t}] + b_c )
        L_u = sig( W_u * [c_{t-1}, x_{t}] + b_u )
        L_r = sig( W_r * [c_{t-1}, x_{t}] + b_r )
        '''
        xc = torch.cat((input, hidden), 0)
        Lu = torch.sigmoid(torch.matmul(xc, self.Wu) + self.bu)
        Lr = torch.sigmoid(torch.matmul(xc, self.Wr) + self.br)

        c_prev = hidden
        xr = torch.cat((input, torch.mul(c_prev, Lr)), 0)
        c_tilda = torch.tanh(torch.matmul(xr, self.Wc) + self.bc)
        hidden = torch.mul(c_tilda, Lu) + torch.mul(c_prev, 1-Lu)

        y = torch.matmul(hidden, self.Wh_y) + self.by
        return y, hidden

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
        return "GRU(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)

class LowLevelGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(LowLevelGRU, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.Wc = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wu = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wr = M(input_size+hidden_size, hidden_size, 0.01)
        self.bc = V(hidden_size)
        self.bu = V(hidden_size)
        self.br = V(hidden_size)

        # self.out = nn.Linear(hidden_size, output_size)
        self.Wh_y = M(hidden_size, output_size, 0.01)
        self.by = V(output_size)

    def step(self, input, hidden):
        '''
        implements the following update equation:

        c_{t} = L_u .* c^_{t} + (1 .- L_u) .* c_{t-1}

        where,

        c^_{t} = tanh( W_c * [L_r .* c_{t-1}, x_{t}] + b_c )
        L_u = sig( W_u * [c_{t-1}, x_{t}] + b_u )
        L_r = sig( W_r * [c_{t-1}, x_{t}] + b_r )
        '''
        xc = torch.cat((input, hidden), 0)
        Lu = torch.sigmoid(torch.matmul(xc, self.Wu) + self.bu)
        Lr = torch.sigmoid(torch.matmul(xc, self.Wr) + self.br)

        c_prev = hidden
        xr = torch.cat((input, torch.mul(c_prev, Lr)), 0)
        c_tilda = torch.tanh(torch.matmul(xr, self.Wc) + self.bc)
        hidden = torch.mul(c_tilda, Lu) + torch.mul(c_prev, 1-Lu)

        y = torch.matmul(hidden, self.Wh_y) + self.by
        return y, hidden

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
        return "GRU(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)
#
# class MidLevelGRU(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
#         super(MidLevelGRU, self).__init__()
#         self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
#         self.rollout = rollout
#
#         self.cTildaLayer = nn.Linear(input_size+hidden_size, hidden_size)
#         self.updateGate = nn.Linear(input_size+hidden_size, hidden_size)
#         self.relevanceGate = nn.Linear(input_size+hidden_size, hidden_size)
#
#         self.out = nn.Linear(hidden_size, output_size)
#
#     def step(self, input, hidden):
#         '''
#         implements the following update equation:
#
#         c_{t} = L_u .* c^_{t} + (1 .- L_u) .* c_{t-1}
#
#         where,
#
#         c^_{t} = tanh( W_c * [L_r .* c_{t-1}, x_{t}] + b_c )
#         L_u = sig( W_u * [c_{t-1}, x_{t}] + b_u )
#         L_r = sig( W_r * [c_{t-1}, x_{t}] + b_r )
#         '''
#         xc = torch.cat((input, hidden), 0)
#         Lu = torch.sigmoid( self.updateGate(xc) )
#         Lr = torch.sigmoid( self.relevanceGate(xc) )
#
#         c_prev = hidden
#         xr = torch.cat((input, torch.mul(c_prev, Lr)), 0)
#         c_tilda = torch.tanh( self.cTildaLayer(xr) )
#         hidden = torch.mul(c_tilda, Lu) + torch.mul(c_prev, 1-Lu)
#
#         y = self.out(hidden)
#         return y, hidden
#
#     def forward(self, x, h):
#         output = torch.zeros(self.rollout, self.y_dim)
#         for i in range(self.rollout):
#             output[i], h = self.step(x[0, i], h)
#         return output, h
#
#     def init_hidden(self):
#         return torch.zeros(self.h_dim)
#
#     def break_connection(self, hidden):
#         return hidden.detach()
#
#     def __repr__(self):
#         return "GRU(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)

class HighLevelGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(HighLevelGRU, self).__init__()

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        self.outLayer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        yhat = self.outLayer(out)
        h = self.break_connection(h)
        return yhat, h

    def init_hidden(self):
        return None

    def break_connection(self, hidden):
        return hidden.detach()

    def __repr__(self):
        return self.gru.__repr__()


'''

Classification GRU

'''

class HighLevelGRU_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(HighLevelGRU_classifier, self).__init__()
        self.h_dim, self.n_layers, self.rollout = hidden_size, n_layers, rollout

        self.embedding = nn.Embedding(input_size, input_size)

        self.rnn = nn.GRU(
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
        return None

    def break_connection(self, hidden):
        return hidden.detach()
