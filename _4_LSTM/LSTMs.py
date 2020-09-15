import torch
import torch.nn as nn
torch.manual_seed(1)

M = lambda n, m, r : nn.Parameter(torch.randn(n, m) * r)
V = lambda n : nn.Parameter(torch.zeros(n))

'''

Regression LSTMs

'''

class LowLevelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(LowLevelLSTM, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.Wc = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wu = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wf = M(input_size+hidden_size, hidden_size, 0.01)
        self.Wo = M(input_size+hidden_size, hidden_size, 0.01)
        self.bc = V(hidden_size)
        self.bu = V(hidden_size)
        self.bf = V(hidden_size)
        self.bo = V(hidden_size)

        # self.out = nn.Linear(hidden_size, output_size)
        self.Wh_y = M(hidden_size, output_size, 0.01)
        self.by = V(output_size)

    def step(self, input, hidden):
        '''
        implements the following update equation:

        c_{t} = L_u .* c^_{t} + L_f .* c_{t-1}
        a_{t} = L_o .* tanh( c_{t} )

        where,

        c^_{t} = tanh( W_c * [a_{t-1}, x_{t}] + b_c )
        L_u = sig( W_u * [a_{t-1}, x_{t}] + b_u )
        L_f = sig( W_f * [a_{t-1}, x_{t}] + b_f )
        L_o = sig( W_o * [a_{t-1}, x_{t}] + b_o )
        '''
        (a, c) = hidden
        ax = torch.cat((a, input), 0)

        c_tilda = torch.tanh( torch.matmul(ax, self.Wc) + self.bc )
        Lu = torch.sigmoid( torch.matmul(ax, self.Wu) + self.bu )
        Lf = torch.sigmoid( torch.matmul(ax, self.Wf) + self.bf )
        Lo = torch.sigmoid( torch.matmul(ax, self.Wo) + self.bo )

        c = torch.mul(c_tilda, Lu) + torch.mul(c, Lf)
        a = torch.mul(Lo, torch.tanh(c))

        y = torch.matmul(a, self.Wh_y) + self.by
        return y, (a, c)

    def forward(self, x, h):
        output = torch.zeros(self.rollout, self.y_dim)
        for i in range(self.rollout):
            output[i], h = self.step(x[0, i], h)
        return output, h

    def init_hidden(self):
        return (torch.zeros(self.h_dim), torch.zeros(self.h_dim))

    def break_connection(self, hidden):
        (a, c) = hidden
        return (a.detach(), c.detach())

    def __repr__(self):
        return "LSTM(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)

# class MidLevelLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
#         super(MidLevelLSTM, self).__init__()
#         self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
#         self.rollout = rollout
#
#         self.cTildaLayer = nn.Linear(input_size+hidden_size, hidden_size)
#         self.updateGate = nn.Linear(input_size+hidden_size, hidden_size)
#         self.forgetGate = nn.Linear(input_size+hidden_size, hidden_size)
#         self.outputGate = nn.Linear(input_size+hidden_size, hidden_size)
#
#         self.outLayer = nn.Linear(hidden_size, output_size)
#
#     def step(self, input, hidden):
#         '''
#         implements the following update equation:
#
#         c_{t} = L_u .* c^_{t} + L_f .* c_{t-1}
#         a_{t} = L_o .* tanh( c_{t} )
#
#         where,
#
#         c^_{t} = tanh( W_c * [a_{t-1}, x_{t}] + b_c )
#         L_u = sig( W_u * [a_{t-1}, x_{t}] + b_u )
#         L_f = sig( W_f * [a_{t-1}, x_{t}] + b_f )
#         L_o = sig( W_o * [a_{t-1}, x_{t}] + b_o )
#         '''
#         (a, c) = hidden
#         ax = torch.cat((a, input), 0)
#
#         c_tilda = torch.tanh( self.cTildaLayer(ax) )
#         Lu = torch.sigmoid( self.updateGate(ax) )
#         Lf = torch.sigmoid( self.forgetGate(ax) )
#         Lo = torch.sigmoid( self.outputGate(ax) )
#
#         c = torch.mul(c_tilda, Lu) + torch.mul(c, Lf)
#         a = torch.mul(Lo, torch.tanh(c))
#
#         y = self.outLayer(a)
#         return y, (a, c)
#
#     def forward(self, x, h):
#         output = torch.zeros(self.rollout, self.y_dim)
#         for i in range(self.rollout):
#             output[i], h = self.step(x[0, i], h)
#         return output, h
#
#     def init_hidden(self):
#         return (torch.zeros(self.h_dim), torch.zeros(self.h_dim))
#
#     def break_connection(self, hidden):
#         (a, c) = hidden
#         return (a.detach(), c.detach())
#
#     def __repr__(self):
#         return "LSTM(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)

class MidLevelLSTM_peephole(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(MidLevelLSTM_peephole, self).__init__()
        self.x_dim, self.h_dim, self.y_dim = input_size, hidden_size, output_size
        self.rollout = rollout

        self.cTildaLayer = nn.Linear(input_size+hidden_size, hidden_size)
        self.updateGate = nn.Linear(input_size+2*hidden_size, hidden_size)
        self.forgetGate = nn.Linear(input_size+2*hidden_size, hidden_size)
        self.outputGate = nn.Linear(input_size+2*hidden_size, hidden_size)

        self.outLayer = nn.Linear(hidden_size, output_size)

    def step(self, input, hidden):
        '''
        implements the following update equation:

        c_{t} = L_u .* c^_{t} + L_f .* c_{t-1}
        a_{t} = L_o .* tanh( c_{t} )

        where,

        c^_{t} = tanh( W_c * [a_{t-1}, x_{t}] + b_c )
        L_u = sig( W_u * [a_{t-1}, x_{t}, c_{t-1}] + b_u )
        L_f = sig( W_f * [a_{t-1}, x_{t}, c_{t-1}] + b_f )
        L_o = sig( W_o * [a_{t-1}, x_{t}, c_{t-1}] + b_o )
        '''
        (a, c) = hidden
        ax = torch.cat((a, input), 0)
        axc = torch.cat((a, input, c), 0)

        c_tilda = torch.tanh( self.cTildaLayer(ax) )
        Lu = torch.sigmoid( self.updateGate(axc) )
        Lf = torch.sigmoid( self.forgetGate(axc) )
        Lo = torch.sigmoid( self.outputGate(axc) )

        c = torch.mul(c_tilda, Lu) + torch.mul(c, Lf)
        a = torch.mul(Lo, torch.tanh(c))

        y = self.outLayer(a)
        return y, (a, c)

    def forward(self, x, h):
        output = torch.zeros(self.rollout, self.y_dim)
        for i in range(self.rollout):
            output[i], h = self.step(x[0, i], h)
        return output, h

    def init_hidden(self):
        return (torch.zeros(self.h_dim), torch.zeros(self.h_dim))

    def break_connection(self, hidden):
        (a, c) = hidden
        return (a.detach(), c.detach())

    def __repr__(self):
        return "LSTM(\n\tx_dim=%d\n\th_dim=%d\n\ty_dim=%d\n)"%(self.x_dim, self.h_dim, self.y_dim)

class HighLevelLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(HighLevelLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        self.outLayer = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        hs, (hn, cn) = self.lstm(x, h)
        yhat = self.outLayer(hs)
        h = (hn.detach(), cn.detach())
        return yhat, h

    def init_hidden(self):
        return None

    def break_connection(self, hidden):
        (a, c) = hidden
        return (a.detach(), c.detach())

    def __repr__(self):
        return self.lstm.__repr__()


'''

Classification LSTM

'''

class HighLevelLSTM_classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, rollout):
        super(HighLevelLSTM_classifier, self).__init__()
        self.h_dim, self.n_layers, self.rollout = hidden_size, n_layers, rollout

        self.embedding = nn.Embedding(input_size, input_size)

        self.rnn = nn.LSTM(
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
        return output, self.break_connection(hidden_state)

    def init_hidden(self, length=None):
        return None

    def break_connection(self, hidden):
        return hidden[0].detach(), hidden[1].detach()
