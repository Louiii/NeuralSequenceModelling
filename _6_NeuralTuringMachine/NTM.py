import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, algo='LSTM'):
        super(Controller, self).__init__()
        self.input_dim, self.output_dim, self.n_layers = input_dim, output_dim, n_layers
        self.algo = algo

        # For autoregressive networks the hidden state is a learned parameter
        if algo=='LSTM':
            self.net = nn.LSTM(input_size=input_dim, hidden_size=output_dim, num_layers=n_layers)
            self.lstm_h_bias = Parameter(torch.randn(self.n_layers, 1, self.output_dim) * 0.05)
            self.lstm_c_bias = Parameter(torch.randn(self.n_layers, 1, self.output_dim) * 0.05)
        elif algo=='GRU':
            self.net = nn.GRU(input_size=input_dim, hidden_size=output_dim, num_layers=n_layers)
            self.h_bias = Parameter(torch.randn(self.n_layers, 1, self.output_dim) * 0.05)
        elif algo=='RNN':
            self.net = nn.RNN(input_size=input_dim, hidden_size=output_dim, num_layers=n_layers)
            self.h_bias = Parameter(torch.randn(self.n_layers, 1, self.output_dim) * 0.05)
        else:
            dim = max(input_dim, output_dim)
            layers = [nn.Linear(input_dim, dim), nn.Sigmoid()]
            for i in range(1, n_layers):
                layers += [nn.Linear(dim, dim), nn.Sigmoid()] if i+1<n_layers else [nn.Linear(dim, output_dim), nn.Sigmoid()]
            self.net = nn.Sequential(*layers)

        for p in self.net.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (self.input_dim + self.output_dim) ** 0.5
                nn.init.uniform_(p, -stdev, stdev)

    def create_new_state(self, batch_size):# dim: (n_layers * n_directions, batch, hidden_dim)
        if self.algo=='LSTM':
            lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
            lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
            return lstm_h, lstm_c
        if self.algo=='ANN': return None
        return self.h_bias.clone().repeat(1, batch_size, 1)

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        if self.algo=='ANN': return self.net(x).squeeze(0), None
        outp, state = self.net(x, prev_state)
        return outp.squeeze(0), state

class Memory(nn.Module):
    def __init__(self, n_locations, vec_size):
        super(Memory, self).__init__()
        self.n_locations, self.vec_size = n_locations, vec_size
        self.K = torch.cosine_similarity

        # register_buffer == model parameter, will be saved in state_dict, but not trained by the optimiser
        # memory bias allows heads to learn to initially address memory locations by content
        self.register_buffer('mem_bias', torch.Tensor(n_locations, vec_size))
        stdev = 1 / (n_locations + vec_size)**0.5
        nn.init.uniform_(self.mem_bias, -stdev, stdev)# initialise memory bias

    def reset(self, batch_size):
        """ init memory from bias, for start of sequence """
        self.batch_size = batch_size
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def content_addressing(self, key, beta):
        key = key.view(self.batch_size, 1, -1)
        return torch.softmax(beta * self.K(self.memory+1e-16, key+1e-16, dim=-1), dim=1)

    def focus_location(self, g, w_c, w_prev):
        return g * w_c + (1 - g) * w_prev

    def convolutional_shift(self, w_g, s):
        result = torch.zeros(w_g.size())
        for b in range(self.batch_size):
            t = torch.cat([w_g[b][-1:], w_g[b], w_g[b][:1]])
            result[b] = F.conv1d(t.view(1, 1, -1), s[b].view(1, 1, -1)).view(-1)
        return result

    def sharpen(self, w_tilda, gamma):
        w = w_tilda ** gamma
        return torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def write(self, w, e, a):
        self.prev_mem = self.memory
        self.memory = torch.Tensor(self.batch_size, self.n_locations, self.vec_size)
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add

    def get_w(self, key, beta, g, s, gamma, w_prev):
        w_c = self.content_addressing(key, beta)
        w_g = self.focus_location(g, w_c, w_prev)
        w_tilda = self.convolutional_shift(w_g, s)
        return self.sharpen(w_tilda, gamma)

class Head(nn.Module):
    def __init__(self, memory, controller_dim):
        super(Head, self).__init__()
        self.memory = memory
        self.n_locations, self.vec_size = memory.n_locations, memory.vec_size
        self.controller_dim = controller_dim
        self.create_new_state = lambda batch_size: torch.zeros(batch_size, self.n_locations)

    def unpack(self, mat, indices):
        indices = np.cumsum([0] + indices)
        return [mat[:, i1:i2] for i1, i2 in zip(indices[:-1], indices[1:])]

    def reset_params(self, linear):
        nn.init.xavier_uniform_(linear.weight, gain=1.4)
        nn.init.normal_(linear.bias, std=0.01)

    def address_memory(self, key, beta, g, s, gamma, w_prev):
        beta, g, s, gamma = F.softplus(beta), torch.sigmoid(g), F.softmax(s, dim=1), 1 + F.softplus(gamma)
        return self.memory.get_w(key.clone(), beta, g, s, gamma, w_prev)


class Read(Head):
    def __init__(self, memory, controller_dim):
        super(Read, self).__init__(memory, controller_dim)
        self.divisions = [self.vec_size, 1, 1, 3, 1]
        self.lin = nn.Linear(controller_dim, sum(self.divisions))
        self.reset_params(self.lin)
        self.is_read = True

    def forward(self, embeddings, w_prev):
        out = self.lin(embeddings)
        k, beta, g, s, gamma = self.unpack(out, self.divisions)
        w = self.address_memory(k, beta, g, s, gamma, w_prev)
        r = self.memory.read(w)
        return r, w

class Write(Head):
    def __init__(self, memory, controller_dim):
        super(Write, self).__init__(memory, controller_dim)
        self.divisions = [self.vec_size, 1, 1, 3, 1, self.vec_size, self.vec_size]
        self.lin = nn.Linear(controller_dim, sum(self.divisions))
        self.reset_params(self.lin)
        self.is_read = False

    def forward(self, embeddings, w_prev):
        out = self.lin(embeddings)
        k, beta, g, s, gamma, e, a = self.unpack(out, self.divisions)
        e = torch.sigmoid(e)# e should be in [0, 1]
        w = self.address_memory(k, beta, g, s, gamma, w_prev)
        self.memory.write(w, e, a)
        return w

class NTM(nn.Module):
    ''' 
    param: n_head_types 
            -> can be list of bools (read if true else write) specifying the type and 
               order of heads.
            -> or it can be an int, n; giving n alternating read head and write heads.
    '''
    def __init__(self, input_dim, controller_dim, output_dim, n_layers, 
                           n_head_types=8, n_locations=50, vec_size=10):
        super(NTM, self).__init__()

        self.input_dim, self.output_dim = input_dim, output_dim
        n_heads = n_head_types if type(n_head_types)==int else sum(n_head_types)

        self.memory = Memory(n_locations, vec_size)
        self.controller = Controller(input_dim + vec_size*n_heads, controller_dim, n_layers)

        self.heads = nn.ModuleList([])
        if type(n_head_types) is list:
            for h in n_head_types:
                self.heads += [Read(self.memory, controller_dim)] if h else [Write(self.memory, controller_dim)]
        else:
            for i in range(n_head_types):
                self.heads += [Read(self.memory, controller_dim), Write(self.memory, controller_dim)]

        self.n_locations, self.vec_size = self.memory.n_locations, self.memory.vec_size

        self.num_read_heads = 0
        self.init_r = []
        for head in [h for h in self.heads if h.is_read]:# initialise the read values to random biases
            read_bias = torch.randn(1, self.vec_size) * 0.01
            self.register_buffer("read{}_bias".format(self.num_read_heads), read_bias.data)
            self.init_r.append(read_bias)
            self.num_read_heads += 1

        self.lin = nn.Linear(self.controller.output_dim + self.num_read_heads * self.vec_size, output_dim)
        nn.init.xavier_uniform_(self.lin.weight, gain=1)
        nn.init.normal_(self.lin.bias, std=0.01)

        self.n_params = lambda : sum([p.data.view(-1).size(0) for p in self.parameters()])

    def init_sequence(self, batch_size):
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.create_new_state(batch_size)

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        return init_r, controller_state, heads_state

    def step(self, x=None, final_nonlin=True):
        if x is None: x = torch.zeros(self.batch_size, self.input_dim)
        prev_reads, prev_controller_state, prev_head_states = self.previous_state

        contr_x = torch.cat([x] + prev_reads, dim=1)
        controller_out, controller_state = self.controller(contr_x, prev_controller_state)

        reads, head_states = [], []
        for head, prev_head_state in zip(self.heads, prev_head_states):
            if head.is_read:
                r, head_state = head(controller_out, prev_head_state)
                reads.append(r)
            else:
                head_state = head(controller_out, prev_head_state)
            head_states.append(head_state)

        final_x = torch.cat([controller_out] + reads, dim=1)
        out = self.lin(final_x)
        if final_nonlin: out = torch.sigmoid(out)

        self.previous_state = (reads, controller_state, head_states)
        return out, self.previous_state

    def forward(self, x, second_section=0, final_nonlin=True):# x, y for i/o are batch-first
        if second_section > 0:
            self(x, 0, final_nonlin)
            y_hat = torch.zeros(self.batch_size, second_section, self.output_dim)
            for i in range(second_section):
                y_hat[:,i,:], _ = self.step(final_nonlin=final_nonlin)
            return y_hat
        y_hat = torch.zeros(self.batch_size, x.size(1), self.output_dim)
        for i in range(x.size(1)):
            y_hat[:,i,:], _ = self.step(x[:,i,:], final_nonlin=final_nonlin)
        return y_hat

