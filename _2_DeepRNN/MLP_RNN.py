import torch
import torch.nn as nn

torch.manual_seed(1)


class MLP_RNN(nn.Module):
    '''
    RNN with variable topology:

    x->X is a feedforward network, dimensions: in_dims
    h->H is a feedforward network, dimensions: h_dims
    [X|H]->h is addition plus a bias, followed by a non-linearity
    '''
    def __init__(self, in_dims, h_dims):
        super(MLP_RNN, self).__init__()
        self.x_dims, self.h_dims = in_dims, h_dims

        self.i2x = nn.ModuleList([nn.Linear(i, j) for i,j in zip(in_dims[:-1], in_dims[1:])])
        self.xh2h = nn.ModuleList([nn.Linear(i, j) for i,j in zip(h_dims[:-1], h_dims[1:])])

    def feedforward(self, x, lins, no_final_nlin=True):
        next_layer = x
        for i in range(len(lins)):
            a = lins[i](next_layer)
            next_layer = a if i==len(lins)-1 or no_final_nlin else torch.tanh(a)
        return next_layer

    def step(self, x, hidden):
        hx = torch.cat((hidden, x), -1)
        return self.feedforward(hx, self.xh2h)

    def forward(self, x, h):
        rollout = x.size(1)
        x = self.feedforward(x, self.i2x)
        h = h.squeeze(0)
        hs = torch.zeros(rollout, self.h_dims[-1])
        for i in range(rollout):
            h = self.step(x[:, i], h)
            hs[i] = h
        return hs.unsqueeze(0), h.unsqueeze(0)

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.h_dims[-1])

    # def parameters(self):
    #     self.ps = self.i2x + self.xh2h
    #     return list(map(lambda x:x.weight, self.ps))+list(map(lambda x:x.bias, self.ps))

    def __repr__(self):
        return "RNN(\n\tx_dims = %s\n\th_dims = %s\n)"%(str(self.x_dims), str(self.h_dims))


class FullMLP_RNN(nn.Module):
    '''
    h->y is a feedforward network, dimensions: out_dims
    '''
    def __init__(self, in_dims, h_dims, out_dims):
        super(FullMLP_RNN, self).__init__()
        self.x_dims, self.h_dims, self.y_dims = in_dims, h_dims, out_dims
        self.rnn = MLP_RNN(in_dims, h_dims)
        self.h2o = nn.ModuleList([nn.Linear(i, j) for i,j in zip(out_dims[:-1], out_dims[1:])])

    def forward(self, x, h):
        hs, h = self.rnn(x, h)
        output = self.rnn.feedforward(hs, self.h2o)
        return output, h

    def init_hidden(self, batch_size=1):
        return self.rnn.init_hidden(batch_size)

    # def parameters(self):
    #     return self.rnn.parameters() + [l.weight for l in self.h2o] + [l.bias for l in self.h2o]


"""
In general, there are two fundamental ways that one could use skip 
connections through different non-sequential layers: 

a) addition as in residual architectures, 
b) concatenation as in densely connected architectures.

(a) ResNet: skip connections via addition 
The core idea is to backpropagate through the identity function, 
by just using a vector addition. Then the gradient would simply be 
multiplied by one and its value will be maintained in the earlier 
layers. This is the main idea behind Residual Networks (ResNets): 
they stack these skip residual blocks together. We use an identity 
function to preserve the gradient. 

(b) DenseNet: skip connections via concatenation 
As stated, for many dense prediction problems, there is low-level 
information shared between the input and output, and it would be 
desirable to pass this information directly across the net. The 
alternative way that you can achieve skip connections is by 
concatenation of previous feature maps. The most famous deep 
learning architecture is DenseNet. 
"""


"""
Let the hidden state be split in two: H = [H1; H2]

A standard RNN maps [X; H1; H2] -> [H1; H2]

Let the network mapping [..] -> H1 be called f1, and the network mapping 
[..] -> H2 be called f2.

If f1 is some function of just H1 and X, and f2 is some function of just 
H2 and X, then this is equivalent to having two RNNs. 

Now assume f1 is deeper than f2. For example consider f2 as a single layer, 
this can be considered a skip connection.

To simpify things further assume f1 and f2 are a function of X, and the 
hidden state they map to (and some of the other hidden states).

[--- H1 ---][--- X ---][--- H2 ---]
     |       /       \      |
     v      v         v     v
[--- H1 ---]           [--- H2 ---]

Finally, let p1 be a number specifying how many units from H1 and p2 from H2. 
The RNN I define has f1 : [H1; H2[:p2]; X] -> H1 and f2 : [H2; H1[:p1]; X] -> H2.
"""

class MySkipRNN(nn.Module):
    def __init__(self, in_dim, out_dim, h1_dim, h2_dim, p1=0, p2=0, depth_h1=2):
        super(MySkipRNN, self).__init__()
        self.h1_dim, self.h2_dim = h1_dim, h2_dim
        self.p1, self.p2 = p1, p2

        self.deep_rnn = MLP_RNN([in_dim+p2], [in_dim+p2+h1_dim]+[h1_dim]*depth_h1)
        self.shallow_rnn = nn.RNN(input_size=in_dim,hidden_size=h2_dim,num_layers=1,batch_first=True)

        self.out = nn.Linear(h1_dim + h2_dim, out_dim)

    def forward(self, x, hidden_state):
        '''
        x:        {batch,    time_step, input_size}
        h_state:  {n_layers, batch,     hidden_size}
        f_out:    {batch,    time_step, hidden_size}
        '''
        h1, h2 = torch.split(hidden_state, [self.h1_dim, self.h2_dim], dim=-1)

        # compute the shallow first [X; H2] -> H2'
        h2s, h2 = self.shallow_rnn(x, h2) # just a standard RNN

        h2p2 = h2s[:,:,:self.p2]

        x1 = torch.cat((x, h2p2), -1)
        h1s, h1 = self.deep_rnn(x1, h1)

        hidden_state = torch.cat((h1, h2), dim=-1)
        hs = torch.cat((h1s, h2s), dim=-1)

        y_hat = self.out( hs )
        return y_hat, hidden_state

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.h1_dim + self.h2_dim)

    def parameters(self):
        return list(self.shallow_rnn.parameters()) + list(self.deep_rnn.parameters()) + [self.out.weight, self.out.bias]


class ResNetBlock(nn.Module):
    def __init__(self, in_dim, actv='relu'):
        super(ResNetBlock, self).__init__()

        self.layer1 = nn.Linear(in_dim, in_dim)
        self.layer2 = nn.Linear(in_dim, in_dim)
        self.activation = torch.tanh if actv=='tanh' else torch.relu

    def forward(self, x):
        y = self.layer2(self.activation(self.layer1(x)))
        return self.activation(x + y)

class ResRNN(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim):
        super(ResRNN, self).__init__()
        self.h_dim, self.y_dim = h_dim, out_dim

        self.inp = nn.Linear(in_dim, h_dim)
        self.xh_res = ResNetBlock(2 * h_dim)
        self.xh2h = nn.Linear(2 * h_dim, h_dim)
        self.out = nn.Linear(h_dim, out_dim)

    def step(self, x, h):
        x = self.inp(x)
        xh = torch.cat((x, h), dim=0)
        temp = self.xh_res(xh)
        h = self.xh2h(temp)
        y_hat = self.out(h)
        return y_hat, h

    def forward(self, x, h):
        rollout = x.size(1)
        output = torch.zeros(rollout, self.y_dim)
        for i in range(rollout):
            output[i], h = self.step(x[0, i], h)
        return output.unsqueeze(0), h

    def init_hidden(self):
        return torch.zeros(self.h_dim)

