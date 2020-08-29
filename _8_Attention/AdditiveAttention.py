import torch
import torch.nn as nn

class Encoder(torch.nn.Module):
    def __init__(self, in_dim, h_dim, n_layers=2):
        self.rnn = nn.GRU(input_size=in_dim, 
                          hidden_size=h_dim, 
                          num_layers=n_layers,
                          bidirectional=True)

    def forward(self, x, h):
        return self.rnn(x, h)

class Decoder(torch.nn.Module):
    def __init__(self, enc_h_dim, dec_h_dim, n_layers=1):
        super().__init__()    

        self.rnn = nn.GRU(input_size=enc_h_dim, 
                          hidden_size=dec_h_dim, 
                          num_layers=n_layers)
        self.out = torch.nn.Linear(encoder_dim, out_dim)

        # context vector 
        self.v = torch.nn.Parameter(torch.rand(decoder_dim))
        self.Wa = torch.nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.Ua = torch.nn.Linear(encoder_dim, decoder_dim)

    def Uhs(self, enc_hs):
        return self.Ua(enc_hs)

    def compute_alpha(self, query):
        ''' query: [decoder_dim] 
            values: [seq_length, encoder_dim] '''
      
        weights = torch.tanh(self.Wa(query).repeat(self.Uh.size(0), 1) + self.Uh) @ self.v # [seq_length]

        weights = torch.nn.functional.softmax(weights, dim=0)        
        return weights @ values # [encoder_dim] 

    def context(self, enc_hs, prev_s):
        alpha = self.compute_alpha(prev_s)
        return torch.matmul(alpha, enc_hs)

    def step(self, y_prev, enc_hs, prev_s):
        c = self.context(enc_hs)
        # some input of y_prev and c into self.rnn
        x = torch.cat((y_prev, c), dim=-1)
        out, s = self.rnn(x, s)
        yhat = self.out(out)
        return s, yhat

    def forward(self, enc_hs):
        self.Uh = self.Uhs(enc_hs)
        prev_s = torch.zeros()
        yhat = None
        yhats = [self.step()]
        while yhat is not EOF:
            prev_s, yhat = self.step(yhats[-1], enc_hs, prev_s)
            yhats.append(yhat)
        return torhc.Tensor(yhats)

