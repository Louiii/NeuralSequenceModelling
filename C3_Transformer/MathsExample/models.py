import torch, torch.nn as nn
from torch.functional import F

class M1(nn.Module):
    def __init__(self, vocab_size, hyp):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, hyp.h_dim)
        self.prj = nn.Linear(hyp.h_dim, vocab_size, bias=False)

    def forward(self, x):
        # this can only learn length 1 dependencies
        return self.prj(self.tok_emb(x))

class MaskedSelfAttention(nn.Module):
    def __init__(self, h_dim, block_size):
        super().__init__()
        self.qry = nn.Linear(h_dim, h_dim)
        self.key = nn.Linear(h_dim, h_dim)
        self.val = nn.Linear(h_dim, h_dim)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, block_size, block_size))

    def forward(self, x):
        B, T, H = x.size()
        k, q, v = self.key(x), self.qry(x), self.val(x)# (B, T, H)

        att = torch.bmm(q, k.transpose(-2, -1)) / H**0.5
        att = att.masked_fill(self.mask[:,:T,:T] == 0, float('-inf'))
        weights = F.softmax(att, dim=-1)
        return torch.bmm(weights, v)# (B, T, T) x (B, T, H) -> (B, T, H)

class MultiHeadMaskedSelfAttention(nn.Module):
    def __init__(self, h_dim, block_size, nheads):
        super().__init__()
        self.qry = nn.Linear(h_dim, h_dim)
        self.key = nn.Linear(h_dim, h_dim)
        self.val = nn.Linear(h_dim, h_dim)

        assert h_dim%nheads==0, 'h_dim must be divisible by nheads'
        self.H = h_dim // nheads
        self.nH = nheads

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, block_size, block_size))

    def cast(self, a):
        ''' Bring the head dimension into the batch dimension
        a: [B, T, allH] -> [B * self.nH, T, self.H] '''
        return a.view(self.B, self.T, self.nH, self.H).transpose(1, 2).reshape(self.B*self.nH, self.T, self.H)

    def uncast(self, a):
        ''' Bring the head dim out of the batch dim and into the hidden dim
        a: [B * self.nH, T, self.H] -> [B, T, allH] '''
        return a.view(self.B, self.nH, self.T, self.H).transpose(1, 2).reshape(self.B, self.T, self.allH)

    def forward(self, x):
        self.B, self.T, self.allH = x.size()

        k = self.cast(self.key(x))# (B*nH, T, H)
        q = self.cast(self.qry(x))# (B*nH, T, H)
        v = self.cast(self.val(x))# (B*nH, T, H)

        att = torch.bmm(q, k.transpose(-2, -1)) / self.allH**0.5
        att = att.masked_fill(self.mask[:,:self.T,:self.T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        return self.uncast(torch.bmm(att, v))# (B, T, T) x (B, T, hs) -> (B, T, hs)

class M2(nn.Module):
    def __init__(self, vocab_size, hyp, multihead=True):
        super().__init__()
        self.block_size, self.nhead = hyp.block_size, hyp.nhead

        self.tok_emb = nn.Embedding(vocab_size, hyp.h_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, hyp.block_size, hyp.h_dim))

        self.prj1 = nn.Linear(hyp.h_dim, hyp.h_dim, bias=False)

        if multihead:
            self.atn = MultiHeadMaskedSelfAttention(hyp.h_dim, hyp.block_size, hyp.nhead)
        else:
            self.atn = MaskedSelfAttention(hyp.h_dim, hyp.block_size)

        # self.prj = nn.Linear(hyp.h_dim, hyp.h_dim, bias=False)
        # self.prj2category = nn.Linear(hyp.h_dim, vocab_size, bias=False)
        dims = [hyp.h_dim, hyp.h_dim, vocab_size]
        layers = []
        for i in range(len(dims)-1):
            layers.append( nn.Linear(dims[i], dims[i+1]) )
            if i<len(dims)-1: layers += [nn.ReLU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, ixs):
        b, t = ixs.size()
        if t > self.block_size: 
            ixs = ixs[:, -self.block_size]
        
        # print('\nixs.shape: '+str(ixs.shape))
        # print(ixs[0])
        # print(''.join([dataset.i2s(x_) for x_ in ixs[0]]))

        token_emb = self.tok_emb(ixs) # each index maps to a (learnable) vector
        position_emb = self.pos_emb[:, :t, :]

        x = token_emb + position_emb# [Batch, Seq, Emb]

        x = self.prj1(x)

        # print('\nx.shape: '+str(x.shape))
        # print(x[0])

        y = self.atn(x)

        # output residual projection
        # y = self.resid_drop(self.prj(y))
        # y = y + self.prj(y)

        # # print('\ny.shape: '+str(y.shape))
        # # print(y[0])

        # y = self.prj2category(y)
        y = self.mlp(y)

        # print('\ny.shape: '+str(y.shape))
        # print(y[0])

        return y

class Block(nn.Module):
    def __init__(self, vocab_size, hyp, multihead=True):
        super().__init__()
        self.block_size, self.nhead = hyp.block_size, hyp.nhead

        if multihead:
            self.atn = MultiHeadMaskedSelfAttention(hyp.h_dim, hyp.block_size, hyp.nhead)
        else:
            self.atn = MaskedSelfAttention(hyp.h_dim, hyp.block_size)

        self.mlp = nn.Sequential(
            nn.Linear(hyp.h_dim, 4 * hyp.h_dim),
            nn.GELU(),
            nn.Linear(4 * hyp.h_dim, hyp.h_dim),
            nn.Dropout(hyp.dropout),
        )

    def forward(self, x):
        x = x + self.atn(x)
        return x + self.mlp(x)

class M3(nn.Module):
    def __init__(self, vocab_size, hyp, multihead=True):
        super().__init__()
        self.block_size, self.nhead = hyp.block_size, hyp.nhead

        self.tok_emb = nn.Embedding(vocab_size, hyp.h_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, hyp.block_size, hyp.h_dim))

        self.blocks = nn.Sequential(*[Block(vocab_size, hyp, multihead=True) for _ in range(hyp.nlayers)])
        self.prj = nn.Linear(hyp.h_dim, vocab_size, bias=True)
        self.ln = nn.LayerNorm(hyp.h_dim)

    def forward(self, ixs):
        b, t = ixs.size()
        if t > self.block_size: 
            ixs = ixs[:, -self.block_size]

        token_emb = self.tok_emb(ixs) # each index maps to a (learnable) vector
        position_emb = self.pos_emb[:, :t, :]

        x = token_emb + position_emb# [Batch, Seq, Emb]
        x = self.blocks(x)
        x = self.ln(x)
        
        return self.prj(x)

class Extention(nn.Module):
    def __init__(self, base_model, vocab_size, hyp):
        super().__init__()
        self.base = base_model
        self.max_T = hyp.max_T
        self.vocab_size = vocab_size

        # dims = [hyp.h_dim, hyp.h_dim, vocab_size]
        # layers = []
        # for i in range(len(dims)-1):
        #     layers.append( nn.Linear(dims[i], dims[i+1]) )
        #     if i<len(dims)-1: layers += [nn.ReLU()]
        # self.mlp = nn.Sequential(*layers)

    def forward(self, ixs):
        '''
        plug in x, then append each char for yhat
        return just yhat
        compute the loss on yhat, y
        '''


        # # ixs: [Batch, Seq] -> y: [Batch, Seq, Seq]
        # att = self.base.internal(ixs)
        # B, T, T = att.size()
        B, T = ixs.size()

        # we want an output sequence for each input in the batch
        # out = [Batch, NewSeq, Vocab]
        temperature = 50
        y = torch.zeros(B, self.max_T, self.vocab_size)
        xy = ixs
        for i in range(self.max_T):
            logits = self.base(xy)
            y[:,i,:] = logits[:, -1, :]
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # print(probs[0])
            w_ix = torch.multinomial(probs, num_samples=1)
            # _, w_ix = torch.topk(probs, k=1, dim=-1)
            xy = torch.cat((xy, w_ix), dim=1)
        return y