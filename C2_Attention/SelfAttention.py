import torch
import torch.nn as nn

'''

In Seq2Seq, the simplest attention model, Q = hidden_state_decoder, 
K = All the encoder hidden outputs , V = K in this case. In case of 
self attention , Q = V = K


Adding to it, Multihead attention is just a clever way to parameter-
ise dot-product attention mechanism

'''

class Attention:# (nn.Module):
    def __init__(self, emb_dim, score_fn='dotScaled', bias=True):
        # super().__init__()
        options = {'dot':self.dot, 'dotScaled':self.dotScaled, 
                    'contentBased':self.contentBased}
        self.score = options[score_fn]
        self.softmax = nn.Softmax(dim=-2)
    
    def dot(self, Q, K):
        return torch.bmm(K, Q.transpose(-1, -2).contiguous())
    
    def dotScaled(self, Q, K):
        n = K.size(1)
        return self.dot(Q, K) / n**0.5
    
    def contentBased(self, Q, K):
        dot_ = self.dot(Q, K)
        st = 'bij,bij->b' if len(Q.size())==3 else 'ij,ij->'# if batching
        col_vecs = torch.einsum(st+'i', Q, K)
        outer = torch.bmm(col_vecs.unsqueeze(-1), col_vecs.unsqueeze(-2))
        return torch.div(dot_, outer + 1e-16)
    
    def __call__(self, Q, K, V):
        W = self.softmax(self.score(Q, K))# dim: (batch_n, seq_len, seq_len)
        Y = torch.bmm(V.transpose(-1, -2).contiguous(), W).transpose(-1, -2).contiguous()
        # (batch_n, emb_dim, seq_len) * (batch_n, seq_len (softmax), seq_len) -> (batch_n, seq_len, emb_dim)
        return Y


class Head(nn.Module):
    def __init__(self, in_dim, latent_dim, score_fn='dotScaled', bias=True):
        super().__init__()
        
        self.key_prj = nn.Linear(in_dim, latent_dim, bias=bias)
        self.qry_prj = nn.Linear(in_dim, latent_dim, bias=bias)
        self.val_prj = nn.Linear(in_dim, latent_dim, bias=bias)

        self.attention = Attention(latent_dim, score_fn=score_fn, bias=bias)

    def forward(self, Q, K, V):
        K = self.key_prj(Q)
        Q = self.qry_prj(K)
        V = self.val_prj(V)

        return self.attention(Q, K, V)

class MultiHead(nn.Module):
    def __init__(self, in_dim, latent_dim=None, n_heads=4, score_fn='dotScaled', bias=True):
        super().__init__()
        if latent_dim is None: latent_dim = in_dim
        self.n_heads, self.in_dim, self.latent_dim = n_heads, in_dim, latent_dim
        self.heads = nn.ModuleList([Head(in_dim, latent_dim, score_fn, bias) for _ in range(n_heads)])
        self.out_proj = nn.Linear(n_heads * latent_dim, in_dim, bias=bias)

    def forward(self, Q, K, V):
        # batch_size, seq_len, in_dim = Q.size()
        heads_out = torch.cat([h(Q, K, V) for h in self.heads], dim=-1)
        return self.out_proj( heads_out )








class SelfAttention(nn.Module):
    def __init__(self, emb_dim, score_fn='dotScaled', bias=True):
        super().__init__()
        
        self.key_prj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.qry_prj = nn.Linear(emb_dim, emb_dim, bias=bias)
        self.val_prj = nn.Linear(emb_dim, emb_dim, bias=bias)

        options = {'dot':self.dot, 'dotScaled':self.dotScaled, 
                    'contentBased':self.contentBased}
        self.score = options[score_fn]

        self.softmax = nn.Softmax(dim=-2)

    def dot(self, Q, K):
        return torch.bmm(K, Q.transpose(-1, -2).contiguous())
    
    def dotScaled(self, Q, K):
        n = K.size(1)
        return self.dot(Q, K) / n**0.5
    
    def contentBased(self, Q, K):
        dot_ = self.dot(Q, K)
        st = 'bij,bij->b' if len(Q.size())==3 else 'ij,ij->'# if batching
        col_vecs = torch.einsum(st+'i', Q, K)
        outer = torch.bmm(col_vecs.unsqueeze(-1), col_vecs.unsqueeze(-2))
        return torch.div(dot_, outer + 1e-16)

    def forward(self, embedded):
        ''' embedded (batch_n, seq_len, emb_dim) acts as keys, queries and values. '''
        batch_n, seq_len, emb_dim = embedded.size()

        K = self.key_prj(embedded)
        Q = self.qry_prj(embedded)
        V = self.val_prj(embedded)

        W = self.softmax(self.score(Q, K))# dim: (batch_n, seq_len, seq_len)

        Y = torch.bmm(V.transpose(-1, -2).contiguous(), W).transpose(-1, -2).contiguous()
        # (batch_n, emb_dim, seq_len) * (batch_n, seq_len (softmax), seq_len) -> (batch_n, seq_len, emb_dim)
        return Y


# emb_dim = 23
# s_attn = SelfAttention(emb_dim)

# x = torch.randn(3, 13, emb_dim)
# y = s_attn(x)
# print(y.size())