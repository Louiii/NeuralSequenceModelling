import torch
import torch.nn as nn

'''
Self-Attention:
Relating different positions of the same input sequence. Theoretically the self-attention 
can adopt any score functions above, but just replace the target sequence with the same 
input sequence.

Global/Soft: 
Attending to the entire input state space.

Local/Hard:
Attending to the part of input state space; i.e. a patch of the input image.
'''

class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
    """

    def __init__(self, dimensions, attention_type, decoder_type):
        super(Attention, self).__init__()
        self.type = attention_type
        if self.type in ['general', 'location-base attention']:
            self.W_a = nn.Linear(dimensions, dimensions, bias=False)
        if self.type == 'additive':
            self.W1_a = nn.Linear(decoder_dim, decoder_dim, bias=False)
            self.W2_a = nn.Linear(encoder_dim, decoder_dim, bias=False)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.decoder_dim).uniform_(-0.1, 0.1))

        sc_fn = {'dot product':self.dotProduct, 'scaled dot product':self.scaledDotProduct, 
                 'location-base attention':self.locationBase, 'general':self.general, 
                 'additive':self.additive, 'content-base attention':self.contentBase}
        self.score = sc_fn[self.type]

        if decoder_type=="Simple":
            self.decoder = nn.RNN(dimensions, dimensions, bias=False)
        elif decoder_type=="GRU":
            self.decoder = nn.GRU(dimensions, dimensions, bias=False)

        # self.lin_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def dotProduct(self, query, context):# Luong2015
        return torch.mm(query, context.transpose(0, 1).contiguous())# q^T * Hs

    def scaledDotProduct(self, query, context):# Vaswani2017
        n = context.size(1)# dim of  source hidden state
        '''
        motivated by the concern when the input is large, the softmax function may have an 
        extremely small gradient, hard for efficient learning.
        '''
        return torch.mm(query, context.transpose(0, 1).contiguous()) / n**0.5# q^T * Hs

    def locationBase(self, query, context):# Luong2015
        return self.W_a(query)

    def general(self, query, context):# Luong2015
        query = self.W_a(query)
        return torch.mm(query, context.transpose(0, 1).contiguous())

    def additive(self, query, context):# Bahdanau2015
        # v . tanh( W * [s_t; h_i] ) { W * [s_t; h_i] == [W1 * s_t, W2 * h_i], where W == [W1, W2] }
        WS = self.W1_a(query)
        WH = self.W2_a(context)
        return self.tanh(WS + WH) @ self.v

    def contentBase(self, query, context):# Graves2014
        # cosine[query, context]
        return nn.CosineSimilarity(query, context)

    def forward(self, query, context):
        """
        Args:
            query [output length, dimensions]: e.g. previous decoder hidden 
            state to query the context.
            context [query length, dimensions]: e.g. the encoder hidden states, 
            data to apply the attention mechanism.

        Returns:
            * output (long tensor) [output length, dimensions]:
              Tensor containing the attended features.
            * weights [output length, query length]:
              Tensor containing attention weights.
        """
        output_len, dimensions = query.size()
        query_len = context.size(0)

        # (output_len, dimensions) * (query_len, dimensions) -> (output_len, query_len)
        attention_scores = self.score(query, context)

        # Compute weights across every context sequence
        attention_weights = self.softmax(attention_scores)

        # (output_len, query_len) * (query_len, dimensions) -> (output_len, dimensions)
        context_vector = torch.mm(attention_weights, context)


        _, output = self.decoder(context_vector.unsqueeze(0), query.unsqueeze(0))# X, h
        output = output.squeeze(0)

        # combined = torch.cat((context_vector, query), dim=1)
        # output = self.lin_out(combined)
        # output = self.tanh(output)

        return output, attention_weights

class EncoderDecoder(nn.Module):
    """ Seq2seq model with attention """
    def __init__(self, x_dim, h_dim, y_dim, attention_type='general', 
                enc_type="Simple", dec_type="Simple", max_out_len=100):
        super(EncoderDecoder, self).__init__()
        self.h_dim = h_dim
        self.y_dim = y_dim
        self.max_out_len = max_out_len

        enc_types = {"Simple":nn.RNN, "GRU":nn.GRU}
        self.encoder = enc_types[enc_type](x_dim, h_dim)
        # if enc_type=="Simple":
        #     self.encoder = nn.RNN(x_dim, h_dim)
        # elif enc_type=="GRU":
        #     self.encoder = nn.GRU(x_dim, h_dim)

        self.attention = Attention(h_dim, attention_type, dec_type)

        self.h2o = nn.Linear(h_dim, y_dim)
        self.tanh = nn.Tanh()

        self.EOF = None

        
    def encode(self, x, eh):
        output, hn = self.encoder(x, eh)
        return output# hidden states

    def step(self, hs, dec_h):
        dec_h = torch.zeros(1, self.h_dim) if dec_h==None else dec_h
        next_dec_h, _ = self.attention(dec_h, hs)
        return next_dec_h

    def forward(self, x, eh=None, dh=None):
        hs = self.encode(x, eh)
        hs = hs.squeeze(1)

        ys = torch.zeros(self.max_out_len, self.y_dim)
        for i in range(self.max_out_len):
            dh = self.step(hs, dh)
            ys[i,:] = self.tanh(self.h2o(dh))
            if ys[i,:]==self.EOF: break

        return ys[:i+1,:]









'''

Encoder-decoder model with additive attention mechanism: Bahdanau et al., 2015.
   ╭-----╮   ╭-----╮
...|s_t-1|-->| s_t |...  Decoder
   ╰-----╯ / ╰-----╯
           |
           ⊕             Context vector
         ⟋/| ⟍  
       ⟋ / |   ⟍
  a1 ⟋a2/a3|   an⟍      Alignment weights at time step t. These also depend on s_t-1
   ⟋   /   |       ⟍
┌--┐ ┌--┐ ┌--┐     ┌--┐ 
|h1|>|h2|>|h3|>... |hn|  Encoder: forward
|g1|<|g1|<|g1|<... |g1|  Encoder: backward
└--┘ └--┘ └--┘     └--┘
 x1   x2   x3       xn


The equations:
x = [x1,..., xn]
y = [y1,..., ym]

Say Encoder is a bidirectional RNN, H_i = [h_i; g_i]
Decoder has one direction, s_t = f(s_t-1, y_t-1, c_t)

c_t = sum_i=1,..,n { a_{t,i} * h_i }
a_{t,i} = align(y_t, x_i) = Softmax(score(s_{t-1}, h_i))

score(s_t, h_i) = feedforward({dim(s)+dim(h),..., 1})
e.g.            = v . tanh( W * [s_t; h_i] ) {params: v, W}


The algorithm:
1. Run Encoder, store h_i i=1,..,n
2. for t in range(m):# while True: # and stop when EOF char is output
       compute a_{t,i} for all i
       compute context vector, c_t
       take one step for the decoder with s_t-1 as hidden and c_t as input
       produce y_t

'''



in_seq_len  = 12
x_dim = 1
h_dim = 256
y_dim = 3
model = EncoderDecoder(x_dim, h_dim, y_dim)

X = torch.randn(in_seq_len, 1, x_dim)
Y_hat = model(X)

print(Y_hat.size())

# attention = Attention(h_dim)
# query = torch.randn(1, h_dim)
# context = torch.randn(5, h_dim)
# output, weights = attention(query, context)

# print(output.size())
# print(weights.size())