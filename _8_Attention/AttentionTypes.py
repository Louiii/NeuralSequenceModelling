import torch
import torch.nn as nn

# TODO: Doesn't fully support batching and multi-layered RNNs


'''
Self-Attention:
Relating different positions of the same input sequence. Theoretically the self-attention 
can adopt any score functions above, but just replace the target sequence with the same 
input sequence.

Global/Soft: 
Attending to the entire input state space.

Local/Hard:
Attending to the part of input state space; i.e. a patch of the input image.


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



class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.
    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:
    """

    def __init__(self, enc_dim, dec_dim, attention_type='general', max_length=None):
        super(Attention, self).__init__()

        self.type = attention_type
        if self.type == 'general':
            self.W_a = nn.Linear(dec_dim, dec_dim, bias=False)
        if self.type == 'location-based':
            self.W_l = nn.Linear(dec_dim, max_length)
        if self.type == 'additive':
            self.W1_a = nn.Linear(dec_dim, dec_dim, bias=False)
            self.W2_a = nn.Linear(enc_dim, dec_dim, bias=False)
            self.v = torch.nn.Parameter(torch.FloatTensor(dec_dim).uniform_(-0.1, 0.1))

        sc_fn = {'dot product':self.dotProduct, 'scaled dot product':self.scaledDotProduct, 
                 'location-based':self.locationBased, 'general':self.general, 
                 'additive':self.additive, 'content-based':self.contentBased}
        if attention_type not in sc_fn:
            raise NotImplementedError(attention_type+" not supported. Options: "+str(sc_fn.keys()))
        else:
            self.score = sc_fn[self.type]

        # self.lin_out = nn.Linear(dec_dim * 2, dec_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.cos_sim = nn.CosineSimilarity(dim=2)

    def dotProduct(self, query, context):# Luong2015
        context = context.permute([1, 2, 0]).contiguous()
        return torch.bmm(query, context)# q^T * Hs

    def scaledDotProduct(self, query, context):# Vaswani2017
        n = context.size(1)# dim of  source hidden state
        '''
        motivated by the concern when the input is large, the softmax function may have an 
        extremely small gradient, hard for efficient learning.
        '''
        context = context.permute([1, 2, 0]).contiguous()
        return torch.bmm(query, context) / n**0.5# q^T * Hs

    def locationBased(self, query, context):# Luong2015
        return self.W_l(query)[:,:,:context.size(0)]

    def general(self, query, context):# Luong2015
        query = self.W_a(query)
        context = context.permute([1, 2, 0]).contiguous()
        return torch.bmm(query, context)

    def additive(self, query, context):# Bahdanau2015
        # v . tanh( W * [s_t; h_i] ) { W * [s_t; h_i] == [W1 * s_t, W2 * h_i], where W == [W1, W2] }
        WS = self.W1_a(query)
        WH = self.W2_a(context).transpose(0, 1).contiguous()
        scr = self.tanh(WS + WH) @ self.v
        return scr.unsqueeze(0)

    def contentBased(self, query, context):# Graves2014
        # cosine[query, context]
        return self.cos_sim(query, context).transpose(0, 1).unsqueeze(0)


    # def bahdanau_forward(self, query, context):# query should initially be the last encoder hidden state
    #     attention_weights = self.softmax(self.score(query, context))
    #     # (out_dim, query_len) * (query_len, dimensions) -> (out_dim, dimensions)
    #     context_vector = torch.mm(attention_weights, context)


    #     # _, output = self.decoder(context_vector.unsqueeze(0), query.unsqueeze(0))# X, h
    #     # output = output.squeeze(0)

    #     combined = torch.cat((context_vector, query), dim=1)
    #     output = self.lin_out(combined)
    #     output = self.tanh(output)

    #     dec_input = torch.cat((context_vector, output), dim=1)

    #     return output, attention_weights, dec_input

    def forward(self, query, context):
        """
        Args:
            query [batch size, output length, dimensions]: e.g. previous decoder hidden 
            state to query the context.
            context [batch size, query length, dimensions]: e.g. the encoder hidden states, 
            data to apply the attention mechanism.

        Returns:
            * output (long tensor) [batch size, output length, dimensions]:
              Tensor containing the attended features.
            * weights [batch size, output length, query length]:
              Tensor containing attention weights.
        """
        # output_len, dimensions = query.size()
        # query_len = context.size(0)

        # (out_dim, h_dim) * (query_len, h_dim) -> (out_dim, query_len)
        # Compute weights across every context sequence
        attention_weights = self.softmax(self.score(query, context))

        # (out_dim, query_len) * (query_len, dimensions) -> (out_dim, dimensions)
        # print("attn weights, context: "+str(attention_weights.size())+", "+str(context.transpose(0,1).size()))
        context_vector = torch.bmm(attention_weights, context.transpose(0,1).contiguous())


        # _, output = self.decoder(context_vector.unsqueeze(0), query.unsqueeze(0))# X, h
        # output = output.squeeze(0)

        # query_context = torch.cat((context_vector, query), dim=1)
        # query_context = self.lin_out(query_context)
        # query_context = self.tanh(query_context)

        return context_vector, attention_weights

class RNN(nn.Module):
    def __init__(self, x_dim, h_dim, rnn_algo, bidir=False):
        super(RNN, self).__init__()

        self.n_dir, self.h_dim, self.lstm = 1+bidir, h_dim, rnn_algo=="LSTM"

        rnns = {"Simple":nn.RNN, "GRU":nn.GRU, "LSTM":nn.LSTM}
        if rnn_algo not in rnns:
            raise NotImplementedError(rnn_algo+" not supported. Options: "+str(rnns.keys()))
        else:
            self.rnn = rnns[rnn_algo](x_dim, h_dim, bidirectional=bidir)
    
    def forward(self, x, h):
        return self.rnn(x, h)
    
    def initHidden(self, bs=1):
        if self.lstm:
            return (torch.zeros(self.n_dir, bs, self.h_dim),
                    torch.zeros(self.n_dir, bs, self.h_dim))
        return torch.zeros(self.n_dir, bs, self.h_dim)

class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, rnn_type, bidir=True):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding(x_dim, h_dim)
        self.rnn = RNN(h_dim, h_dim, rnn_type, bidir=bidir)
        self.initHidden = self.rnn.initHidden
        self.h_dim = self.rnn.h_dim

    def forward(self, x, h):
        embedded = self.embed(x)
        return self.rnn(embedded, h)

class LuongDecoder(nn.Module):
    def __init__(self, attention, h_dim, y_dim, decoder_type):
        super(LuongDecoder, self).__init__()

        self.attn = attention

        self.embed = nn.Embedding(y_dim, h_dim)
        self.classifier = nn.Linear(2 * h_dim, y_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        self.rnn = RNN(h_dim, h_dim, decoder_type)
    
    def forward(self, prev_y_hat, dec_h, enc_hs):
        embed_y = self.embed(prev_y_hat)
        
        _, dec_h = self.rnn(embed_y, dec_h)
        dec_h_ = dec_h[0] if self.rnn.lstm else dec_h# only use h from lstm (not c)

        context, attention_weights = self.attn(dec_h_, enc_hs)

        dec_h_context = torch.cat((context, dec_h_), dim=-1)
        
        y_hat_probs = self.log_softmax(self.classifier(dec_h_context))
        
        return y_hat_probs, dec_h, attention_weights
