# import torch
# import torch.nn as nn


# class Trainer:
#   def __init__(self, ):
#       self.a = None



# class Unsupervised(Trainer):
#   def __init__(self, seq_data):
#       self.vocab = bpe(seq_data, num_merges=40000)
#       emb_dim = 768
#       self.tok_emb = nn.Embedding(len(vocab), emb_dim)
#       num_layers = 12
#       num_heads = 12# per layer
#       pos_num_dim = 3072
#       self.optimiser = nn.optim.Adam(lr=2.5e-4)
#       atn_drop = 0.1
#       res_drop = 0.1
#       emb_drop = 0.1
#       # Modified version of L2 regularisation was also used for non-bias weights.

from Model import *
from BPE import BPE

import io
import torch
# from Vocab import *
import string
from tqdm import tqdm

# _patterns_dict = {r'\'':' \'  ',
#                   r'\"':'',
#                   r'\.':' . ',
#                   r'<br \/>':' ',
#                   r',':' , ',
#                   r'\(':' ( ',
#                   r'\)':' ) ',
#                   r'\!':' ! ',
#                   r'\?':' ? ',
#                   r'\;':' ',
#                   r'\:':' ',
#                   r'\s+':' '}
# char_filt = [(re.compile(p), r) for p, r in _patterns_dict.items()]


# def tidy_data(train_path, test_path, valid_path):
#     '''
#     Make the data nice:
#     - make chars ascii
#     - 

#     '''
#     def open_ascii(path):
#         data = open(path, 'r').read()
#         return "".join(filter(lambda x: x in printable, data)).split('\n')[:2000]

#     def normalise(line):
#         """
#         Basic normalization for a line of text.
#         Normalization includes
#         - lowercasing
#         - complete some basic text normalization for English words as follows:
#             remove '\"'
#             add spaces before and after '\'' '.' ',' '(' ')' '!' '?'
#             replace ';' ':' '<br \/>' with single space
#             replace multiple spaces with single space

#         Returns a list of tokens after splitting on whitespace.
#         """
#         line = line.lower()
#         for pattern_re, replaced_str in char_filt:
#             line = pattern_re.sub(replaced_str, line)
#         return line.split()

#     printable = set(string.printable)

#     train = open_ascii(train_path)
#     test = open_ascii(test_path)
#     valid = open_ascii(valid_path)

#     vocab, bpe_vocab = build_vocab_from_iterator(map(normalise, iter(train)))

#     return (vocab, bpe_vocab, normalise), (train, test, valid)

# def text2tensor(raw_text_iter):
#     data = [torch.tensor([vocab[token] for token in tokeniser(item)],
#                          dtype=torch.long) for item in raw_text_iter]
#     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# (vocab, bpe_vocab, tokeniser), (train, test, valid) = tidy_data('wikitext-2/wiki.train.tokens', 
#                                         'wikitext-2/wiki.test.tokens', 
#                                         'wikitext-2/wiki.valid.tokens')

def open_and_process(path):
    string_data = open(path, 'r').read()
    printable = set(string.printable)
    not_printable = {'@', '%', '*', '`','+', '~', '\x0c', '^', '$', '#', '_'}
    string_data = string_data.lower()
    string_data = "".join(filter(lambda x: x in printable, string_data))
    string_data = "".join(filter(lambda x: x not in not_printable, string_data))
    return string_data

bpe = BPE('BPE-Maps')
tr_str = open_and_process('stories_data/train.txt')
ts_str = open_and_process('stories_data/test.txt')
vl_str = open_and_process('stories_data/validation.txt')

vocab = bpe.get_vocab(tr_str)
print(len(vocab))
# bpe.generate_tokens(vocab, num_merges=10000)# takes a long time, but only needs to be done once
bpe.precompute_encode_map()# takes a long time, but only needs to be done once

sorted_tokens = bpe.load(bpe.sorted_tokens_path)

get_ixs = lambda data:bpe.encode(data, sorted_tokens, to_ix=True)
tr_ixs, ts_ixs, vl_ixs = get_ixs(tr_str), get_ixs(ts_str), get_ixs(vl_str)

train_data = torch.tensor(tr_ixs, dtype=torch.long)
test_data = torch.tensor(ts_ixs, dtype=torch.long)
valid_data = torch.tensor(vl_ixs, dtype=torch.long)

def text2tensor(words):
    enc = bpe.encode(words, sorted_tokens, to_ix=True)
    return torch.tensor(enc, dtype=torch.long)

# # print(vocab.words_and_frequencies)

# # print(''.join(train)[:100])
# # print('^^ train,   test vv')
# # print(''.join(test)[:100])
# # print('validation vv')
# # print(''.join(valid)[:100])
# print('building tensors...', end='')
# train_data, test_data, valid_data = text2tensor(train), text2tensor(test), text2tensor(valid)
# print('done.')

# # print(train_data.shape)
# # print(' '.join([vocab.itos[int(i)] for i in train_data[:4000]]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 64
eval_batch_size = 16

print('batching tensors...', end='')
train_data = batchify(train_data, batch_size)
valid_data = batchify(valid_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)
print('done.')

"""Functions to generate input and target sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``get_batch()`` function generates the input and target sequence for
the transformer model. It subdivides the source data into chunks of
length ``bptt``. For the language modeling task, the model needs the
following words as ``Target``. For example, with a ``bptt`` value of 2,
we’d get the following two Variables for ``i`` = 0:

![](https://github.com/pytorch/tutorials/blob/gh-pages/_downloads/_static/img/transformer_input_target.png?raw=1)


It should be noted that the chunks are along dimension 0, consistent
with the ``S`` dimension in the Transformer model. The batch dimension
``N`` is along dimension 1.
"""

bptt = 50

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

"""Initiate an instance
--------------------

The model is set up with the hyperparameter below. The vocab size is
equal to the length of the vocab object.
"""

ntokens = len(sorted_tokens)#len(vocab.stoi) # the size of vocabulary
emsize = 16 # embedding dimension
nhid = 16 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

"""Run the model
-------------

`CrossEntropyLoss <https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss>`__
is applied to track the loss and
`SGD <https://pytorch.org/docs/master/optim.html?highlight=sgd#torch.optim.SGD>`__
implements stochastic gradient descent method as the optimizer. The initial
learning rate is set to 5.0. `StepLR <https://pytorch.org/docs/master/optim.html?highlight=steplr#torch.optim.lr_scheduler.StepLR>`__ is
applied to adjust the learn rate through epochs. During the
training, we use
`nn.utils.clip_grad_norm\_ <https://pytorch.org/docs/master/nn.html?highlight=nn%20utils%20clip_grad_norm#torch.nn.utils.clip_grad_norm_>`__
function to scale all the gradient together to prevent exploding.
"""

criterion = nn.CrossEntropyLoss()
lr = 5.0 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

log_interval = 2

import time
def train():
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    ixs = range(0, train_data.size(0) - 1, bptt)
    pbar = tqdm(zip(range(len(ixs)), ixs))
    for batch, i in pbar:
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time

            strn = '| epoch {:3d} | {:5d}/{:5d} batches | \
                  lr {:02.2f} | ms/batch {:5.2f} | \
                  loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss))
            # print(strn)
            pbar.set_description(strn)

            total_loss = 0
            start_time = time.time()

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

"""Loop over epochs. Save the model if the validation loss is the best
we've seen so far. Adjust the learning rate after each epoch."""

best_val_loss = float("inf")
epochs = 2 # The number of epochs
best_model = None

print('starting training')
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, valid_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model

    scheduler.step()

"""Evaluate the model with the test dataset
-------------------------------------
Apply the best model to check the result with the test dataset."""

test_loss = evaluate(best_model, test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


@torch.no_grad()
def inference(model, context, steps, temperature=1.0, sample=False, top_k=None):
    words = context.split(' ')
    x = text2tensor(words).unsqueeze(dim=0)

    src_mask = model.generate_square_subsequent_mask(bptt).to(device)

    for i in range(steps):
        if len(x) < bptt:
            mask = src_mask[:len(x), :len(x)]
        else:
            mask = src_mask
        
        logits = model(x, mask)

        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        if sample:
            w_ix = torch.multinomial(probs, num_samples=1)
        else:
            _, w_ix = torch.topk(probs, k=1, dim=-1)

        x = torch.cat((x, w_ix), dim=1)

    sorted_tokens_inv = {v:k for k,v in sorted_tokens.items()}
    completion = bpe.decode([int(i) for i in x.squeeze()], sorted_tokens_inv)
    # completion = ' '.join([vocab.itos[int(i)] for i in x.squeeze()])
    print(completion)

context = "But why this happens"

sample = True
top_k = 10
temperature = 1.0
inference(model, context, 100, temperature=temperature, sample=sample, top_k=top_k)
