'''
Eng-Fra translation data preprocessing, from:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math

SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalizeString(s):
    s = s.lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/classification/%s_%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

'''
Prepare the training data:

To train, for each pair we will need an input tensor (indexes of the words in
the input sentence) and target tensor (indexes of the words in the target
sentence). While creating these vectors we will append the EOS token to both
sequences.
'''

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, device):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair, input_lang, output_lang, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)

def evaluate(encoder, decoder, sentence, input_lang, output_lang, device, max_length=MAX_LENGTH):
    '''
    Evaluation is mostly the same as training, but there are no targets so we
    simply feed the decoder’s predictions back to itself for each step. Every
    time it predicts a word we add it to the output string, and if it predicts
    the EOS token we stop there. We also store the decoder’s attention outputs
    for display later.
    '''
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, device)
        input_length = input_tensor.size()[0]
        enc_h = encoder.initHidden()

        enc_out, enc_h = encoder(input_tensor, enc_h)

        dec_inp = torch.tensor([[SOS_token]], device=device)  # SOS

        dec_h = enc_h

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, enc_out.size(0))

        for di in range(max_length):
            decoder_output, dec_h, decoder_attention = decoder(dec_inp, dec_h, enc_out)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)

            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            dec_inp = topi.squeeze(0).detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, device, n=10):
    '''
    We can evaluate random sentences from the training set and print out the
    input, target, and output to make some subjective quality judgements:
    '''
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang, output_lang, device)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence, enc, dec, input_lang, output_lang, device):
    output_words, attentions = evaluate(enc, dec, input_sentence, input_lang, output_lang, device)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def save_model(enc, dec, PATH):
    torch.save(enc, PATH+'enc')
    torch.save(dec, PATH+'dec')

def load_model(PATH):
    enc = torch.load(PATH+'enc')
    enc.eval()
    dec = torch.load(PATH+'dec')
    dec.eval()
    return enc, dec

def checkpoint_save(epoch, enc, dec, enc_opt, dec_opt, PATH):
    torch.save({'epoch':epoch,
                'enc_state_dict': enc.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'enc_opt_state_dict': enc_opt.state_dict(),
                'dec_opt_state_dict': dec_opt.state_dict()}, PATH)

def load_checkpoint(init_enc, init_dec, init_enc_opt, init_dec_opt, PATH):
    checkpoint = torch.load(PATH)
    init_enc.load_state_dict(checkpoint['enc_state_dict'])
    init_dec.load_state_dict(checkpoint['dec_state_dict'])
    init_enc_opt.load_state_dict(checkpoint['enc_opt_state_dict'])
    init_dec_opt.load_state_dict(checkpoint['dec_opt_state_dict'])
    return init_enc, init_dec, init_enc_opt, init_dec_opt