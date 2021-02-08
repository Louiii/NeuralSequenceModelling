from tqdm import tqdm
from collections import Counter
from collections import defaultdict
import torch
import re
import string
from BytePairEncoding import bpe

class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    # TODO (@mttk): Populate classs with default values of special symbols
    UNK = '<unk>'

    def __init__(self, counter, max_size=None, min_freq=1, specials=('<unk>', '<pad>'),
                 vectors=None, unk_init=None, vectors_cache=None, specials_first=True):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary. Default: ['<unk'>, '<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: 'torch.zeros'
            vectors_cache: directory for cached vectors. Default: '.vector_cache'
            specials_first: Whether to add special tokens into the vocabulary at first.
                If it is False, they are added into the vocabulary at last.
                Default: True.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()
        self.unk_index = None
        if specials_first:
            self.itos = list(specials)
            # only extend max size if specials are prepended
            max_size = None if max_size is None else max_size + len(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        self.words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        self.words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in self.words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if Vocab.UNK in specials:  # hard-coded for now
            unk_index = specials.index(Vocab.UNK)  # position in list
            # account for ordering of specials, set variable
            self.unk_index = unk_index if specials_first else len(self.itos) + unk_index
            self.stoi = defaultdict(self._default_unk_index)
        else:
            self.stoi = defaultdict()

        if not specials_first:
            self.itos.extend(list(specials))

        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None


    def _default_unk_index(self):
        return self.unk_index

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __getstate__(self):
        # avoid picking defaultdict
        attrs = dict(self.__dict__)
        # cast to regular dict
        attrs['stoi'] = dict(self.stoi)
        return attrs

    def __setstate__(self, state):
        if state.get("unk_index", None) is None:
            stoi = defaultdict()
        else:
            stoi = defaultdict(self._default_unk_index)
        stoi.update(state['stoi'])
        state['stoi'] = stoi
        self.__dict__.update(state)

    # def __eq__(self, other):
    #     if self.freqs != other.freqs:
    #         return False
    #     if self.stoi != other.stoi:
    #         return False
    #     if self.itos != other.itos:
    #         return False
    #     if self.vectors != other.vectors:
    #         return False
    #     return True

    def __len__(self):
        return len(self.itos)

    def lookup_indices(self, tokens):
        indices = [self.__getitem__(token) for token in tokens]
        return indices

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1

def build_vocab_from_iterator(iterator, num_lines=None):
    """
    Build a Vocab from an iterator.

    Arguments:
        iterator: Iterator used to build Vocab. Must yield list or iterator of tokens.
        num_lines: The expected number of elements returned by the iterator.
            (Default: None)
            Optionally, if known, the expected number of elements can be passed to
            this factory function for improved progress reporting.
    """

    counter = Counter()
    with tqdm(unit_scale=0, unit='lines', total=num_lines) as t:
        for tokens in iterator:
            counter.update(tokens)
            t.update(1)

    word_vocab = Vocab(counter)

    num_merges = len(word_vocab)//10
    print('byte pair encoding with %d merges...'%num_merges, end='')
    bpe_vocab = bpe(vocab=word_vocab.stoi, num_merges=num_merges)
    print('done')

    print(bpe_vocab)

    print('len(bpe_vocab) = %d'%len(bpe_vocab))
    print('len(word_vocab) = %d'%len(word_vocab))

    return word_vocab, bpe_vocab
