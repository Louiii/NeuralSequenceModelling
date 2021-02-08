import re, collections, json
from tqdm import tqdm

class BPE:
    def __init__(self, save_path):
        self.sorted_tokens_path = save_path+'/sorted-tokens'
        self.wrd2ixs_path = save_path+'/wrd2ixs'
        self.wrd2tok_path = save_path+'/wrd2tok'
        self.all_words = set([])
        self.wrd2ixs = None
        self.wrd2tok = None

    def get_vocab(self, train_string):
        vocab = collections.defaultdict(int)

        for line in train_string.split('\n'):

            words = line.strip().split()
            for word in words:
                self.all_words.add(word)
                vocab[' '.join(list(word)) + ' </w>'] += 1

        return vocab

    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def get_tokens_from_vocab(self, vocab):
        tokens_frequencies = collections.defaultdict(int)
        vocab_tokenisation = {}
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                tokens_frequencies[token] += freq
            vocab_tokenisation[''.join(word_tokens)] = word_tokens
        return tokens_frequencies, vocab_tokenisation

    def measure_token_length(self, token):
        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)

    def tokenize_word(self, string, sorted_tokens, unknown_token='</u>'):
        
        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace('.', '[.]'))

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            break
        return string_tokens

    def save(self, item, path):
        with open(path+'.json', 'w') as fp:
            json.dump(item, fp)

    def load(self, path):
        with open(path+'.json', 'r') as fp:
            item = json.load(fp)
        return item

    def generate_tokens(self, vocab, num_merges=10000):
        tokens_frequencies, vocab_tokenisation = self.get_tokens_from_vocab(vocab)
        print('\nGenerating tokens')
        for i in tqdm(range(num_merges)):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            tokens_frequencies, vocab_tokenisation = self.get_tokens_from_vocab(vocab)

        sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (self.measure_token_length(item[0]), item[1]), reverse=True)
        sorted_tokens = {token:i for i, (token, freq) in enumerate(sorted_tokens_tuple)}

        self.save(sorted_tokens, self.sorted_tokens_path)
        print(sorted_tokens)

    def wrd2tok_(self, key):
        if key in self.wrd2tok: 
            return self.wrd2tok[key]
        return []

    def wrd2ixs_(self, key):
        if key in self.wrd2ixs: 
            return self.wrd2ixs[key]
        return []

    def loadmaps(self):
        self.wrd2ixs = self.load(self.wrd2ixs_path)
        self.wrd2tok = self.load(self.wrd2tok_path)

    def encode(self, word, sorted_tokens, to_ix=False):
        self.loadmaps()

        if type(word)==list: return [self.wrd2ixs_(w) for w in word]
        if self.wrd2tok is not None: 
            if to_ix: return self.wrd2ixs_(word)
            return self.wrd2tok_(word)
        sorted_tokens = [token for token, i in sorted_tokens.items()]
        enc = self.tokenize_word(string=word+'</w>', sorted_tokens=sorted_tokens, unknown_token='</u>')
        if to_ix: return [sorted_tokens[tok] for tok in enc]
        return enc

    def precompute_encode_map(self):
        print('Computing BPE Encoding and Recording Map')
        sorted_tokens = self.load(self.sorted_tokens_path)
        sorted_tokens_lst = [token for token, i in sorted_tokens.items()]

        self.wrd2tok = {}
        self.wrd2ixs = {word:[] for word in self.all_words}
        for word in tqdm(self.all_words):
            enc = self.tokenize_word(string=word, sorted_tokens=sorted_tokens_lst, unknown_token='</u>')
            self.wrd2tok[word] = enc
            for tok in enc:
                self.wrd2ixs[word].append(sorted_tokens[tok])
        self.save(self.wrd2ixs, self.wrd2ixs_path)
        self.save(self.wrd2tok, self.wrd2tok_path)

    def decode(self, tokens, sorted_tokens_inv=None):
        if type(tokens[0])==list:# either list of ints or strings
            if type(tokens[0][0])==int:
                return [''.join([sorted_tokens_inv[i] for i in toks]) for toks in tokens]
            return [''.join(toks) for toks in tokens]
        if type(tokens[0])==int:
            return ''.join([sorted_tokens_inv[i] for i in tokens])
        return ''.join(tokens)



