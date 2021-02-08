import random, torch
from torch.utils.data import Dataset


def rdm_op(difficulty):
    i=random.randint(0, difficulty)
    if i==0:
        return 'add ', ' and ', lambda x, y:x+y, ''
    elif i==1:
        return 'subtract ', ' from ', lambda x, y:y-x, ''
    elif i==2:
        return 'multiply ', ' by ', lambda x, y:x*y, 'with '
    return 'divide ', ' by ', lambda x, y:x/y, 'by '

def rdm_int():
    i=10**random.randint(0, 2)
    a = random.randint(0, 10*i)-5*i
    return a + 1 if a == 0 else a

def generate_data(n, tr_split=0.8, difficulty=3, extra=True):
    xs, ys = [], []
    for _ in range(n):
        a, b = rdm_int(), rdm_int()
        op, wrd, fn, _ = rdm_op(difficulty)
        x = op + str(a) + wrd + str(b)
        y = fn(a, b)
        if extra and random.randint(0, 1):
            c = rdm_int()
            op, wrd, fn, wrd2 = rdm_op(difficulty)
            x += ', then ' + op + wrd2 + str(c)
            y = fn(y, c)
        y = '%.4f'%y if type(y) is float else str(y)
        xs.append(x)
        ys.append(y)
    xys = list(set(list(zip(xs, ys))))
    i = int(len(xys)*0.8)
    tr_xys, ts_xys = xys[:i], xys[i:]
    return zip(*tr_xys), zip(*ts_xys)

class CharDataset(Dataset):
    def __init__(self, xs, ys, ts_xs, ts_ys, block_size):
        chars = sorted(list(set(':'.join(xs+ts_xs)+'_'.join(ys+ts_ys))))

        data_size, vocab_size = len(xs), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        self.block_size = block_size
        self.vocab_size = vocab_size

        x_cat = ''
        for x,y in zip(xs,ys): x_cat += x + ':' + y + '_'

        self.data_pretrain = x_cat
        self.xs = xs
        self.ys = ys

        self.ts_xs = ts_xs
        self.ts_ys = ts_ys
        self.n_test = len(ts_xs)

        self.selfsupervised = True

    def rnd_ts(self):
        return self.ts_xs[random.randint(0, self.n_test)]

    def s2i(self, s):
        if s in self.stoi:
            return self.stoi[s]
        return len(self.stoi)

    def i2s(self, i):
        i = int(i)
        if i in self.itos:
            return self.itos[i]
        return '?'
    
    def __len__(self):
        if self.selfsupervised:
            return len(self.data_pretrain) - self.block_size
        return len(self.xs)

    def __getitem__(self, idx):
        if self.selfsupervised:
            chunk = self.data_pretrain[idx:idx + self.block_size + 1]
            dix = [self.s2i(s) for s in chunk]
            x = torch.tensor(dix[:-1], dtype=torch.long)
            y = torch.tensor(dix[1:], dtype=torch.long)
            return x, y

        dix = [self.s2i(s) for s in self.xs[idx]]#+ [self.s2i(':')]
        dix += [self.s2i(':')]*(self.block_size//2 - len(dix))
        x = torch.tensor(dix, dtype=torch.long)

        dix = [self.s2i(s) for s in self.ys[idx]]
        dix += [self.s2i('_')]*(self.block_size//2 - len(dix))
        y = torch.tensor(dix, dtype=torch.long)
        return x, y
