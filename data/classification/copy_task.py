import random
import torch
import numpy as np

def make_data(step, batch_size):
    seq_width = 8
    seq_min_len = 1
    seq_max_len = 20

    seq_len = random.randint(seq_min_len, seq_max_len)
    seq = np.random.binomial(1, 0.5, (batch_size, seq_len, seq_width))
    seq = torch.from_numpy(seq)

    inp = torch.zeros(batch_size, seq_len + 1, seq_width + 1)# additional channel used for the delimiter
    inp[:, :seq_len, :seq_width] = seq
    inp[:, seq_len, seq_width] = 1.0 # delimiter in our control channel
    outp = seq.clone()

    return inp.float(), outp.float()
    # inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)# additional channel used for the delimiter
    # inp[:seq_len, :, :seq_width] = seq
    # inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
    # outp = seq.clone()

    # return inp.float(), outp.float()

# x, y = make_data(1)
# print(x.size())
# print(y.size())