import audioowl
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

file_path = 'data/regression/Obama.mp3'
waveform = audioowl.get_waveform(file_path, sr=22050)# sr=sample_rate
data_len = len(waveform)

def time_seq(step, rollout):
    start, end = step * rollout, (step+1) * rollout
    start, end = start % data_len, end % data_len
    if start > end:
        return np.array(list(range(start, data_len))+list(range(end)))
    return np.array(list(range(start, end)))

def make_data(step, batch_size, rollout=None, time=False):
    # TODO; ACCOMODATE BATCH_SIZE
    if rollout==None: rollout = np.random.randint(5, 30)

    t = time_seq(step, rollout)
    xt = t - 1
    x = torch.tensor(waveform[xt][np.newaxis, :, np.newaxis]).float()
    y = torch.from_numpy(waveform[t][np.newaxis, :, np.newaxis]).float()
    if time: return x, y, t
    return x, y
