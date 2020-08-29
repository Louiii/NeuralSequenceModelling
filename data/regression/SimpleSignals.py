import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
nn = torch.nn

def x_signal(x):
    y = 0
    y += (x > 10 and x <= 20) * (x - 10) * 0.1
    y += (x > 20 and x <= 30)
    y += (x > 30 and x <= 40) * (40 - x) * 0.1
    y += (x > 60 and x <= 70) * (60 - x) * 0.1
    y -= (x > 70 and x <= 80)
    y += (x > 80 and x <= 90) * (x - 90) * 0.1
    return y * 0.5

v_x_signal = np.vectorize(x_signal)

def sequence(t):
    t = np.remainder(t, 100)
    return v_x_signal(t)#.astype(float)

def time_seq(step, rollout):
    start, end = step * np.pi *0.5, (step+1)*np.pi *0.5   # time range
    return np.linspace(start, end, rollout, dtype=np.float32, endpoint=False)

def make_data(step, batch_size, rollout=None, time=False):
    '''                                                            _
    y: a high frequency cos wave plus a low frequency cts signal _/ \_   _
                                                                      \_/
    x: the low freq signal and the phase of the cos wave at each timestep.
    '''
    # TODO; ACCOMODATE BATCH_SIZE
    if rollout==None: rollout = np.random.randint(5, 30)

    t = time_seq(step, rollout)
    sig = sequence(t)
    phases = np.mod(t, 2 * np.pi)
    # prev_cos = np.zeros(rollout)
    # prev_cos[1:] = 0.8 * np.cos(t)[:-1]# part of y signal

    x_np = np.vstack((phases, sig)).T#, prev_cos)).T
    y_np = 0.7 * np.cos(t) + sig

    x = torch.from_numpy(x_np[np.newaxis, :]).float()
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).float()

    if time: return x, y, t
    return x, y# x and y: {batch, time_step, input_size}

def make_sin_cos_data(t):
    sig = sequence(t)
    x_np = np.sin(t) + sig
    y_np = np.cos(t) + sig
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]).float()
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).float()
    return x, y# x and y: {batch, time_step, input_size}

def make_x_phase_data(t, rollout):
    phases = np.mod(t, 2 * np.pi)
    x_np = phases# ones/zeros doesn't work :/
    x_np = np.zeros(rollout)
    x_np[1:] = np.cos(t)[:-1]
    y_np = np.cos(t)
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis]).float()
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis]).float()
    return x, y




def init_loop_plot(title=None):
    plt.figure(1, figsize=(10, 5))
    plt.ion()
    plt.ylim(-1.5, 1.5)
    if title is not None: plt.title(title)

def loop_plot(step, t, y, y_hat):
    if step==0:
        plt.plot(t, y.numpy().flatten(), 'g.', ms=2, label='true')
        plt.plot(t, y_hat.data.numpy().flatten(), 'b.', ms=2, label='model')
        plt.legend()
    else:
        plt.plot(t, y.numpy().flatten(), 'g.', ms=2)
        plt.plot(t, y_hat.data.numpy().flatten(), 'b.', ms=2)
        if step%10==0: plt.xlim(max(t[-1]-80, 0), t[-1]+20)
    plt.draw(); plt.pause(0.02)

def close_loop_plot():
    plt.ioff()
    plt.show()
