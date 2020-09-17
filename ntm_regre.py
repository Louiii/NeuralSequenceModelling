from tqdm import tqdm
import torch
from torch import nn, optim
import numpy as np

from C1_NeuralTuringMachine.NTM import NTM
from data.regression.SimpleSignals import *

def clip_grads(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters: p.grad.data.clamp_(-10, 10)

controller_size = 32
controller_layers = 1
n_heads = 1
n_locations = 40
vec_size = 5
num_batches = 1000
batch_size = 1
x_dim = 2
y_dim = 1

print_steps = 200


model = NTM(x_dim, controller_size, y_dim, controller_layers, n_heads, n_locations, vec_size)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.003)

print("Total number of parameters: %d"%model.n_params())

losses, costs = [], []
mean_loss = mean_cost = None
pbar = tqdm(range(1, num_batches+1))
for step in pbar:
    X, Y = make_data(step, batch_size)
    model.init_sequence(batch_size)

    y_hat = model(X, final_nonlin=False)

    loss = criterion(y_hat, Y)
    optimiser.zero_grad()
    loss.backward()
    clip_grads(model)
    optimiser.step()

    loss = loss.item()
    losses.append(loss)

    pbar.set_description('loss: %.3f, mean loss: %.4f'%(loss, mean_loss) if mean_loss is not None else 'loss: %.3f'%loss)
    if step % print_steps==0:
        mean_loss = sum(losses)/print_steps
        losses = []

def generate(model, time_steps):
    init_loop_plot('Feeding input x to the model, continuously predicting y')

    model.eval()

    with torch.no_grad():
        for step in range(time_steps):
            X, Y, t = make_data(step, batch_size, time=True)

            model.init_sequence(batch_size)

            y_hat = model(X, final_nonlin=False)

            loop_plot(step, t, Y, y_hat)
    close_loop_plot()

print('Generation')
generate(model, 300)
