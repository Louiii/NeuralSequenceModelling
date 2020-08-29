import time, random
from tqdm import tqdm
from attr import attrs, attrib, Factory
import torch
from torch import nn, optim
import numpy as np

from _6_NeuralTuringMachine.NTM import NTM
from data.classification.copy_task import make_data

def clip_grads(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in parameters: p.grad.data.clamp_(-10, 10)

controller_size = 100
controller_layers = 1
n_heads = 1
sequence_width = 8
# sequence_min_len = 1
# sequence_max_len = 20
n_locations = 128
vec_size = 20
num_batches = 50000
batch_size = 1

print_steps = 200


model = NTM(sequence_width + 1, controller_size, sequence_width, 
            controller_layers, n_heads, n_locations, vec_size)
criterion = nn.BCELoss()
optimiser = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)



get_ms = lambda : time.time() * 1000
seed = int(get_ms() // 1000)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


print("Total number of parameters: %d"% model.n_params())

tot_loss, tot_cost = 0, 0
mean_loss = mean_cost = None
pbar = tqdm(range(1, num_batches))
for batch_num in pbar:
    X, Y = make_data(batch_num, batch_size)

    optimiser.zero_grad()

    outp_seq_len = Y.size(1)
    model.init_sequence(batch_size)

    y_out = model(X, outp_seq_len)

    loss = criterion(y_out, Y)
    loss.backward()

    clip_grads(model)
    optimiser.step()

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: x > 0.5)
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))# The cost is the number of error bits per sequence
    loss, cost = loss.item(), cost.item() / batch_size

    tot_loss += loss
    tot_cost += cost

    pbar.set_description('loss: %.2f, mean loss: %.2f, mean cost: %.1f'%(loss, mean_loss, mean_cost) if mean_loss is not None else 'loss: %.2f'%loss)
    if batch_num % print_steps==0:
        mean_loss, mean_cost = tot_loss/print_steps, tot_cost/print_steps
        tot_loss, costs = 0, 0


# def evaluate(model, criterion, X, Y):
#     """Evaluate a single batch (without training)."""
#     inp_seq_len = X.size(0)
#     outp_seq_len, batch_size, _ = Y.size()

#     # New sequence
#     model.init_sequence(batch_size)

#     # Feed the sequence + delimiter
#     states = []
#     for i in range(inp_seq_len):
#         o, state = model(X[i])
#         states.append(state)

#     # Read the output (no input given)
#     y_out = torch.zeros(Y.size())
#     for i in range(outp_seq_len):
#         y_out[i], state = model()
#         states.append(state)

#     loss = criterion(y_out, Y)

#     y_out_binarized = y_out.clone().data
#     y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

#     # The cost is the number of error bits per sequence
#     cost = torch.sum(torch.abs(y_out_binarized - Y.data))

#     result = {
#         'loss': loss.data.item(),
#         'cost': cost / batch_size,
#         'y_out': y_out,
#         'y_out_binarized': y_out_binarized,
#         'states': states
#     }
#     return result
