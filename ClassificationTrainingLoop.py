hidden_size = 128
rollout = 30
n_layers = 1
lr = 0.003
epochs = 200
op_rollout = 200    # total num of characters in output test sequence
load_chk = False    # load weights from save_path directory to continue training
save_path = "./model/CharRNN.pth"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = data = open('data/classification/all_pytorch.txt', 'r').read()[:10000]
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print("----------------------------------------")
print("Data has {} characters, {} unique".format(data_size, vocab_size))
print("----------------------------------------")

# char to index and index to char maps
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# convert data from chars to indices
data = list(data)
for i, ch in enumerate(data):
    data[i] = char_to_ix[ch]

# data tensor on device
data = torch.tensor(data).to(device)
data = torch.unsqueeze(data, dim=1)


type = ['low-level', 'high-level'][0]
if type=='low-level':
    from _1_SimpleRNN.SimpleRNNs import LowLevelRNN_classifier as RNN
else:
    from _1_SimpleRNN.SimpleRNNs import HighLevelRNN_classifier as RNN

model = RNN(vocab_size, hidden_size, vocab_size, n_layers, rollout)
print(model)

# # model instance
# model = RNN(vocab_size, vocab_size, hidden_size, n_layers, rollout).to(device)

# load checkpoint if True
if load_chk:
    model.load_state_dict(torch.load(save_path))
    print("Model loaded successfully !!")
    print("----------------------------------------")

# loss function and optimiser
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=lr)

# training loop
for i_epoch in range(1, epochs+1):

    # random starting point (1st 100 chars) from data to begin
    data_ptr = np.random.randint(100)
    n = 0
    running_loss = 0
    hidden_state = model.init_hidden()

    while True:
        input_seq = data[data_ptr : data_ptr+rollout]
        target_seq = data[data_ptr+1 : data_ptr+rollout+1]

        # forward pass
        output, hidden_state = model(input_seq, hidden_state)

        # compute loss
        loss = loss_fn(torch.squeeze(output), torch.squeeze(target_seq))
        running_loss += loss.item()

        # compute gradients and take optimiser step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # update the data pointer
        data_ptr += rollout
        n += 1

        # if at end of data : break
        if data_ptr + rollout + 1 > data_size:
            break

    # print loss and save weights after every epoch
    print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, running_loss/n))
    # torch.save(model.state_dict(), save_path)

    # sample / generate a text sequence after every epoch
    data_ptr = 0
    hidden_state = model.init_hidden(1)

    # random character from data to begin
    rand_index = np.random.randint(data_size-1)
    input_seq = data[rand_index : rand_index+1]

    print("----------------------------------------")
    while True:
        # forward pass
        output, hidden_state = model(input_seq, hidden_state)

        # construct categorical distribution and sample a character
        output = F.softmax(torch.squeeze(output), dim=0)
        dist = Categorical(output)
        index = dist.sample()

        # print the sampled character
        print(ix_to_char[index.item()], end='')

        # next input is current output
        input_seq[0][0] = index.item()
        data_ptr += 1

        if data_ptr > op_rollout:
            break

    print("\n----------------------------------------")
