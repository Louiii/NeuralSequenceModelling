import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from data.classification.text_loader import TextLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


hidden_size = 128
rollout = 30
n_layers = 1
lr = 0.003
epochs = 200
op_rollout = 200    # total num of characters in output test sequence
load_chk = False    # load weights from save_path directory to continue training
save_path = "./model/CharRNN.pth"

make_data = TextLoader(rollout=rollout)

algo = ['simplernn-low-level',
        'simplernn-high-level',
        'gru-high-level',
        'lstm-high-level'
        ][3]

print(algo)
if algo=='simplernn-low-level':
    from A1_SimpleRNN.SimpleRNNs import LowLevelRNN_classifier as RNN
elif algo=='simplernn-high-level':
    from A1_SimpleRNN.SimpleRNNs import HighLevelRNN_classifier as RNN
elif algo=='gru-high-level':
    from A3_GRU.GRUs import HighLevelGRU_classifier as RNN
elif algo=='lstm-high-level':
    from A4_LSTM.LSTMs import HighLevelLSTM_classifier as RNN

model = RNN(make_data.vocab_size, hidden_size, make_data.vocab_size, n_layers, rollout)
print(model)

if load_chk:
    model.load_state_dict(torch.load(save_path))
    print("Model loaded successfully !!")
    print("----------------------------------------")

optimiser = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.CrossEntropyLoss()

def generate(model, time_steps):
    model.eval()

    h_state = model.init_hidden(1)

    # random character from data to begin
    t = np.random.randint(make_data.data_size-1)
    x = make_data.data[t : t+1]

    print("-"*40)
    with torch.no_grad():
        for _ in range(op_rollout):
            output, h_state = model(x, h_state)

            output = F.softmax(torch.squeeze(output), dim=0)
            dist = Categorical(output)
            index = dist.sample()

            print(make_data.ix_to_char[index.item()], end='')
            
            x[0,0] = index.item()# next input is current output
    print("\n","-"*40)

for i_epoch in range(1, epochs+1):
    n, tot_loss = make_data.data_size//rollout, 0
    h_state = model.init_hidden()

    for t in [rollout * i for i in range(n)]:
        x, y = make_data(t)

        output, h_state = model(x, h_state)
        h_state = model.break_connection(h_state)# detach the hidden state, break the connection from last rollout

        loss = loss_func(torch.squeeze(output), torch.squeeze(y))
        tot_loss += loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    print("Epoch: {0} \t Loss: {1:.8f}".format(i_epoch, tot_loss/n))
    # torch.save(model.state_dict(), save_path)

    generate(model, None)# sample a text sequence after every epoch
