rollout = 12
h_dim = 32
lr = 0.003
n_layers = 2
batch_size = 1
plot_training = False
passes = 10000

example_data = ['simple waveform', 'Obama audio'][0]
if example_data=='simple waveform':
    from data.regression.SimpleSignals import *
    x_dim = 2
    y_dim = 1
    '''
    The function make_data produces:                                   _
        y: a high frequency cos wave plus a low frequency cts signal _/ \_   _
                                                                          \_/
        x: the low freq signal and the phase of the cos wave at each timestep.
    '''
elif example_data=='Obama audio':
    from data.regression.AudioLoader import *
    from data.regression.SimpleSignals import init_loop_plot, loop_plot, close_loop_plot
    x_dim = 1
    y_dim = 1
    '''
    This data contains x as the previous y values, so it only has to predict one
    time step ahead.
    '''


torch.manual_seed(1)

algo = ['simplernn-low-level',
        'simplernn-mid-level',
        'simplernn-high-level',
        'stackedrnn',
        'gru-simplified-low-level',
        'gru-low-level',
        # 'gru-mid-level',
        'gru-high-level',
        'lstm-low-level',
        # 'lstm-mid-level',
        'lstm-mid-level_peephole',
        'lstm-high-level',
        'dilated rnn'
        ][2]

print(algo)
if algo=='simplernn-low-level':
    from _1_SimpleRNN.SimpleRNNs import LowLevelRNN as RNN
elif algo=='simplernn-mid-level':
    from _1_SimpleRNN.SimpleRNNs import MidLevelRNN as RNN
elif algo=='simplernn-high-level':
    from _1_SimpleRNN.SimpleRNNs import HighLevelRNN as RNN
elif algo=='stackedrnn':
    from _2_DeepRNN.StackedHiddenRNN import StackedRNN as RNN
elif algo=='gru-simplified-low-level':
    from _3_GRU.GRUs import LowLevelGRU_simplified as RNN
elif algo=='gru-low-level':
    from _3_GRU.GRUs import LowLevelGRU as RNN
# elif algo=='gru-mid-level':
#     from _3_GRU.GRUs import MidLevelGRU as RNN
elif algo=='gru-high-level':
    from _3_GRU.GRUs import HighLevelGRU as RNN
elif algo=='lstm-low-level':
    from _4_LSTM.LSTMs import LowLevelLSTM as RNN
# elif algo=='lstm-mid-level':
#     from _4_LSTM.LSTMs import MidLevelLSTM as RNN
elif algo=='lstm-mid-level_peephole':
    from _4_LSTM.LSTMs import MidLevelLSTM_peephole as RNN# not working properly for some reason
elif algo=='lstm-high-level':
    from _4_LSTM.LSTMs import HighLevelLSTM as RNN
elif algo=='dilated rnn':
    from _5_DilatedRNN.DilatedRNNs import DilatedRNN as RNN
    rollout, lr, n_layers, passes = 10, 3e-3, 2, 10000

model = RNN(x_dim, h_dim, y_dim, n_layers, rollout)
print(model)

optimiser = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.MSELoss()

h_state = model.init_hidden()

if plot_training: init_loop_plot()

pbar = tqdm(range(passes))
for step in pbar:
    x, y = make_data(step, batch_size, rollout=rollout)

    y_hat, h_state = model(x, h_state)
    h_state = model.break_connection(h_state)# detach the hidden state, break the connection from last rollout

    loss = loss_func(y_hat, y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.set_description('Loss: %.6f'%loss.data.item())
    if plot_training: loop_plot(step, t, y, y_hat)
if plot_training: close_loop_plot()

def generate(model, time_steps):
    init_loop_plot('Feeding input x to the model, continuously predicting y')

    model.eval()

    h_state = model.init_hidden()
    with torch.no_grad():
        for step in range(time_steps):
            x, y, t = make_data(step, batch_size, rollout=rollout, time=True)

            y_hat, h_state = model(x, h_state)

            loop_plot(step, t, y, y_hat)
    close_loop_plot()

print('Generation')
generate(model, 300)
