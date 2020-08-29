from data.regression.SimpleSignals import *

# setting rollout to none in the simplesignals file generates a 
# random rollout for each datapoint, making it much slower to 
# learn but a more powerful model
rollout = 10 

algo = ['mlp rnn', 'skip rnn', 'res rnn'][2]
if algo=='mlp rnn':
    from _2_DeepRNN.MLP_RNN import FullMLP_RNN
    x_dims = [2, 6]
    h_dims = [20, 20]
    h_dims = [h_dims[-1]+x_dims[-1]] + h_dims
    y_dims = [h_dims[-1], 1]

    model = FullMLP_RNN(x_dims, h_dims, y_dims)
elif algo=='skip rnn': 
    from _2_DeepRNN.MLP_RNN import MySkipRNN
    in_dim = 2
    out_dim = 1
    h1_dim, p1 = 15, 6
    h2_dim, p2 = 12, 4
    depth_h1 = 2

    model = MySkipRNN(in_dim, out_dim, h1_dim, h2_dim, p1, p2, depth_h1)
elif algo=='res rnn': 
    from _2_DeepRNN.MLP_RNN import ResRNN
    in_dim = 2
    out_dim = 1
    h_dim = 20

    model = ResRNN(in_dim, h_dim, out_dim)
print(model)

lr = 0.0003
n_layers = 1
batch_size = 1
plot_training = False
passes = 10000

optimiser = torch.optim.Adam(model.parameters(), lr=lr)
loss_func = nn.MSELoss()

h_state = model.init_hidden()#None# init

if plot_training: init_loop_plot()

pbar = tqdm(range(passes))
for step in pbar:
    x, y = make_data(step, batch_size, rollout=rollout)

    y_hat, h_state = model(x, h_state)
    h_state = h_state.detach()# detach the hidden state, break the connection from last rollout

    loss = loss_func(y_hat, y)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    pbar.set_description('Loss: %.4f'%loss.data.item())
    if plot_training: loop_plot(step, t, y, y_hat)
if plot_training: close_loop_plot()

def generate(model, time_steps):
    init_loop_plot('Feeding input x to the model, continuously predicting y')

    model.eval()

    h_state = model.init_hidden()  # for initial hidden state
    with torch.no_grad():
        for step in range(time_steps):
            x, y, t = make_data(step, batch_size, rollout=rollout, time=True)

            y_hat, h_state = model(x, h_state)

            loop_plot(step, t, y, y_hat)
    close_loop_plot()

print('Generation')
generate(model, 300)