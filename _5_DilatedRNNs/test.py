import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class DRNN(nn.Module):

    def __init__(self, num_features, n_hidden, n_layers, dropout=0):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]


        cell = nn.LSTM
        layers = nn.ModuleList([cell(num_features, n_hidden, dropout=dropout, batch_first=True)])
        for i in range(n_layers-1):
            layers += [cell(n_hidden, n_hidden, dropout=dropout, batch_first=True)]

        self.cells = layers

    # Initially, no hidden layer. After every output we pass the output
    # hidden layer to the next input (if stateful, if not we reset).
    # hidden is a list of hidden states for each layer I think.
    # hidden[0] is hidden state of layer 0, hidden[1] is hidden state of
    # layer 1, etc. For the very first (batch of) inputs, we don't have an
    # initial hidden state to feed in. After our first bat
    def forward(self, inputs, hidden=None):
        outputs = []

        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            # If we don't input a hidden state (i.e it must be the various
            # first batch), then we don't care about the final hidden state
            # value. The hidden state for each layer will be all 0s initially.
            if hidden is None:
                print("Dilation:", dilation)
                inputs, outs = self.drnn_layer(cell, inputs, dilation)
                print("Output Size", inputs.size())
                print("h_n Size", outs[0].size())
                print("c_n Size", outs[1].size())
                print()

                # inputs = (seq_len, batch, hidden_size) = (8, 4, 6)
                # outs[0] = outs[1] = (num_layers, batch, hidden_size) = (1,
                # 4, 6)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation,
                                                    hidden[i])

            # inputs contains outputs which is of dimension:
            # [seq_len, batch_size, hidden_size]
            # we then append final 1, 2, 4, 8, 16, etc outputs
            outputs.append(inputs[:,-dilation:,:])

        return inputs, outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        '''
        x:        {batch,    time_step, input_size}
        h_state:  {n_layers, batch,     hidden_size}
        f_out:    {batch,    time_step, hidden_size}
        '''
        batch_size, n_steps, _ = inputs.size()
        hidden_size = cell.hidden_size

        # Pad the inputs to ensure that the sequence length is a multiple of the dilation
        inputs, _ = self._pad_inputs(inputs, n_steps, rate)

        # Convert the inputs into the correct dilated sequences. e.g for dilation = 4
        # [0, 1, 2, 3, 4, 5, 6, 7] to:
        # [0, 4], [1, 5], [2, 6], [3, 7]
        dilated_inputs = self._prepare_inputs(inputs, rate)

        # Input the values into the layer. If the layer is the first layer
        # we must initialise the hidden layer, if not then we don't need to.
        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        # Split the outputs up back into the correct shape, and remove any
        # padding.
        print("Dilated Output Size", dilated_outputs.size())
        print("Dilated h_n Size", hidden[0].size())
        print("Dilated c_n Size", hidden[1].size())
        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        # splitted_hn = self._split_outputs(hidden[0], rate)
        # hidden = (splitted_hn, hidden[1])

        # Return the outputs (usual outputs, including all hidden layers),
        # and the hidden layer tuple (h_n, c_n).
        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        # If we are inputting into the initial layer
        if hidden is None:
            # Initialise the c, h layers with zeros. We have gone from
            # seq_len * batch size to (seq_len/rate) * (batch_size * rate)
            # for our input, so we need to initialise our hidden state
            # accordingly
            hidden = self.init_hidden(batch_size * rate, hidden_size)
        # Input the values, and the hidden state (either 0s or previous
        # hidden state), and get the outputs
        print('\n\n')
        print(dilated_inputs.size())
        print('\n\n')
        print(hidden[0].size())
        print(hidden[1].size())
        print('\n\n')
        dilated_outputs, hidden = cell(dilated_inputs, hidden)
        return dilated_outputs, hidden

    # Simply remove the padding by taking the first seq_len values from the
    # output.
    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    # Split the outputs of our dilated RNN back into the correct size
    # of [seq_len, batch_size, hidden_size]. Currrently:
    # dilated outputs.size = (new_seq_len, new_batch_size, hidden_size)
    # = [seq_len/rate, batch_size * rate, hidden_size]
    def _split_outputs(self, dilated_outputs, rate):
        # Get the original batch_size = the current batch size // rate
        # = batch_size * rate // rate
        batchsize = dilated_outputs.size(1) // rate

        # Divide our new bigger batch size into old batch size chunks
        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        # stack() without an axis specified concatenates by adding a new axis
        # transpose(1, 0) swaps the dimensions of axis 1 and axis 0
        # A tensor of the same shape as torch.stack((blocks).transpose(1,
        # 0) would have a different memory layout to what it has here,
        # hence we need to call contiguous to create a copy with the correct
        # underlying memory arrangement. spooky

        # Stack and tranpose them to re-interleave
        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()

        # Now we convert back to [seq_len/rate * rate,  batch_size *
        # rate/rate, hidden_size)
        # = [seq_len, batch_size, hidden_size]
        interleaved = interleaved.view(
            dilated_outputs.size(0) * rate,
            batchsize,
            dilated_outputs.size(2)
        )
        return interleaved

    # This function extends each sequence so that its length is a multiple
    # of the dilation, by padding it with 0s - possibly try padding with the
    # mean values?
    def _pad_inputs(self, inputs, n_steps, rate):
        # If window_size (seq_len) divides dilation exactly
        # 48 / 8 = 6 e.g,
        is_even = (n_steps % rate) == 0

        # Pad if necessary
        if not is_even:
            # eg :
            # rate (dilation) = 32
            # dilated_steps = 48 // 32 + 1 = 1 + 1 = 2
            dilated_steps = n_steps // rate + 1

            # zeros.shape = [2 * 32 - 48, batch_size, num_features]
            # = [16, batch_size, num_features]
            zeros_ = torch.zeros(inputs.size(0),
                                 dilated_steps * rate - inputs.size(0),
                                 inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            # Concatenate sequence with the zeros
            inputs = torch.cat((inputs, zeros_), dim=1)
        
        else:# No need to pad
            dilated_steps = n_steps // rate

        # Return possibly padded input, and the number of dilation steps
        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[:, j::rate, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(1, batch_size, hidden_dim)
        memory = torch.zeros(1, batch_size, hidden_dim)
        return (hidden, memory)
        # else:
        #     return hidden

n_input = 3
n_hidden = 6
n_layers = 4
batch_size = 4
seq_size = 8

model = DRNN(n_input, n_hidden, n_layers)

x1 = torch.randn(batch_size, seq_size, n_input)
x2 = torch.randn(batch_size, seq_size, n_input)

out, hidden = model(x1)
