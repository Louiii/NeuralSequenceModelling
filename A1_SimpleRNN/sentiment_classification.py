import json, string, random
import torch
import torch.nn as nn
from tqdm import tqdm

with open('../data/classification/imdb_data.json') as f:
    data = json.load(f)

# all_characters = string.ascii_letters + " .,;'"
# n_letters = len(all_characters)
# n_examples = len(data)
# n_categories = 2
# all_categories = {0:'negative', 1:'positive'}
#
# charToIndex = lambda char: all_characters.find(char)
#
# def seqToTensor(seq):
#     tensor = torch.zeros(len(seq), 1, n_letters)
#     for li, char in enumerate(seq):
#         tensor[li][0][charToIndex(char)] = 1
#     return tensor
#
# def randomTrainingExample():
#     i = random.randint(0, n_examples - 1)
#     [x, y] = data[i]
#     category_tensor = torch.tensor([y], dtype=torch.long)
#     seq_tensor = seqToTensor(x)
#     return y, x, category_tensor, seq_tensor
#
# def categoryFromOutput(output):
#     top_n, top_i = output.topk(1)
#     category_i = top_i[0].item()
#     return all_categories[category_i], category_i





words, data = data[0], data[1:]
n_words = len(words)
print(n_words)
# print(words)

wordToIndex = dict()
for i, word in enumerate(words):
    wordToIndex[word] = i

n_examples = len(data)
n_categories = 2
all_categories = {0:'negative', 1:'positive'}

def seqToTensor(seq):
    # tensor = torch.zeros(len(seq), 1, n_words)
    # for li, char in enumerate(seq):
    #     tensor[li][0][wordToIndex[char]] = 1
    tensor = torch.zeros(len(seq), 1, 1, dtype=torch.long)
    for li, char in enumerate(seq):
        tensor[li, 0, 0] = wordToIndex[char]
    return tensor

def randomTrainingExample():
    i = random.randint(0, n_examples - 1)
    [x, y] = data[i]
    category_tensor = torch.tensor([y], dtype=torch.long)
    # print(x)
    seq_tensor = seqToTensor(x.split(' '))
    return y, x, category_tensor, seq_tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


# '''
# This is a character-level RNN which takes in an arbitrary length sequence and
# determines whether the sequence (review) had a positive or negative sentiment.
# The sequences are movie reviews from imdb.
# '''
class RNN(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
        self.h_dim, self.n_layers = hidden_size, n_layers
        self.embedding = nn.Embedding(embed_size, input_size)
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedding, hidden_state)
        output = self.decoder(output)
        return output, (hidden_state[0].detach(), hidden_state[1].detach())#hidden_state.detach()

    def init_hidden(self, length=None):
        return None
        # if length==None:
        #     return torch.zeros(self.n_layers, self.rollout, self.h_dim)
        # return torch.zeros(self.n_layers, length, self.h_dim)
#
# # class RNN(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super(RNN, self).__init__()
# #         self.hidden_size = hidden_size
# #         self.xh = nn.Linear(input_size + hidden_size, hidden_size)
# #         self.xy = nn.Linear(hidden_size, output_size)
# #         self.softmax = nn.LogSoftmax(dim=1)
# #
# #     def forward(self, input, hidden):
# #         combined = torch.cat((input, hidden), 1)
# #         hidden = self.xh(combined)
# #         return hidden
# #
# #     def predict(self, hidden):
# #         output = self.xy(hidden)
# #         output = self.softmax(output)
# #         return output
# #
# #     def initHidden(self):
# #         return torch.zeros(1, self.hidden_size)
#
n_hidden = 50
learning_rate = 0.001

rnn = RNN(input_size=200,
          embed_size=n_words,
          hidden_size=n_hidden,
          output_size=2,
          n_layers=1)
optimiser = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, seq_tensor):
    hidden = rnn.init_hidden()

    # rnn.zero_grad()

    for i in range(seq_tensor.size()[0]):
        y_hat, hidden = rnn(seq_tensor[i], hidden)
    # y_hat = rnn.predict(hidden)
    y_hat = torch.squeeze(y_hat, 0)
    # print('y_hat: '+str(y_hat.size())+', category_tensor: '+str(category_tensor.size()))
    # print('y_hat: '+str(y_hat)+', category_tensor: '+str(category_tensor))
    loss = criterion(y_hat, category_tensor)
    loss.backward()
    optimiser.step()
    optimiser.zero_grad()

    return y_hat, loss.item()


n_iters = 10000
print_every = 10
plot_every = 10



# Keep track of losses for plotting
current_loss = 0
all_losses = []
acc = 0

pbar = tqdm(range(1, n_iters + 1))
for iter in pbar:
    category, line, category_tensor, seq_tensor = randomTrainingExample()
    output, loss = train(category_tensor, seq_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess_i==category else '✗ (%s)' % category
        n = iter // print_every
        acc = (guess_i==category)/n + acc*(n-1)/n
        pbar.set_description('Acc: %.2f, Loss: %.4f / %s %s' % (acc, loss, guess, correct))
        print(line)

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
