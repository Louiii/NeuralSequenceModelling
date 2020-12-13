# Landmark Algorithms in Neural Sequence Modelling

# UNFINISHED, TODO:

- Finish wavenet 
- Sort multi-headed attention issue
- Build transformer
- Add protein dataset compatibility
- Write D section

> Organise classification training loop to best fit all types of data!
> Organise whole file system to best fit with encoder-decoder architectures.
> Use teacher forcing

## OTHER STUFF:

lstm - learning the initial state
highway layers/recurrent dropout
self-attention, key-value attention, masked attention
ensembling, mixture of experts
bayesian hyper param tuning: https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf
model coverage- dealing with repitition in attention systems
beam search: https://arxiv.org/pdf/1703.03906.pdf
SNAIL
SelfAttnGAN
PointerNetwork
Wavenet

```
.
├── data
│   ├── classification
│   │   ├── all_pytorch.txt
│   │   ├── copy_task.py
│   │   ├── eng_fra.txt
│   │   └── imdb_data.json
│   └── regression
│       ├── SimpleSignals.py
│       └── AudioLoader.py
│
│
├── _1_SimpleRNN
│   ├── SimpleRNNs.py
│   └── BidirectionalRNNs.py
├── _2_DeepRNN ── VariableLayerRNN.py
├── _3_GRU ── GRUs.py
├── _4_LSTM ── LSTMs.py
├── _5_NeuralTuringMachine ── NTM.py
├── _6_Wavenet ── WaveNet.py
├── _7_Attention ── Attention.py
├── _8_Transformer ── Transformer.py
│
│
├── ClassificationTrainingLoop.py
└── RegressionTrainingLoop.py
```

simpleRNN:-> have files to load and train each type of data (reg/clas, all/end), separate file for low level rnn class and high level rnn class.



RNNs are the adaptation of feedforward artificial neural networks to process arbitrary length sequence data.
I will label the sequence to predict as y_1:t = [y_1, y_2,.., y_t].
In some cases we may have an input at each time step, x_1:t = [x_1, x_2,..., x_t].

RNNs can be trained in many ways depending on the task. For typical sequence data and RNN will take in each item in a sequence (x_1:t) and predict the value (y^_i) at each time step (i=1:t). Another way is to take in a sequence (x_1:t) and only predict the final value (y^_t). The final standard method, best suited to tasks like translation, is to take in a whole sequence (x_1:t) and then produce (y^_1:t) afterwards.

# Datasets

### Classification

- Large Movie Review Dataset: http://ai.stanford.edu/~amaas/data/sentiment/

    <span style="color:grey">
This dataset contains movie reviews along with their associated binary sentiment polarity labels. The core dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. The overall distribution of labels is balanced (25k pos and 25k neg). The train and test sets contain a disjoint set of movies. In the labeled train/test sets, a negative review has a score <= 4 out of 10, and a positive review has a score >= 7 out of 10. Thus reviews with more neutral ratings are not included in the train/test sets.
</span>


- all_pytorch.txt

  <span style="color:grey">
This dataset
</span>


- eng_fra.txt

  <span style="color:grey">
This dataset
</span>


### Regression

- SimpleSignal.py: this file generates simple sequences

- AudioLoader.py



# 1 SimpleRNN

This folder contains two implementations of a simple recurrent neural network in PyTorch, one using the built in RNN class, another only using tensors.



# 2 DeepRNN

RNNs can be deepened in different ways, I explore these in the files MLP_RNN and StackedHiddenRNN. Of course an obvious way to improve an RNNs ability to model more complicated functions would be to add more layers to each mapping: X -> x, [x; h] -> h and h -> y_pred. However, generally it is avoided to deepen the step [x; h] -> h as it makes gradients vanish over less time steps. A way to mitigate this is using skip connections to decrease the distance of shortest path though the computational graph through time, I implement these two techniques in 'MPL_RNN.py'. I couldn't find very comprehensive information online for skip connections for RNNs, it can be done using addition operations or by concatenation. I will present a categorisation below of ways to add skip connections with concatenation for only two different depth feedforward neural networks. The most common way to deepen RNNs is to stack RNNs; this involves the first RNN mapping [x; h_rnn1] -> h_rnn1, the second doing [h_rnn1; h_rnn2] -> h_rnn2,.., with the m-th rnn computing [x; h_rnn{m-1}] -> h_rnnm -> y.
So,
       h^1_0    h^2_0     h^m_0<br />
        |        |         |<br />
	    v        v         v<br />
x_0 -> h^1_1 -> h^2_1 ... h^m_1 -> y^_1<br />
        |        |         |<br />
	    v        v         v<br />
x_1 -> h^1_2 -> h^2_2 ... h^m_2 -> y^_2<br />
:       |        |     :   |<br />
:       v        v     :   v<br />
x_n -> h^1_n -> h^2_n ... h^m_n -> y^_n<br />


Finally, I also consider ResNet blocks...

## My classification of two different depth networks mapping [x; h] -> h

Let the hidden state be split in two: H = [H1; H2]

A standard RNN maps [X; H1; H2] -> [H1; H2]

Let the network mapping [..] -> H1 be called f1, and the network mapping [..] -> H2 be called f2.

If f1 is some function of just H1 and X, and f2 is some function of just H2 and X, then this is equivalent to having two RNNs.

Now assume f1 is deeper than f2. For example consider f2 as a single layer, this can be considered a skip connection.

To simpify things further assume f1 and f2 are a function of X, and the hidden state they map to (and some of the hidden states).

[--- H1 ---][--- X ---][--- H2 ---]<br />
     |       /       \      |<br />
     v      v         v     v<br />
[--- H1 ---]           [--- H2 ---]<br />

Finally, let p1 be a number specifying how many units from H1 and p2 from H2. The RNN I define has f1 : [H1; H2[:p2]; X] -> H1 and f2 : [H2; H1[:p1]; X] -> H2.

# GRU


# LSTM


# Dilated RNN

https://arxiv.org/abs/1710.02224


# Neural Turing Machine


# Attention

Encoder-decoder model with additive attention mechanism: Bahdanau et al., 2015.<br />
   ╭-----╮   ╭-----╮<br />
...|s_t-1|-->| s_t |...  Decoder <br />
   ╰-----╯ / ╰-----╯<br />
           |<br />
           ⊕             Context vector<br />
         ⟋/| ⟍  <br />
       ⟋ / |   ⟍<br />
  a1 ⟋a2/a3|   an⟍      Alignment weights at time step t. These also depend on s_t-1<br />
   ⟋   /   |       ⟍<br />
┌--┐ ┌--┐ ┌--┐     ┌--┐ <br />
|h1|>|h2|>|h3|>... |hn|  Encoder: forward<br />
|g1|<|g1|<|g1|<... |g1|  Encoder: backward<br />
└--┘ └--┘ └--┘     └--┘<br />
 x1   x2   x3       xn<br />


The equations:<br />
x = [x1,..., xn]<br />
y = [y1,..., ym]<br />

Say Encoder is a bidirectional RNN, H_i = [h_i; g_i]<br />
Decoder has one direction, s_t = f(s_t-1, y_t-1, c_t)<br />

c_t = sum_i=1,..,n { a_{t,i} * h_i }<br />
a_{t,i} = align(y_t, x_i) = Softmax(score(s_{t-1}, h_i))<br />

score(s_t, h_i) = feedforward({dim(s)+dim(h),..., 1})<br />
e.g.            = v . tanh( W * [s_t; h_i] ) {params: v, W}<br />


The algorithm:<br />
1. Run Encoder, store h_i i=1,..,n<br />
2. for t in range(m):# while True: # and stop when EOF char is output<br />
       compute a_{t,i} for all i<br />
       compute context vector, c_t<br />
       take one step for the decoder with s_t-1 as hidden and c_t as input<br />
       produce y_t<br />

## Multi-headed Attention



# Transformer
