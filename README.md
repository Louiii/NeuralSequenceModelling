# Landmark Algorithms in Neural Sequence Modelling


# TODO:

> ... dilated LSTMs

> Wavenet

> Attention

> Transformer

> Organise classification training loop to best fit all types of data!

> Organise whole file system to best fit with encoder-decoder architectures.

> Use teacher forcing

> Download DNA datasets


- Leaky units?

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
       h^1_0    h^2_0     h^m_0
        |        |         |
	    v        v         v
x_0 -> h^1_1 -> h^2_1 ... h^m_1 -> y^_1
        |        |         |
	    v        v         v
x_1 -> h^1_2 -> h^2_2 ... h^m_2 -> y^_2
:       |        |     :   |
:       v        v     :   v
x_n -> h^1_n -> h^2_n ... h^m_n -> y^_n


Finally, I also consider ResNet blocks...

## My classification of two different depth networks mapping [x; h] -> h

Let the hidden state be split in two: H = [H1; H2]

A standard RNN maps [X; H1; H2] -> [H1; H2]

Let the network mapping [..] -> H1 be called f1, and the network mapping [..] -> H2 be called f2.

If f1 is some function of just H1 and X, and f2 is some function of just H2 and X, then this is equivalent to having two RNNs. 

Now assume f1 is deeper than f2. For example consider f2 as a single layer, this can be considered a skip connection.

To simpify things further assume f1 and f2 are a function of X, and the hidden state they map to (and some of the hidden states).

[--- H1 ---][--- X ---][--- H2 ---]
     |       /       \      |
     v      v         v     v
[--- H1 ---]           [--- H2 ---]

Finally, let p1 be a number specifying how many units from H1 and p2 from H2. The RNN I define has f1 : [H1; H2[:p2]; X] -> H1 and f2 : [H2; H1[:p1]; X] -> H2.

# GRU



# Attention

this is the first instance of attention that sparked the revolution - additive attention (also known as Bahdanau attention) proposed by Bahdanau et al.
