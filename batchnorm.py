# We simply copy the multi - layer perceptron architecture here, and we will modify the 
# forward pass to include batch normalization and kaiming initialization.

# First open the dataset in names.txt
with open('names.txt', 'r', encoding='utf8') as file:
  text = file.read()

# Build the vocabulary.
words = text.split()
vocabulary = sorted(list(set(''.join(words).lower())))
stoi = { char: idx for idx, char in enumerate(vocabulary) }
stoi['^'] = len(stoi)
itos = { idx: char for idx, char in enumerate(vocabulary) }
itos[len(itos)] = '^'
vocab_size = len(stoi)

# Create a training set, using block_size characters to predict the next character output.
block_size = 3
X, Y = [], []
for w in words:
  context = [stoi['^']] * block_size
  for c in w.lower() + '^':
    i = stoi[c]
    X.append(context)
    Y.append(i)
    context = context[1:] + [i]

import torch

X = torch.tensor(X, dtype=torch.long)
Y = torch.tensor(Y, dtype=torch.long)

# Split the dataset into training and validation and test sets. 80, 10, 10.
train_size = int(0.8 * len(X))
valid_size = int(0.1 * len(X))
test_size = len(X) - train_size - valid_size
Xtr, Ytr = X[:train_size], Y[:train_size]
Xdev, Ydev = X[train_size:train_size + valid_size], Y[train_size:train_size + valid_size]
Xte, Yte = X[train_size + valid_size:], Y[train_size + valid_size:]

# Create a model.
# We want an embedding layer, a hidden tanh layer, and a softmax output layer.
import torch.nn as nn
from torch.nn import functional as F


# Start with the embedding layer, which is a matrix of size vocab_size x embedding_size.
embedding_size = 27
C = nn.Embedding(vocab_size, embedding_size)

# Use a gain of 5/3 because we are using a tanh activation function.
kaiming_gain = 5/3

# Then the hidden layer.
hidden_size = 300
W1 = torch.randn((embedding_size * block_size, hidden_size), requires_grad=True)
scaling_factor = (kaiming_gain / (embedding_size * block_size))**0.5
W1.data *= scaling_factor

# Don't need a bias term here because we are using batch normalization.
# b1 = torch.randn((hidden_size,), requires_grad=True)

# Here and above, we use the scaling factor to do a kaiming initialization of the weights.
W2 = torch.randn((hidden_size, vocab_size), requires_grad=True)
scaling_factor = (kaiming_gain / hidden_size)**0.5
W2.data *= scaling_factor

# Initialize the biases to 0.
b2 = torch.zeros((vocab_size,), requires_grad=True)

# BatchNorm parameters.
batchnorm_gain = torch.ones((1, hidden_size), requires_grad=True)
batchnorm_bias = torch.zeros((1, hidden_size), requires_grad=True)
# BatchNorm running parameters / buffers.
bnmean_running = torch.zeros((1, hidden_size))
bnstd_running = torch.ones((1, hidden_size))


params = [C.weight, W1, W2, b2, batchnorm_gain, batchnorm_bias]
print("Number of parameters:", sum(p.numel() for p in params))

# Ensure that requires_grad is set to True for all parameters.
for p in params:
  p.requires_grad = True

# Create batches.
batch_size = 32

learning_rates = [0.1, 0.01]
lossi = []
stepi = []

# Good practice to train on a higher learning rate and then on one 10x smaller.
for rate in learning_rates:
  # Forward pass.
  for i in range(30000):
    # Get a batch.
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))

    # Forward pass.
    emb = C(X[ix]) # batch_size, block_size, embedding_size
    # Linear layer.
    hPreact = emb.view(-1, embedding_size * block_size) @ W1 # + b1 # batch_size, hidden_size
    # Batch normalization layer. What this will do is ensure that every single neuron will be unit gaussian
    # on this batch.
    # ----------------------------
    # Estimate the mean and standard deviation of the entire training set.
    bnmeani = hPreact.mean(0, keepdim=True)
    bnstdi = hPreact.std(0, keepdim=True)
    hPreact = batchnorm_gain * (hPreact - bnmeani) / bnstdi + batchnorm_bias
    with torch.no_grad():
      # Update the running mean and standard deviation.
      bnmean_running = 0.99 * bnmean_running + 0.01 * bnmeani
      bnstd_running = 0.99 * bnstd_running + 0.01 * bnstdi
    # ----------------------------
    # Nonlinearity.
    H = torch.tanh(hPreact) # batch_size, 300
    logits = H @ W2 + b2 # batch_size, vocab_size
    loss = F.cross_entropy(logits, Ytr[ix])

    # Backward pass.
    for p in params:
      p.grad = None
    loss.backward()
    for p in params:
      p.data += -rate * p.grad

    # Track stats
    lossi.append(loss.item())
    stepi.append(i)

    if i % 10000 == 0:
      print("Loss: ", loss.item())

# Calibrate the batchnorm layer.
with torch.no_grad():
  # Pass the training set through
  emb = C(Xtr)
  embat = emb.view(emb.shape[0], -1)
  hpreact = embat @ W1 # + b1

  # Measure the mean and standard deviation of the entire training set.
  bnmean = hpreact.mean(0, keepdim=True)
  bnstd = hpreact.std(0, keepdim=True)


# Demonstrate that bnmean_running and bnstd_running are close to bnmean and bnstd.
print("avg difference bnmean_running - bnmean: ", torch.mean(bnmean_running - bnmean))
print("avg difference bnstd_running - bnstd: ", torch.mean(bnstd_running - bnstd))

# Training loss
emb = C(Xtr)
embat = emb.view(emb.shape[0], -1) # Concat into (N, block_size * num_embeddings)
hpreact = embat @ W1 # + b1
hpreact = batchnorm_gain * (hpreact - bnmean) / bnstd + batchnorm_bias
H = torch.tanh(hpreact)
logits = H @ W2 + b2
loss = F.cross_entropy(logits, Ytr)

print("Training loss: ", loss.item())