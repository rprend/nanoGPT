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

# Then the hidden layer.
hidden_size = 300
W1 = torch.randn((embedding_size * block_size, hidden_size), requires_grad=True)
b1 = torch.randn((hidden_size,), requires_grad=True)

# Then the output layer.
W2 = torch.randn((hidden_size, vocab_size), requires_grad=True)
b2 = torch.randn((vocab_size,), requires_grad=True)

params = [C.weight, W1, b1, W2, b2]
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

    emb = C(X[ix]) # batch_size, block_size, embedding_size
    H = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1) # batch_size, 300
    logits = H @ W2 + b2 # batch_size, vocab_size
    loss = F.cross_entropy(logits, Ytr[ix])

    # print("Loss: ", loss.item())

    # Backward pass.
    for p in params:
      p.grad = None
    loss.backward()
    for p in params:
      p.data += -rate * p.grad

    # Track stats
    lossi.append(loss.item())
    stepi.append(i)

# Training loss
emb = C(Xtr)
H = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1)
logits = H @ W2 + b2
loss = F.cross_entropy(logits, Ytr)

print("Training loss: ", loss.item())

import matplotlib.pyplot as plt
plt.plot(stepi, lossi)
plt.show()

# Now check the dev loss.
emb = C(Xdev)
H = torch.tanh(emb.view(-1, embedding_size * block_size) @ W1 + b1)
logits = H @ W2 + b2
loss = F.cross_entropy(logits, Ydev)

print("Dev loss: ", loss.item())