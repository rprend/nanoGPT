# We're redoing everything from the batchnorm example, but this time we're "torchifying" it
# by organizing the code into classes and functions.

import torch

class Linear:
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in ** 0.5
    self.bias = torch.zeros(fan_out) if bias else None

  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias

    return self.out

  def parameters(self):
    return [self.weight] + ([self.bias] if self.bias is not None else [])


class BatchNorm1d:
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    # Note that this layer behaves differently when we are training vs running evaluation.
    self.training = True
    # Parameters. Trained with backprop.
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)

  def __call__(self, x):
    if self.training:
      mean = x.mean(dim=0)
      var = x.var(dim=0)
      # Update running mean and variance.
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
    else:
      mean = self.running_mean
      var = self.running_var

    xhat = (x - mean) / torch.sqrt(var + self.eps)
    self.out = self.gamma * xhat + self.beta
    return self.out

  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out

  def parameters(self):
    return []


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


n_embd = 10
n_hidden = 100
g = torch.Generator().manual_seed(42)

C = torch.randn((vocab_size, n_embd), generator=g)

# We're making a 6-layer multi-layer perceptron.
layers = [
  Linear(n_embd * block_size, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, n_hidden), BatchNorm1d(n_hidden), Tanh(),
  Linear(n_hidden, vocab_size), BatchNorm1d(vocab_size)
]

with torch.no_grad():
  # Make the last layer have a smaller weight.
  # layers[-1].weight *= .1
  layers[-1].gamma *= .1
  # All the rest apply a gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      # Because we're using tanh activation functions.
      layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print("Number of parameters:", sum(p.nelement() for p in parameters))
for p in parameters:
  p.requires_grad = True

max_steps = 20000
batch_size = 32
lossi = []
ud = []

for step in range(max_steps):
  # Mini-batch.
  indices = torch.randint(len(Xtr), (batch_size,), generator=g)
  Xbatch, Ybatch = Xtr[indices], Ytr[indices]

  # Forward pass.
  x = C[Xbatch]
  x = x.view(-1, n_embd * block_size) # Concatenate the embeddings.
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Ybatch)

  # Backward pass.
  for layer in layers:
    layer.out.retain_grad()
  for p in parameters:
    p.grad = None
  loss.backward()

  # Update.
  lr = 0.1 if step < 10000 else 0.01
  for p in parameters:
    p.data += -lr * p.grad

  # Track stats.
  if step % 1000 == 0:
    print("Step", step, "Loss", loss.item())
  with torch.no_grad():
    ud.append([(lr*p.grad.std() / p.data.std()).log10().item() for p in parameters])
  lossi.append(loss.log10().item())

  if step > 1000:
    break

# Visualize histograms of the forward pass activations. What we want to look at is to see how many of our
# activations are saturated at close to -1 or 1. We can see that the saturation stabilizes because we initialize our
# gain to 5/3. This came out empirically from the Kaiming initialization. If the gain is set muuch higher, we will
# saturate the activations and the model will not learn. If the gain is set much lower, we will not saturate the
# activations and the model will not learn.
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 4))
legends = []
for (i, layer) in enumerate(layers[:-1]): # Skip the last layer.
  if isinstance(layer, Tanh):
    t = layer.out
    print('Layer', i, 'Mean:', t.mean().item(), 'Std:', t.std().item())
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[1:].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('Tanh activations distribution')
# plt.show()

# Now we do the same with the gradients. Notice again that we have to very carefully set the gain (since we
# have no batchnorm), in order to avoid saturation.
plt.figure(figsize=(20, 4))
legends = []
for (i, layer) in enumerate(layers[:-1]): # Skip the last layer.
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('Layer', i, 'Mean:', t.mean().item(), 'Std:', t.std().item())
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[1:].detach(), hy.detach())
    legends.append(f'Layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('Tanh gradients distribution')
# plt.show()

# Now let's visualize all of the 2 dimensional parameters
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('Parameter', i, 'Shape:', tuple(p.shape), 'Mean:', p.mean().item(), 'Std:', p.std().item())
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[1:].detach(), hy.detach())
    legends.append(f'Parameter {i} {tuple(p.shape)}')

plt.legend(legends)
plt.title('Parameter gradients distribution')
# plt.show()

# Update to data ratio over time
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append(f'Parameter {i} {tuple(p.shape)}')

plt.plot([0, len(ud)], [-3, -3], 'k')
plt.legend(legends)
# plt.show()