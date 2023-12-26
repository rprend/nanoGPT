
# Let's create a basic bigram neural net with cross entropy loss and
# one-hot encoding.

# First, let's create a dataset.
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf8') as file:
  text = file.read()

# Create a training set of bigrams.
words = text.split()

# Encode character to a number with a stoi function.
chars = sorted(list(set(''.join(words).lower())))
stoi = { char: idx for idx, char in enumerate(chars) }
stoi['^'] = len(stoi)
itos = { idx: char for idx, char in enumerate(chars) }
itos[len(itos)] = '^'

xs, ys = [], []
for w in words:
  # Create a bigram from each character, with a . as the first character and the end.
  w = '^' + w.lower() + '^'
  for i in range(len(w) - 1):
    xs.append(stoi[w[i]])
    ys.append(stoi[w[i + 1]])

vocab_size = len(stoi)

xs = torch.tensor(xs, dtype=torch.long)
ys = torch.tensor(ys, dtype=torch.long)
num = len(xs)
print("Number of bigrams:", num)

g = torch.Generator().manual_seed(42)
# Initialize random parameters.
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

for i in range(15):
  xenc = F.one_hot(xs, vocab_size).float()

  # Forward pass. Do activation function of exponentiation and normalization.
  logits = xenc @ W
  counts = logits.exp()
  probs = counts / counts.sum(dim=1, keepdim=True)
  loss = -probs[torch.arange(len(ys)), ys].log().mean() # + 0.01 * (W ** 2).sum()
  print(loss.item())

  # Backward pass. Compute the loss.
  W.grad = None
  loss.backward()

  # Update
  W.data += -50 * W.grad

# Let's see what the model outputs.
for i in range(5):
  out = []
  ix = len(stoi) - 1

  while True:
    xenc = F.one_hot(torch.tensor([ix]), vocab_size).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(dim=1, keepdim=True)

    # Sample from the distribution.
    ix = torch.multinomial(probs, 1).item()
    out.append(itos[ix])

    if ix == len(stoi) - 1:
      break

print(''.join(out))