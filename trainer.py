# Character level language model.
with open('input.txt', 'r', encoding='utf8') as file:
    text = file.read()

# print(len(text))

# Get a set of all the characters that appear in the data
chars = sorted(list(set(text)))
vocab_size = len(chars)
# print('Vocab size:', vocab_size)

# Build an encoder and a decoder for the characters
charToIdx = {char: idx for idx, char in enumerate(chars)}
idxToChar = {idx: char for idx, char in enumerate(chars)}

# Encoder is a function which takes a string and returns a list of indices.
# Decoder is a function which takes a list of indices and returns a string.
encoder = lambda x: [charToIdx[char] for char in x]
decoder = lambda x: [idxToChar[idx] for idx in x]

# Tokenize the data
import torch
data = torch.tensor(encoder(text), dtype=torch.long)
# 1115394 different tokens!
print(data.shape)

# Separate the dataset into training and validation sets. First 90% is training, last 10% is validation.
train_size = int(0.9 * len(data))
# val_size = len(data) - train_size
# train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
train_data = data[:train_size]
val_data = data[train_size:]

# Now we want to put the data into batches. We will use a DataLoader for this.
# Block size is also sometimes called context size.
block_size = 8
# print(train_data[:block_size + 1])

# Each batch actually has batch_size predictions.
# x = train_data[:block_size]
# y = train_data[1:block_size + 1]
# for t in range(block_size):
#     context = x[:t + 1]
#     target = y[t]
    # print(f"when input is {context} the target is {target}")

# Now actually create the batches.
batch_size = 4
block_size = 8
torch.manual_seed(1337)

# Generate a small batch of data of inputs x and targets y.
def get_batch(split):
  data = train_data if split == 'train' else val_data

  # Generate batch sized number of random indices.
  rand_indices = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[idx:idx + block_size] for idx in rand_indices])
  y = torch.stack([data[idx + 1:idx + block_size + 1] for idx in rand_indices])

  return x, y

xb, yb = get_batch('train')

for b in range(batch_size):
  for t in range(block_size):
    context = xb[b, :t + 1]
    target = yb[b, t]
    print(f"when input is {context} the target is {target}")