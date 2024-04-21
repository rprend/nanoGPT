import torch

# What we're doing is demonstrating the initialization of a hidden layer in a neural net.
x = torch.randn(1000, 10)
w = torch.randn(10, 200)
y = x @ w

print(x.mean(), x.std())
print(y.mean(), y.std())

# Import pyplot as plt
import matplotlib.pyplot as plt

# Here what we see is that the mean of the input stays about the same between x and y, but
# the standard deviation grows significantly. It goes from ~1 to ~3. This is not good because we
# want our activations to stay similar. We can experiment by scaling w up or down-- this will grow
# or shrink the standard deviation of y. And thus we can derive mathematically the exact amount to scale
# w by to keep the standard deviation of y the same as x. This is the Kaiming initialization. This is equal
# to sqrt(2 / fan_in) aka fan_in ** 0.5, where fan_in is the number of input units to the layer.
plt.figure(figsize=(20, 5))
plt.subplot(121)
plt.hist(x.view(-1).tolist(), 50, density=True)
plt.subplot(122)
plt.hist(y.view(-1).tolist(), 50, density=True)


plt.show()

# Here we show proper initalization of weights if we are using a tanh activation function / nonlinearity.
w2 = torch.randn(10, 200) * ((5/3)/10**0.5)