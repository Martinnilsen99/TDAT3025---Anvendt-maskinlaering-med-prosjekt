import torch
import matplotlib.pyplot as plt
import statistics
import numpy as np

# Observed/training input and output, one array of all days and one for all circumferences
x_observation = []
y_observation = []
with open('Oppgave C/day_head_circumference.csv') as file:
    content = file.readlines()[1:]
    for line in content:
        x, y = line.split(',', 1)
        x_observation.append(float(x))
        y_observation.append(float(y))

x_train = torch.tensor(x_observation).reshape(-1, 1)
y_train = torch.tensor(y_observation).reshape(-1, 1)


class NonLinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * 1 / (1 + torch.exp(-(x @ self.W + self.b))) + 31
        # return 20*np.std(x_numpy)*(x*self.W + self.b) + 31
        # I thought we were supposed to use standard deviation as the symbol was the same, but it seems like it is meant to be sigma(z) = 1 / 1+e^(-z)

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)
        # return torch.mean(torch.square(self.f(x) - y))


model = NonLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.000001)
for epoch in range(100_000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure('Nonlinear regression 2d')
plt.title('Predict head circumference based on age')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x_train, y_train)
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
y = model.f(x).detach().cpu()
plt.plot(x.cpu(), y, color='orange',
         label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')
# Got a straight line with the code below, I assume it should be bent as it is nonlinear
"""
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(
), label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$') 
"""
plt.legend()
plt.show()
