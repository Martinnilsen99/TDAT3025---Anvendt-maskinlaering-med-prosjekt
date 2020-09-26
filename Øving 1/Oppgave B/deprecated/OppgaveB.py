import torch
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d

# Observed/training input and output, one array of all lengths and one for all weights
x_observation = []
y_observation = []
with open('Oppgave B/day_length_weight.csv') as file:
    content = file.readlines()[1:]  # jumps over headerline from csv
    for line in content:
        d, l, w = line.split(',')  # days, length, weight
        x_observation.append([float(l), float(w)])
        y_observation.append(float(d))

x_train = torch.tensor(x_observation).reshape(-1, 2)
y_train = torch.tensor(y_observation).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)
        #  return torch.mean(torch.square(self.f(x) - y))
        # return np.mean(np.power(self.f(x) - y, 2))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)
for epoch in range(100_000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))


""" # Visualize result
fig = plt.figure('Linear regression 3d')
ax = plt.axes(projection='3d', title="Predict days based on length and weight")
# Information for making the plot understandable
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([])  # Removes the lines and information from axes
ax.set_yticks([])
ax.set_zticks([])
ax.w_xaxis.line.set_lw(0)
ax.w_yaxis.line.set_lw(0)
ax.w_zaxis.line.set_lw(0)
ax.quiver([0], [0], [0], [torch.max(x_train[:, 0] + 1)], [0],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [torch.max(x_train[:, 1] + 1)],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [0], [torch.max(y_train + 1)],
          arrow_length_ratio=0, color='black')
# Plot
ax.scatter(np.array(x_observation)[:, 0], np.array(
    x_observation)[:, 1], y_observation)
x = torch.tensor([[torch.min(x_train[:, 0])], [torch.max(x_train[:, 0])]])
y = torch.tensor([[torch.min(x_train[:, 1])], [torch.max(x_train[:, 1])]])
ax.plot(x.flatten(), y.flatten(), model.f(
    x.reshape(1, -1) + model.f(y.reshape(1, -1))).detach().flatten(), label='$f(x)=xW+b$', color="orange")
ax.legend(loc='upper left')
plt.show() """
