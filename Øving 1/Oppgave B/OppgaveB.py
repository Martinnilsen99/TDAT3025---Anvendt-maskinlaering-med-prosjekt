import torch
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# Observed/training input and output, one array of all lengths and one for all weights
day_observation = []
length_observation = []
weight_observation = []
with open('Oppgave B/day_length_weight.csv') as file:
    content = file.readlines()[1:]  # jumps over headerline from csv
    for line in content:
        d, l, w = line.split(',')  # days, length, weight
        day_observation.append(float(d))
        length_observation.append(float(l))
        weight_observation.append(float(w))
day_train = torch.tensor(day_observation).reshape(-1, 1)
length_train = torch.tensor(length_observation).reshape(-1, 1)
weight_train = torch.tensor(weight_observation).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W1 = torch.tensor([[0.0]], requires_grad=True)
        self.W2 = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x1, x2):
        # @ corresponds to matrix multiplication
        return (x1 @ self.W1) + (x2 @ self.W2) + self.b

    # Uses Mean Squared Error
    def loss(self, x1, x2, y):
        return torch.nn.functional.mse_loss(self.f(x1, x2), y)
        #  return torch.mean(torch.square(self.f(x) - y))
        # return np.mean(np.power(self.f(x) - y, 2))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W1, model.W2], 0.0001)
for epoch in range(100_000):
    # Compute loss gradients
    model.loss(length_train, weight_train, day_train).backward()
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W1 = %s, W2 = %s, b = %s, loss = %s" % (model.W1, model.W2,
                                               model.b, model.loss(length_train, weight_train, day_train)))


# Visualize result
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
ax.quiver([0], [0], [0], [torch.max(length_train + 1)], [0],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [torch.max(weight_train + 1)],
          [0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [0], [torch.max(day_train + 1)],
          arrow_length_ratio=0, color='black')
# Plot
ax.scatter(length_observation, weight_observation, day_observation)
x = torch.tensor([[torch.min(length_train)], [torch.max(length_train)]])
y = torch.tensor([[torch.min(weight_train)], [torch.max(weight_train)]])
ax.plot(x.flatten(), y.flatten(), model.f(
    x, y).detach().flatten(), label='$f(x)=xW+b$', color="orange")
ax.legend()
plt.show()
