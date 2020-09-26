import torch
import matplotlib.pyplot as plt

# Observed/training input and output, one array of all lengths and one for all weights
x_observation = []
y_observation = []
with open('Oppgave A/length_weight.csv') as file:
    content = file.readlines()[1:]
    for line in content:
        x, y = line.split(',', 1)
        x_observation.append(float(x))
        y_observation.append(float(y))

x_train = torch.tensor(x_observation).reshape(-1, 1)
y_train = torch.tensor(y_observation).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)
        # return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)
for epoch in range(500_000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure('Linear regression 2d')
plt.title('Predict weight based on length')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train, y_train, 'o')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()
