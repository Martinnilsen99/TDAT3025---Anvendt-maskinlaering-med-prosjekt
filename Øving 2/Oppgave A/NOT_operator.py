# We're making a model for the NOT operator, in which gives True if 0 and False if 1
import torch
import matplotlib.pyplot as plt

# Observed/training input and output. 0 = 1 and 1 = 0
x_train = torch.tensor([[0.0], [1.0]]).reshape(-1, 1)
y_train = torch.tensor([[1.0], [0.0]]).reshape(-1, 1)


class NOT_operator_model:
    def __init__(self):
        # Model variables
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Logits
    def logits(self, x):
        return x @ self.W + self.b

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NOT_operator_model()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.1)
for epoch in range(25_000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()    # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.figure('Oppgave A')
plt.title('NOT-operator')
plt.table(cellText=[[0, 1], [1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x$", "$f(x)$"],
          cellLoc="center",
          loc="lower left")
plt.scatter(x_train, y_train)
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.001).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, color="orange")
plt.show()
