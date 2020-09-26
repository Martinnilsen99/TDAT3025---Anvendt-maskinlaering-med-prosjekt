# Dette er Ole Christians visualiseringsverktøy for 2d lineær regresjon

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

matplotlib.rcParams.update({'font.size': 11})

# regarding the notations, see http://stats.stackexchange.com/questions/193908/in-machine-learning-why-are-superscripts-used-instead-of-subscripts

W_init = np.array([[-0.2]])
b_init = np.array([[4.84]])


class LinearRegressionModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.power(self.f(x) - y, 2))


model = LinearRegressionModel()

# Observed/training input and output
x_train = np.array([[1], [1.5], [2], [3], [4], [5], [6]])
y_train = np.array([[5], [3.5], [3], [4], [3], [1.5], [2]])

fig = plt.figure('Linear regression: 2D')

plot1 = fig.add_subplot(121)

plot1.plot(x_train.squeeze(), y_train.squeeze(), 'o',
           label='$(\\hat x^{(i)},\\hat y^{(i)})$', color='blue')

plot1_f, = plot1.plot([[0], [10]], model.f(
    np.array([[0], [10]])), color='green', label='$y = f(x) = xW+b$')

plot1_info = plot1.text(0.2, 0.6, '')

plot1_loss = []
for i in range(0, x_train.shape[0]):
    line, = plot1.plot([0, 0], [0, 0], color='red')
    plot1_loss.append(line)
    if i == 0:
        line.set_label('$|f(\\hat x^{(i)})-\\hat y^{(i)}|$')

plot1.set_xlim(np.min(x_train[:, 0] - 1), np.max(x_train[:, 0]) + 1)
plot1.set_ylim(np.min(y_train[:, 0] - 1), np.max(y_train[:, 0]) + 2)
plot1.set_xlabel('$x$')
plot1.set_ylabel('$y$')
plot1.legend(loc='upper left')
plot1.set_xticks([])
plot1.set_yticks([])

plot2 = fig.add_subplot(122, projection='3d')

W_grid, b_grid = np.meshgrid(
    np.linspace(-0.55, -0.5, 20), np.linspace(4.7, 5, 20))
loss_grid = np.empty([20, 20])
for i in range(0, W_grid.shape[0]):
    for j in range(0, W_grid.shape[1]):
        loss_grid[i, j] = LinearRegressionModel(
            [[W_grid[i, j]]], [[b_grid[i, j]]]).loss(x_train, y_train)
plot2.plot_surface(W_grid, b_grid, loss_grid, cmap=cm.coolwarm)

plot2_loss, = plot2.plot(np.ravel(model.W), np.ravel(
    model.b), model.loss(x_train, y_train), 'o', color='purple')

plot2.set_xlabel('$W$')
plot2.set_ylabel('$b$')
plot2.set_zlabel('$loss$')
plot2.set_xticks([])
plot2.set_yticks([])
plot2.set_zticks([])
plot2.set_xlim([-0.55, -0.5])
plot2.set_ylim([4.6, 5])
plot2.set_zlim([0.37, 0.4])


def update_figure(event=None):
    if (event is not None):
        if event.key == 'W':
            model.W[0, 0] += 0.01
        elif event.key == 'w':
            model.W[0, 0] -= 0.01

        elif event.key == 'B':
            model.b[0, 0] += 0.01
        elif event.key == 'b':
            model.b[0, 0] -= 0.01

        elif event.key == 'c':
            model.W = W_init.copy()
            model.b = b_init.copy()

    plot1_f.set_data([0, 10], model.f(np.array([[0], [10]])))

    for i in range(0, x_train.shape[0]):
        plot1_loss[i].set_data([x_train[i, 0], x_train[i, 0]], [
                               y_train[i, 0], model.f([[x_train[i, 0]]])])

    plot1_info.set_text('$W=[%.2f]$\n$b=[%.2f]$\n$loss = \\frac{1}{n}\\sum_{i=1}^{n}(f(\\hat x^{(i)}) - \\hat y^{(i)})^2 = %.2f$' %
                        (model.W[0, 0], model.b[0, 0], model.loss(x_train, y_train)))

    plot2_loss.set_data(np.ravel(model.W), np.ravel(model.b))
    plot2_loss.set_3d_properties(model.loss(x_train, y_train))

    fig.canvas.draw()


update_figure()
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()
