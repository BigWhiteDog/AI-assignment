# coding=utf8
'''
This demo is copyed from the following tutorial:
http://nbviewer.jupyter.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
I removed comments and some complex parameter settings.
I also changed the matrix multiplication order, since most ML algorithm will represent one sample in a row.
'''
import numpy as np
import theano
from theano import tensor as T
from matplotlib import pyplot as plt

class Layer(object):
    def __init__(self, layer_name, W_init, b_init, activation):
        n_input, n_output = W_init.shape
        print b_init.shape, n_output
        assert b_init.shape == (n_output,)
        self.W = theano.shared(value=W_init.astype(theano.config.floatX), name='W_%s'%layer_name)
        self.b = theano.shared(value=b_init.astype(theano.config.floatX), name='b_%s'%layer_name)
        self.activation = activation
        self.params = [self.W, self.b]
        
    def output(self, x):
        lin_output = T.dot(x, self.W)
        # lin_output = T.dot(self.W, x)
        lin_output  = lin_output+ self.b
        return (lin_output if self.activation is None else self.activation(lin_output))

class MLP(object):
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)
        
        self.layers = []
        for i, (W, b, activation) in enumerate(zip(W_init, b_init, activations)):
            self.layers.append(Layer(str(i+1), W, b, activation))

        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y)**2)


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []
    for param in params:
        previous_step = theano.shared(param.get_value()*0.)
        step = momentum*previous_step + learning_rate*T.grad(cost, param)
        updates.append((previous_step, step))
        updates.append((param, param - step))
    return updates

def build_model(layer_sizes, learning_rate, momentum):
    W_init = []
    b_init = []
    activations = []
    # layer_size[:-1] = [2, 4]
    # layer_size[1:] = [4, 1]
    # zip(layer_sizes[:-1], layer_sizes[1:]) = [(2, 4), (4, 1)]
    for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
        W_init.append(np.random.randn(n_input, n_output))
        b_init.append(np.ones((n_output, )))
        # top layer use sigmoid activation since this is a binary classifiction.
        activations.append(T.nnet.sigmoid)
    mlp = MLP(W_init, b_init, activations)

    mlp_input = T.matrix('mlp_input', dtype=theano.config.floatX)
    mlp_target = T.matrix('mlp_target', dtype='int64')

    cost = mlp.squared_error(mlp_input, mlp_target)
    train_fun = theano.function([mlp_input, mlp_target], cost,
            updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
    mlp_output_fun = theano.function([mlp_input], mlp.output(mlp_input))
    return train_fun, mlp_output_fun


# samples
np.random.seed(0)
N = 1000
x1 = np.random.random([N/2, 2]) + [0, 1]
y1 = np.ones([N/2, 1])
x0 = np.random.random([N-N/2, 2]) + [0, 0]
y0 = np.zeros([N-N/2, 1])
# X.shape = [1000, 2]
# y.shape = [1000, 1]
X = np.vstack([x1, x0]).astype(theano.config.floatX)
y = np.vstack([y1, y0]).astype('int64')

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, lw=.3, s=3, cmap=plt.cm.cool)
plt.axis([-6, 6, -6, 6])
plt.show()


# build model
# X.shape = [1000, 2]
layer_sizes = [X.shape[1], X.shape[1]*2, 1]
learning_rate = 0.01
momentum = 0.9
train_fun, mlp_output_fun = build_model(layer_sizes, learning_rate, momentum)

# train given iterations
iteration = 0
max_iteration = 20
while iteration < max_iteration:
    current_cost = train_fun(X, y)
    current_output = mlp_output_fun(X)
    accuracy = np.mean((current_output > .5) == y)
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], c=current_output,
                lw=.3, s=3, cmap=plt.cm.cool, vmin=0, vmax=1)
    plt.axis([-6, 6, -6, 6])
    plt.title('iter: {}, Cost: {:.3f}, Accuracy: {:.3f}'.format(iteration, float(current_cost), accuracy))
    plt.show()
    iteration += 1
