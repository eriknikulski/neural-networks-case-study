import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt

N = 100     # number of points per class
D = 2       # dimensionality
K = 3       # number of classes

h = 100     # hidden layer size

reg = 1e-3  # regularization coefficient
step_size = 1e-0


def get_data():
    """Returns spiral dataset"""
    X = np.zeros((N * K, D))    # data matrix (each row = single example)
    Y = np.zeros(N * K, dtype=np.uint8)   # class labels

    for j in range(K):
        ix = range(j * N, (j+1) * N)
        r = np.linspace(0.0, 1, N)     # radius
        t = np.linspace(4 * j, 4 * (j+1), N) + np.random.randn(N) * 0.2     # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X, Y


def show_data(x: npt.NDArray, y: npt.NDArray):
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def train(X: npt.NDArray, Y: npt.NDArray):
    # initialize parameters randomly
    W1 = 0.01 * np.random.randn(D, h)    # (D, h)
    b1 = np.zeros((1, h))                # (1, h)
    W2 = 0.01 * np.random.randn(h, K)    # (h, K)
    b2 = np.zeros((1, K))                # (1, K)

    num_examples = X.shape[0]
    for i in range(10000):
        # compute class scores for linear classifier
        hidden_layer = np.maximum(0, np.dot(X, W1) + b1)        # (N*K, D) dot (D, h) + (1, h) -> (N*K, h)
        scores = np.dot(hidden_layer, W2) + b2                  # (N*K, h) dot (h, K) + (1, K) -> (N*K, K)

        # get unnormalized probabilities
        exp_scores = np.exp(scores)                                         # (N*K, K)
        # normalize them for each example
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)      # (N*K, K)
        correct_logprobs = -np.log(probs[range(num_examples), Y])           # (N*K,)

        # compute loss: average cross-entropy loss and regularization
        data_loss = np.sum(correct_logprobs) / num_examples
        reg_loss = 0.5 * reg * np.sum(W1*W1)
        loss = data_loss + reg_loss

        if i % 1000 == 0:
            print(f'iteration {i}: loss {loss}')

        # at beginning loss should be ~1.1 because -log(1/3) = 1.1

        dscores = probs                             # (N*K, K)
        dscores[range(num_examples), Y] -= 1        # (N*K, K)
        dscores /= num_examples                     # (N*K, K)

        # backpropagate
        dW2 = np.dot(hidden_layer.T, dscores)           # (h, N*K) dot (N*K, K) -> (h, K)
        db2 = np.sum(dscores, axis=0, keepdims=True)    # (1, K)

        dhidden = np.dot(dscores, W2.T)                 # (N*K, K) dot (K, h) -> (N*K, h)
        # backprop ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0

        dW1 = np.dot(X.T, dhidden)                      # (D, N*K) dot (N*K, h) -> (D, h)
        db1 = np.sum(dhidden, axis=0, keepdims=True)    # (1, h)

        dW2 += reg * W2       # regularization loss
        dW1 += reg * W1       # regularization loss

        # SGD like parameter update
        W2 += -step_size * dW2
        b2 += -step_size * db2
        W1 += -step_size * dW1
        b1 += -step_size * db1


def main():
    X, Y = get_data()
    show_data(X, Y)
    train(X, Y)


if __name__ == '__main__':
    main()
