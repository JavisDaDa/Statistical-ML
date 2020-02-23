import numpy as np
from random import shuffle
import scipy.sparse


def softmax_loss_naive(theta, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs:
    - theta: d x K parameter matrix. Each column is a coefficient vector for class k
    - X: m x d array of data. Data are d-dimensional rows.
    - y: 1-dimensional array of length m with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to parameter matrix theta, an array of same size as theta
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in J and the gradient in grad. If you are not              #
    # careful here, it is easy to run into numeric instability. Don't forget    #
    # the regularization term!                                                  #
    #############################################################################
    K = theta.shape[1]
    for i in range(m):
        thetaP = X[i]@theta
        thetaP -= np.max(thetaP)
        prob = np.exp(thetaP) / np.sum(np.exp(thetaP))
        J += np.log(prob[y[i]])
        prob[y[i]] -= 1
        for k in range(K):
            grad[:, k] += X[i, :] * prob[k]
    J /= (-m)
    J += reg * np.sum(np.square(theta)) / (2 * m)
    grad /= m
    grad += reg * theta / m
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad


def softmax_loss_vectorized(theta, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    J = 0.0
    grad = np.zeros_like(theta)
    m, dim = X.shape

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in J and the gradient in grad. If you are not careful      #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization term!                                                      #
    #############################################################################
    K = theta.shape[1]
    thetaP = X @ theta
    thetaP -= np.max(thetaP, axis=1, keepdims=True)
    prob = np.exp(thetaP) / np.sum(np.exp(thetaP), axis=1, keepdims=True)
    label = (y.reshape(-1, 1) == np.arange(K)) * 1
    J = -1.0 / m * np.sum(np.multiply(label, np.log(prob)))
    J += reg * np.sum(np.square(theta)) / (2 * m)
    grad = -1.0 / m * X.T@(label - prob) + reg / m * theta
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return J, grad
