import numpy as np


##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
    """
    SVM hinge loss function for two class problem

    Inputs:
    - theta: A numpy vector of size d containing coefficients.
    - X: A numpy array of shape mxd
    - y: A numpy array of shape (m,) containing training labels; +1, -1
    - C: (float) penalty factor

    Returns a tuple of:
    - loss as single float
    - gradient with respect to theta; an array of same shape as theta
  """

    m, d = X.shape
    grad = np.zeros(theta.shape)
    J = 0

    ############################################################################
    # TODO                                                                     #
    # Implement the binary SVM hinge loss function here                        #
    # 4 - 5 lines of vectorized code expected                                  #
    ############################################################################
    h = X@theta
    J = 1.0 / 2.0 * np.sum(np.square(theta)) / m + C / m * np.sum(np.max([np.zeros(m), 1 - y * h], axis=0))
    grad = 1.0 / m * theta + C / m * X.T@(-y * (y * h < 1))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return J, grad


##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, C):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension d, there are K classes, and we operate on minibatches
    of m examples.

    Inputs:
    - theta: A numpy array of shape d X K containing parameters.
    - X: A numpy array of shape m X d containing a minibatch of data.
    - y: A numpy array of shape (m,) containing training labels; y[i] = k means
      that X[i] has label k, where 0 <= k < K.
    - C: (float) penalty factor

    Returns a tuple of:
    - loss J as single float
    - gradient with respect to weights theta; an array of same shape as theta
    """

    K = theta.shape[1]  # number of classes
    m = X.shape[0]  # number of examples

    J = 0.0
    dtheta = np.zeros(theta.shape)  # initialize the gradient as zero
    delta = 1.0

    #############################################################################
    # TODO:                                                                     #
    # Compute the loss function and store it in J.                              #
    # Do not forget the regularization term!                                    #
    # code above to compute the gradient.                                       #
    # 8-10 lines of code expected                                               #
    #############################################################################
    for i in range(m):
        h = X[i, :]@theta
        hy = h[y[i]]
        for j in range(K):
            if j == y[i]:
                continue
            cur = h[j] - hy + delta
            if cur > 0:
                J += cur
                dtheta[:, j] += X[i, :]
                dtheta[:, y[i]] -= X[i, :]
    J = np.sum(np.square(theta)) / 2 / m + C * J / m
    dtheta = theta / m + C * dtheta / m
    # For jupyter notebook
    # J = C * np.sum(np.square(theta)) / 2 / m + J / m
    # dtheta = C * theta / m + dtheta / m
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dtheta.            #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return J, dtheta


def svm_loss_vectorized(theta, X, y, C):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    J = 0.0
    dtheta = np.zeros(theta.shape)  # initialize the gradient as zero
    delta = 1.0
    m = X.shape[0]
    K = theta.shape[1]
    d = theta.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in variable J.                                                     #
    # 8-10 lines of code                                                        #
    #############################################################################
    h = X@theta
    hy = h - h[range(len(y)), y].reshape((-1, 1))
    hy[hy != 0] += delta
    cur = np.maximum(0.0, hy)
    J = np.sum(np.square(theta)) / 2 / m + C * np.sum(cur) / m
    # For jupyter notebook
    # J = C * np.sum(np.square(theta)) / 2 / m + np.sum(cur) / m
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dtheta.                                       #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    cc = (hy > 0) * 1
    cc[range(len(y)), y] = -np.sum(cc, axis=1)
    dtheta = theta / m + C * X.T@cc / m
    # For jupyter notebook
    # dtheta = C * theta / m + X.T @ cc / m
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return J, dtheta
