import numpy as np
import utils
import math
import scipy 
from scipy import optimize
import random
from scipy.special import xlogy

class LogisticRegressor:

    def __init__(self):
        self.theta = None


    def train(self,X,y,num_iters=1000):

        """
        Train a linear model using scipy's function minimization.
        
        Inputs:
        - X: N X D array of training data. Each training point is a D-dimensional
         row.
        - y: 1-dimensional array of length N with values in the reals.
        - num_iters: (integer) number of steps to take when optimizing

        Outputs:
        - optimal value for theta
        """

        num_train,dim = X.shape
        
        # standardize X so that each column has zero mean and unit variance
        # remember to take out the first column and do the feature normalize

        X_without_1s = X[:,1:]
        X_norm, mu, sigma = utils.std_features(X_without_1s)

        # add the ones back and assemble the XX matrix for training

        XX = np.vstack([np.ones((X_norm.shape[0],)),X_norm.T]).T
        theta = np.zeros((dim,))

        # Run scipy's fmin algorithm to run gradient descent

        theta_opt_norm = scipy.optimize.fmin_bfgs(self.loss, theta, fprime = self.grad_loss, args=(XX,y),maxiter=num_iters)

        # convert theta back to work with original X
        theta_opt = np.zeros(theta_opt_norm.shape)
        theta_opt[1:] = theta_opt_norm[1:]/sigma
        theta_opt[0] = theta_opt_norm[0] - np.dot(theta_opt_norm[1:],mu/sigma)


        return theta_opt

    def loss(self, *args):
        """
        Compute the logistic loss function 


        Inputs:
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.

        Returns: loss as a single float
        """
        theta,X,y = args
        m,dim = X.shape
        J = 0

        ##########################################################################
        # Compute the loss function for unregularized logistic regression        #
        # TODO: 1-2 lines of code expected                                       #
        ##########################################################################
        h = utils.sigmoid(X@theta)
        J = -1.0 / m * (y.T@np.log(h) + (1 - y).T@np.log(1 - h))
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J

    def grad_loss(self, *args):
        """
        Compute the gradient logistic loss function 


        Inputs:
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.

        Returns:  gradient with respect to theta; an array of the same shape as theta
        """
        theta,X,y = args
        m,dim = X.shape
        grad = np.zeros((dim,))

        ##########################################################################
        # Compute the gradient of the loss function for unregularized logistic   #
        # regression                                                             #
        # TODO: 1 line of code expected                                          #
        ##########################################################################
        grad = 1.0 / m * (X.T@(utils.sigmoid(X@theta) - y.T))
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return grad
        

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is the probability of being in class 1
        """
        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # Compute the predicted outputs for X                                     #
        # TODO: 1 line of code expected                                           #
        ###########################################################################
        y_pred = np.round(utils.sigmoid(X@self.theta))
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred


class RegLogisticRegressor:

    def __init__(self) -> object:
        self.theta = None

    def train(self,X,y,reg=1e-5,num_iters=1000,norm=True):

        """
        Train a linear model using scipy's function minimization.
        
        Inputs:
        - X: N X D array of training data. Each training point is a D-dimensional
         row.
        - y: 1-dimensional array of length N with values in the reals.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - norm: a boolean which indicates whether the X matrix is standardized before
                solving the optimization problem

        Outputs:
        - optimal value for theta
        """

        num_train,dim = X.shape

        # standardize features if norm=True

        if norm:
            # take out the first column and do the feature normalize
            X_without_1s = X[:,1:]
            X_norm, mu, sigma = utils.feature_normalize(X_without_1s)
            # add the ones back
            XX = np.vstack([np.ones((X_norm.shape[0],)),X_norm.T]).T
        else:
            XX = X

        # initialize theta
        theta = np.zeros((dim,))

        # Run scipy's fmin algorithm to run gradient descent
        theta_opt_norm = scipy.optimize.fmin_bfgs(self.loss, theta, fprime = self.grad_loss, args=(XX,y,reg),maxiter=num_iters)


        if norm:
            # convert theta back to work with original X
            theta_opt = np.zeros(theta_opt_norm.shape)
            theta_opt[1:] = theta_opt_norm[1:]/sigma
            theta_opt[0] = theta_opt_norm[0] - np.dot(theta_opt_norm[1:],mu/sigma)
        else:
            theta_opt = theta_opt_norm


        return theta_opt

    def loss(self, *args):
        """
        Compute the logistic loss function 


        Inputs:
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        theta,X,y,reg = args
        m,dim = X.shape
        J = 0

        ##########################################################################
        # Compute the loss function for regularized logistic regression          #
        # TODO: 1-2 lines of code expected                                       #
        ##########################################################################
        h = utils.sigmoid(X@theta)
        J = 1 / m * (-y.T@np.log(h) - (1 - y.T)@np.log(1 - h)) + reg / (2 * m) * (theta[1:].T@theta[1:])
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J

    def grad_loss(self, *args):
        """
        Compute the gradient logistic loss function 


        Inputs:
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        theta,X,y,reg = args
        m,dim = X.shape
        grad = np.zeros((dim,))
        ##########################################################################
        # Compute the gradient of the loss function for regularized logistic
        # regression                                                             #
        # TODO: 2 lines of code expected                                          #
        ##########################################################################
        h = utils.sigmoid(X@theta)
        theta[0] = 0
        grad = 1 / m * (X.T@(h - y.T)) + reg / m * theta

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return grad
        

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is a real number.
        """
        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # Compute the predicted outputs for X                                     #
        # TODO: 1 line of code expected                                           #
        #                                                                         #
        ###########################################################################
        y_pred = np.round(utils.sigmoid(X@self.theta))

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred


if __name__ == '__main__':
    A = np.array([0, 3, 1, 3, 0, 1, 1, 1]).reshape((4, 2))
    #prepend 1
    A = np.c_[np.ones(A.shape[0]), A]
    y = np.array([1, 1, 0, 0]).reshape(1, 4)
    theta = np.array([0, -2, 1]).reshape(3, 1)
    reg = 0.07
    #grad + hess
    h = utils.sigmoid(A @ theta)
    m, _ = A.shape
    test = h - y.T
    grad = 1 / m * A.T@(h - y.T) + reg / m * theta
    I = np.identity(grad.shape[0])
    HESS = 1 / m * (np.dot(A.T, A) * np.diag(h) * np.diag(1 - h)) + reg / m * I
    #Newton's law
    hessianInv = np.linalg.inv(HESS)
    theta1 = theta - np.dot(hessianInv, grad)
    theta2 = theta1 - np.dot(hessianInv, grad)
    print(f'After the first iteration is\n {theta1}')
    print(f'After the first iteration is\n {theta2}')
