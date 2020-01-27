import numpy as np
import scipy

class RegularizedLinearRegressor_Multi:

    def __init__(self):
        self.theta = None


    def train(self,X,y,reg=1e-5,num_iters=100):

        """
        Train a linear model using regularized  gradient descent.
        
        Inputs:
        - X: N X D array of training data. Each training point is a D-dimensional
         row.
        - y: 1-dimensional array of length N with values in the reals.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing


        Outputs:
        optimal value for theta
        """
    
        num_train,dim = X.shape
        theta = np.ones((dim,))


        # Run scipy's fmin algorithm to run the gradient descent
        theta_opt = scipy.optimize.fmin_bfgs(self.loss, theta, fprime = self.grad_loss, args=(X,y,reg),maxiter=num_iters)
            
        
        return theta_opt

    def loss(self, *args):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs: (in *args as a tuple)
        - theta: D+1 vector
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        
        pass

    def grad_loss(self, *args):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs: (in *args as a tuple)
        - theta: D+1 vector
        - X: N x D array of data; each row is a data point.
        - y: 1-dimensional array of length N with real values.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.theta; an array of the same shape as theta
        """
        pass

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each row is a D-dimensional point.

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is a real number.
        """
        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted outputs in y_pred.           #
        #  1 line of code expected                                                #
        ###########################################################################
        y_pred = np.dot(X, np.array(self.theta).T)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def normal_equation(self,X,y,reg):
        """
        Solve for self.theta using the normal equations.
        """
        ###########################################################################
        # TODO:                                                                   #
        # Solve for theta_n using the normal equation.                            #
        #  One line of code expected                                              #
        ###########################################################################

        theta_n = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

        ###########################################################################
        return theta_n

class RegularizedLinearReg_SquaredLoss(RegularizedLinearRegressor_Multi):
    "A subclass of Linear Regressors that uses the squared error loss function """

    """
    Function that returns loss and gradient of loss with respect to (X, y) and
    self.theta
        - loss J is a single float
        - gradient with respect to self.theta is an array of the same shape as theta

    """

    def loss (self,*args):
        theta,X,y,reg = args

        num_examples,dim = X.shape
        J = 0
        grad = np.zeros((dim,))
        ###########################################################################
        # TODO:                                                                   #
        # Calculate J (loss) wrt to X,y, and theta.                               #
        #  2 lines of code expected                                               #
        ###########################################################################
        # J = 1.0 / (2.0 * num_examples) * np.sum(np.square((np.dot(X, np.array(theta).T)) - y))
        # J += float(reg) / (2.0 * num_examples) * (np.sum(np.square(theta)))
        # J = 1.0 / (2.0 * num_examples) * np.square(np.array(theta)@X.T - y).sum()
        # J += float(reg) / (2.0 * num_examples) * (np.square(theta).sum())
        # diff = np.sum(np.tile(np.array(theta).T, (num_examples, 1)) * X, axis=1) - y
        # J = (np.sum(np.square(diff)) + reg * (np.sum(np.square(theta[1:])))) / (2. * num_examples)
        J = 1.0 / (2.0 * num_examples) * (np.square(y - X@np.array(theta).T)).sum()
        J += reg / (2.0 * num_examples) * np.square(theta[1:]).sum()
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J

    def grad_loss(self,*args):                                                                          
        theta,X,y,reg = args
        num_examples,dim = X.shape
        grad = np.zeros((dim,))
        # print(X.shape, y.shape, theta.shape)
        ###########################################################################
        # TODO:                                                                   #
        # Calculate gradient of loss function wrt to X,y, and theta.              #
        #  3 lines of code expected                                               #
        ###########################################################################
        # grad[0] = 1.0 / num_examples * np.dot((np.dot(X, np.array(theta).T) - y).T, X)[0]
        # grad[1:] = 1.0 / num_examples * np.dot((np.dot(X, np.array(theta).T) - y).T, X)[1:] + reg / num_examples * theta[1:]
        grad[0] = 1.0 / num_examples * ((X@np.array(theta).T - y).T@X)[0]
        grad[1:] = 1.0 / num_examples * ((X@np.array(theta).T - y).T@X)[1:] + reg / num_examples * theta[1:]
        # diff = np.sum(np.tile(np.array(theta).T, (num_examples, 1)) * X, axis=1) - y
        # grad = (np.matmul(diff, X) + reg * (np.hstack([0, theta[1:]]))) / num_examples
        # grad[0] = 1.0 / num_examples * ((X@np.array(theta).T - y).T@X).sum()[0]
        # grad[1:] = 1.0 / num_examples * ((X @ np.array(theta).T - y).T@ X).sum()[1:] + reg / num_examples * theta[1:]
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return grad


class LassoLinearReg_SquaredLoss(RegularizedLinearRegressor_Multi):
    "A subclass of Linear Regressors that uses the squared error loss function """

    """
    Function that returns loss and gradient of loss with respect to (X, y) and
    self.theta
        - loss J is a single float
        - gradient with respect to self.theta is an array of the same shape as theta

    """

    def loss (self,*args):
        theta,X,y,reg = args

        num_examples,dim = X.shape
        J = 0
        grad = np.zeros((dim,))
        ###########################################################################
        # TODO:                                                                   #
        # Calculate J (loss) wrt to X,y, and theta.                               #
        #  2 lines of code expected                                               #
        ###########################################################################
        # J = 1.0 / (2.0 * num_examples) * sum(np.square((np.dot(X, np.array(theta).T)) - y))
        # J += reg / num_examples * sum(np.abs(theta))
        # diff = np.sum(np.tile(np.array(theta).T, (num_examples, 1)) * X, axis=1) - y
        # # J = (np.sum(np.square(diff)) + reg * (np.sum(np.square(theta[1:])))) / (2. * num_examples)
        # J = (np.sum(np.square(diff)) + reg * (np.sum(np.abs(theta)[1:]))) / (2. * num_examples)
        J = 1.0 / (2.0 * num_examples) *(np.square(y - X@np.array(theta).T)).sum()
        J += reg / num_examples * (np.abs(theta)[1:]).sum()
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return J

    def grad_loss(self,*args):                                                                          
        theta,X,y,reg = args
        num_examples,dim = X.shape
        grad = np.zeros((dim,))

        ###########################################################################
        # TODO:                                                                   #
        # Calculate gradient of loss function wrt to X,y, and theta.              #
        #  3 lines of code expected                                               #
        ###########################################################################
        # grad[0] = 1.0 / num_examples * np.dot((np.dot(X, np.array(theta).T) - y).T, X)[0]
        # grad[1:] = 1.0 / num_examples * np.dot((np.dot(X, np.array(theta).T) - y).T, X)[1:] + reg / num_examples * np.sign(theta[1:])
        # diff = np.sum(np.tile(np.array(theta).T, (num_examples, 1)) * X, axis=1) - y
        # grad = (np.matmul(diff, X) + reg * (np.hstack([0, np.sign(theta[1:])]))) / num_examples
        # grad[0] = 1.0 / num_examples * (X @ np.array(theta).T - y).sum().T @ X[0]
        # grad[1:] = 1.0 / num_examples * (X @ np.array(theta).T - y).sum().T @ X[1:] + reg / num_examples * np.sign(theta[1:])
        grad[0] = 1.0 / num_examples * ((X @ np.array(theta).T - y).T @ X)[0]
        grad[1:] = 1.0 / num_examples * ((X @ np.array(theta).T - y).T @ X)[1:] + reg / num_examples * np.sign(theta[1:])
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return grad
