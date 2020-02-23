from sklearn import linear_model
import numpy as np
import utils

class one_vs_allLogisticRegressor:

    def __init__(self,labels):
        self.theta = None
        self.labels = labels

    def train(self,X,y,reg):
        
        """
        Use sklearn LogisticRegression for training K classifiers in one-vs-rest mode
        Read the documentation carefully and choose an appropriate solver. Choose
        the L2 penalty. Remember that the X data has a column of ones prepended to it.
        Set the appropriate flag in logisticRegression to cover that.
        
        X = m X (d+1) array of training data. Assumes X has an intercept column
        y = 1 dimensional vector of length m (with K labels)
        reg = regularization strength

        Computes coefficents for K classifiers: a matrix with (d+1) rows and K columns
           - one theta of length d for each class
       """
        
        m,dim = X.shape
        theta_opt = np.zeros((dim,len(self.labels)))

        ###########################################################################
        # Compute the K logistic regression classifiers                           #
        # TODO: 7-9 lines of code expected                                        # 
        # use C=1.0/reg, solver is sag and max_iter = 1000                        #    
        ###########################################################################
        model = linear_model.LogisticRegression(C=1.0/reg, fit_intercept=False)
        for k in range(self.labels.size):
            model.fit(X, y == self.labels[k])
            theta_opt[:, k] = model.coef_
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        self.theta = theta_opt

    def predict(self,X):
        """
        Use the trained weights of this linear classifier to predict labels for X.

        Inputs:
        - X: m x (d+1) array of test data. 

        Returns:
        - y_pred: Predicted output for the data in X. y_pred is a 1-dimensional
          array of length m, and each element is a class label from one of the
          set of labels -- the one with the highest probability
        """

        y_pred = np.zeros(X.shape[0])

        ###########################################################################
        # Compute the predicted outputs for X                                     #
        # TODO: 1-2 lines of code expected                                        #
        ###########################################################################
        x = utils.sigmoid(X@self.theta)
        y_pred = np.argmax(x, axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

