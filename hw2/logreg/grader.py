import graderUtil
import pandas as pd
import numpy as np
import os
import shutil
import random
import pickle as pkl
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

if __name__ == "__main__":
    grader = graderUtil.Grader()
    
    ############################################################
    ############################################################
    # 4: Implementing logistic regression (45 points)
    ############################################################
    ############################################################
    
    # Load correct answer
    with open("grader_data.pkl", "rb") as file:
        sol4 = pkl.load(file)
    
    # Load data
    data = pd.read_csv('ex1data1.txt')
    X = np.vstack([data.x1,data.x2]).T
    y = data.y
    
    # Load submission file
    logreg = grader.load('logistic_regressor')
    utils = grader.load('utils')
    
    XX = np.vstack([np.ones((X.shape[0],)),X.T]).T

    ############################################################
    # 4A: Logistic regression (15 points)
    ############################################################
    
    ############################################################
    # 4A1: Implementing logistic regression: the sigmoid function (5 points)
    ############################################################
    def prob4A1():
        expected = []
        observed = []
        for i in [-2., -1., 0., 1., 2.]:
            expected.append(1 / (1 + np.exp(-i)))
            observed.append(utils.sigmoid(i))
        grader.requireIsEqual(expected, observed)
    
    grader.addPart('4A1', prob4A1, 5)
    
    ############################################################
    # 4A2: Cost function and gradient of logistic regression (5 points)
    ############################################################

    def prob4A2_1():
        log_reg1 = logreg.LogisticRegressor()
        theta = np.zeros((XX.shape[1],))
        loss = log_reg1.loss(theta,XX,y)
        grader.requireIsEqual(sol4['loss'], loss)
        
    def prob4A2_2():
        log_reg1 = logreg.LogisticRegressor()
        theta = np.zeros((XX.shape[1],))    
        grad = log_reg1.grad_loss(theta,XX,y)
        grader.requireIsEqual(sol4['grad'], grad)
    
    grader.addPart('4A2.1', prob4A2_1, 3)
    grader.addPart('4A2.2', prob4A2_2, 2)

    ############################################################
    # 4A3: Prediction using a logistic regression model (5 points)
    ############################################################

    def prob4A3():
        log_reg1 = logreg.LogisticRegressor()
        theta_opt = log_reg1.train(XX,y,num_iters=400)
        log_reg1.theta = theta_opt
        loss = log_reg1.loss(theta_opt,XX,y)
        
        grader.requireIsGreaterThan(0.2, loss)
        grader.requireIsLessThan(0.21, loss)
        
        grader.requireIsGreaterThan(-25.3, theta_opt[0])
        grader.requireIsLessThan(-25., theta_opt[0])
        
        grader.requireIsGreaterThan(0.19, theta_opt[1])
        grader.requireIsLessThan(0.22, theta_opt[1])
        
        grader.requireIsGreaterThan(0.19, theta_opt[1])
        grader.requireIsLessThan(0.22, theta_opt[1])

    grader.addPart('4A3', prob4A3, 5)
    
    ############################################################
    # 4B: Regularized logistic regression (20 points)
    ############################################################

    ############################################################
    # 4B1: Cost function and gradient for regularized logistic regression (10 points)
    ############################################################

    def prob4B1_1():
        log_reg1 = logreg.RegLogisticRegressor()
        theta = np.zeros((XX.shape[1],))
        loss = log_reg1.loss(theta,XX,y, 100)
        grader.requireIsGreaterThan(1.78, loss)
        grader.requireIsLessThan(1.79, loss)
        
    def prob4B1_2():
        log_reg1 = logreg.RegLogisticRegressor()
        theta = np.zeros((XX.shape[1],))    
        grad = log_reg1.grad_loss(theta,XX,y)
        grader.requireIsEqual(sol4['grad'], grad)
        grader.requireIsGreaterThan(0.91, grad[1])
        grader.requireIsLessThan(0.92, grad[1])
    
    grader.addPart('4B1.1', prob4A2_1, 5)
    grader.addPart('4B1.2', prob4A2_2, 5)

    ############################################################
    # 4B2: Prediction using the model (2 points)
    ############################################################

    def prob4B2():
        poly = sklearn.preprocessing.PolynomialFeatures(degree=6,include_bias=False)
        X_poly = poly.fit_transform(X)
        XX = np.vstack([np.ones((X_poly.shape[0],)),X_poly.T]).T
        reg_lr1 = logreg.RegLogisticRegressor()
        reg = 100.0
        theta_opt = reg_lr1.train(XX,y,reg=reg,num_iters=1000,norm=False)
        reg_lr1.theta = theta_opt
        predy = reg_lr1.predict(XX)
        accuracy = np.sum(predy==y)/y.size
        grader.requireIsGreaterThan(0.59, accuracy)

    grader.addPart('4B2', prob4B2, 2)

    ############################################################
    # 4B3: Varying λ (3 points)
    ############################################################
    
    grader.addManualPart('4B3', 3)
    
    ############################################################
    # 4B4: Exploring L1 and L2 penalized logistic regression (5 points)
    ############################################################

    grader.addManualPart('4B4', 5)

    ############################################################
    # 4C: Logistic regression for spam classification (10 points)
    ############################################################

    ############################################################
    # 4C1: Feature transformation (2 points)
    ############################################################

    def prob4C1_1():
        test_vector = np.array([0., 1., 2.])
        expected = np.log(1 + test_vector)
        observed = utils.log_features(test_vector)
        grader.requireIsEqual(expected, observed)
    
    def prob4C1_2():
        test_vector = np.array([-1., 0., 1.])
        expected = np.zeros(test_vector.shape)
        expected[test_vector > 0] = 1
        observed = utils.bin_features(test_vector)
        grader.requireIsEqual(expected, observed)
    
    grader.addPart('4C1_1', prob4C1_1, 1)
    grader.addPart('4C1_2', prob4C1_2, 1)

    ############################################################
    # 4C2: Fitting regularized logistic regression models (L2 and L1) (8 points)
    ############################################################
    
    def prob4C2():
        Xtrain,Xtest,ytrain,ytest = utils.load_spam_data()

        Xtrain_std,mu,sigma = utils.std_features(Xtrain)
        Xtrain_logt = utils.log_features(Xtrain)
        Xtrain_bin = utils.bin_features(Xtrain)

        Xtest_std = (Xtest - mu)/sigma
        Xtest_logt = utils.log_features(Xtest)
        Xtest_bin = utils.bin_features(Xtest)


        def run_dataset(X,ytrain,Xt,ytest,typea,penalty):

            best_lambda = utils.select_lambda_crossval(X,ytrain,0.1,5.1,0.5,penalty)

            if penalty == "l2":
                lreg = linear_model.LogisticRegression(penalty=penalty,C=1.0/best_lambda, solver='lbfgs',fit_intercept=True,max_iter=400)
            else:
                lreg = linear_model.LogisticRegression(penalty=penalty,C=1.0/best_lambda, solver='liblinear',fit_intercept=True,max_iter=400)
            lreg.fit(X,ytrain)
            predy = lreg.predict(Xt)
            return np.mean(predy==ytest)

        grader.requireIsGreaterThan(0.9, run_dataset(Xtrain_std,ytrain,Xtest_std,ytest,"std","l2"))
        grader.requireIsGreaterThan(0.9, run_dataset(Xtrain_logt,ytrain,Xtest_logt,ytest,"logt","l1"))
        grader.requireIsGreaterThan(0.9, run_dataset(Xtrain_bin,ytrain,Xtest_bin,ytest,"bin","l2"))
                                    
    grader.addPart('4C2', prob4C2, 8)

    grader.grade()
