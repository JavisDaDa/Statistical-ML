import graderUtil
import pandas as pd
import numpy as np
import os
import shutil
import random
from data_utils import load_CIFAR10
import pickle as pkl

if __name__ == "__main__":
    grader = graderUtil.Grader()
    
    ############################################################
    ############################################################
    # 3: Implementing a k-nearest-neighbor classifier (25 points)
    ############################################################
    ############################################################
    with open("grader_data.pkl", "rb") as file:
        X_train, y_train, X_test, y_test, sol3 = pkl.load(file)
    
    # Load submission file
    knn = grader.load('k_nearest_neighbor')
    
    # Create classifier

    classifier = knn.KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances_two_loops(X_test)
    
    ############################################################
    # 3.1 Distance matrix computation with two loops (5 points)
    ############################################################
    grader.addPart('3.1', 
                   lambda: grader.requireIsEqual(sol3, dists), 
                   5)
    
    ############################################################
    # 3.2 Compute majority label (5 point)
    ############################################################
    def prob3_2_1():
        y_test_pred = classifier.predict_labels(sol3, k=1)
        grader.requireIsEqual(46, np.sum(y_test_pred == y_test))
    grader.addPart('3.2.1', prob3_2_1, 2)
    
    def prob3_2_2():
        y_test_pred = classifier.predict_labels(sol3, k=5)
        grader.requireIsEqual(36, np.sum(y_test_pred == y_test))
    grader.addPart('3.2.2', prob3_2_2, 3)

    ############################################################
    # 3.3 Distance matrix computation with one loop (5 points)
    ############################################################
    grader.addPart('3.3', 
                   lambda: grader.requireIsEqual(sol3, classifier.compute_distances_one_loop(X_test)), 
                   5)
    ############################################################
    # 3.4 Distance matrix computation with no loops (5 points)
    ############################################################
    grader.addPart('3.4', 
                   lambda: grader.requireIsEqual(sol3, classifier.compute_distances_no_loops(X_test)), 
                   5)
    
    ############################################################
    # 3.5 Choosing k by cross validation (5 points)
    ############################################################
    
    grader.addManualPart('3.5', 5)
    grader.grade()

    
