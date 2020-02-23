import graderUtil
import pandas as pd
import numpy as np
import os
import data_utils
import pickle
import math

if __name__ == "__main__":

    grader = graderUtil.Grader()

    # Load submission files
    svm_submitted = grader.load("linear_svm")
    classifier_submitted = grader.load("linear_classifier")
    util_submitted = grader.load("utils")

    # Load dataset.
    testing_data = pickle.load(open("testing_data.txt", 'br'))

    Xs = [x for x, y in testing_data]
    ys = [y for x, y in testing_data]
    binary_X = np.concatenate(Xs[:2], axis=0)
    binary_y = np.concatenate(ys[:2], axis=0)

    multi_X = np.concatenate(Xs, axis=0)
    multi_y = np.concatenate(ys, axis=0)

    num_features = binary_X.shape[1]
    num_classes = len(Xs)


    ############################################################
    # 1: SVM for binary classification (25 points)
    ############################################################

    def prob3_1_a():
        theta = np.zeros((num_features,))
        for i in range(num_features):
            x = i * 3. / num_features - 1.5
            theta[i] = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        true_grad1 = np.array([0.00647588, 0.00798063, 0.00962638, 0.01136516,
                               0.01313335, 0.01485466, 0.01644511, 0.0178196,
                               0.01889933, 0.01961922, 0.01993445, 0.01982504,
                               0.01929794, 0.01838632, 0.01714612, 0.01565035,
                               0.013982, 0.01222649, 0.01046456, 0.00876653])
        true_grad2 = np.array([3.67776057, -1.01126222, 5.73216413, 2.69324782,
                               0.25235274, -5.19952289, 2.96023899, 5.00485327,
                               -4.9775721, -0.17691139, 0.18233956, -8.89597802,
                               3.97272957, -4.93965246, 7.71777673, 4.35327892,
                               1.29210853, 9.67060608, -0.48110686, 3.5146502])

        loss1, grad1 = svm_submitted.binary_svm_loss(theta, binary_X, binary_y, 0.)
        grader.requireIsEqual(6.97912159, loss1)
        grader.requireIsEqual(true_grad1, grad1[np.array(range(0, 3000, 150))])

        loss2, grad2 = svm_submitted.binary_svm_loss(theta, binary_X, binary_y, 0.5)
        grader.requireIsEqual(2584.69127739, loss2)
        grader.requireIsEqual(true_grad2, grad2[np.array(range(0, 3000, 150))])


    def prob3_1_c():
        x_1 = binary_X[0, :]
        x_2 = binary_X[10, :]
        grader.requireIsEqual(0.40887257, util_submitted.gaussian_kernel(x_1, x_2, 5e3))
        grader.requireIsEqual(0.79964457, util_submitted.gaussian_kernel(x_1, x_2, 1e4))


    grader.addPart("3.1.A", prob3_1_a, 5)
    grader.addPart("3.1.C", prob3_1_c, 3)


    ############################################################
    # 4: SVM for multi-class classification (35 points)
    ############################################################

    def prob4_ab(structured_svm):
        theta = np.zeros((num_features, num_classes))
        for j in range(num_classes):
            for i in range(num_features):
                x = i * j * 0.5 / num_features - 0.5 * j
                theta[i, j] = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

        row_idx = np.array(range(0, 3000, 300))
        col_idx = np.array(range(0, 10, 3))

        true_grad1 = np.array([[0.00398942, 0.00129517, 4.43184841e-05, 1.59837411e-07],
                               [0.00398942, 0.00159612, 0.00010222, 1.04791312e-06],
                               [0.00398942, 0.00192527, 0.00021639, 5.66442936e-06],
                               [0.00398942, 0.00227303, 0.00042042, 2.52447508e-05],
                               [0.00398942, 0.00262667, 0.00074970, 9.27619886e-05],
                               [0.00398942, 0.00297093, 0.00122698, 0.00028103],
                               [0.00398942, 0.00328902, 0.00184304, 0.00070197],
                               [0.00398942, 0.00356392, 0.00254086, 0.00144567],
                               [0.00398942, 0.00377986, 0.00321495, 0.00245474],
                               [0.00398942, 0.00392384, 0.00373350, 0.00343657]])
        true_grad2 = np.array([[6.99621860, -4.08019094, -3.29779364, -18.04425677],
                               [14.82546676, 0.12061122, -9.03123614, -19.97450650],
                               [15.14545779, -3.38238696, -3.86667952, -20.56283821],
                               [5.18771309, -2.30520941, 1.25958124, -12.94373353],
                               [6.04687227, 13.55903809, -3.64638743, -9.43561295],
                               [-4.50345363, 1.50293297, -1.42376035, -5.10719999],
                               [2.92193044, 5.46966167, 0.18138549, -6.84998435],
                               [3.45361105, -2.07118383, -0.00254321, -8.71868044],
                               [4.58711350, 0.37003047, 1.75446475, -11.09317056],
                               [6.45893227, 4.77229527, -0.68072363, -3.12324914]])
        true_grad3 = np.array([[13.98844779, -8.16167706, -6.59563159, -36.08851371],
                               [29.64694411, 0.23962633, -18.06257451, -39.94901405],
                               [30.28692615, -6.76669921, -7.73357544, -41.12568209],
                               [10.37143676, -4.61269186, 2.51874206, -25.88749230],
                               [12.08975513, 27.11544952, -7.29352457, -18.87131866],
                               [-9.01089669, 3.00289501, -2.84874770, -10.21468101],
                               [5.83987146, 10.93603432, 0.36092794, -13.70067068],
                               [6.90323268, -4.14593159, -0.00762729, -17.43880656],
                               [9.17023758, 0.73628108, 3.50571454, -22.188795867],
                               [12.91387513, 9.54066670, -1.36518077, -6.24993485]])

        loss1, grad1 = structured_svm(theta, multi_X, multi_y, 0.)
        loss2, grad2 = structured_svm(theta, multi_X, multi_y, 0.5)
        loss3, grad3 = structured_svm(theta, multi_X, multi_y, 1.)

        grader.requireIsEqual(12.23053697, loss1)
        grader.requireIsEqual(true_grad1, grad1[row_idx, :][:, col_idx])
        grader.requireIsEqual(20847.31323018, loss2)
        grader.requireIsEqual(true_grad2, grad2[row_idx, :][:, col_idx])
        grader.requireIsEqual(41682.39592339, loss3)
        grader.requireIsEqual(true_grad3, grad3[row_idx, :][:, col_idx])


    def prob4_c():
        test_X = multi_X[:3, :]
        test_X[1, :] = np.mean(multi_X, axis=0)
        classifier = classifier_submitted.LinearSVM()
        classifier.theta = np.ones((num_features, num_classes)) / num_features
        grader.requireIsEqual(np.array([0, 0, 0]), classifier.predict(test_X))

        classifier.theta = np.zeros((num_features, num_classes))
        for j in range(num_classes):
            for i in range(num_features):
                x = i * j * 0.5 / num_features - 0.5 * j
                classifier.theta[i, j] = np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
        grader.requireIsEqual(np.array([0, 9, 9]), classifier.predict(test_X))


    grader.addPart("4.A", lambda: prob4_ab(svm_submitted.svm_loss_naive), 5)
    grader.addPart("4.B", lambda: prob4_ab(svm_submitted.svm_loss_vectorized), 10)
    grader.addPart("4.C", prob4_c, 5)

    grader.grade()
