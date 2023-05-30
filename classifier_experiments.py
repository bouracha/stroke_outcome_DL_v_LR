import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import random
from datetime import datetime

def experiment(X_train, X_test, y_train, y_test, classifier):
    classifier.fit(X_train, y_train)

    y_train_hat = classifier.predict(X_train)
    y_test_hat = classifier.predict(X_test)
    y_test_hat = np.where(y_test_hat > 0.5, 1, 0)

    acc = accuracy_score(y_test, y_test_hat)
    f1 = f1_score(y_test, y_test_hat)
    recall = recall_score(y_test, y_test_hat)

    fpr_train, tpr_train, thresholds = roc_curve(y_train, y_train_hat)
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_test_hat)
    train_auc = auc(fpr_train, tpr_train)
    test_auc = auc(fpr_test, tpr_test)

    return train_auc, test_auc


def inner_cv(X_train_CV, y_train_CV, model, hyperparameters_list, train_size, num_inner_trials=5):
    max_CV_AUC = 0.0
    max_CV_AUC_std = 0.0
    for hyperparameters in hyperparameters_list:
        if model == "SVM":
            classifier = svm.SVC(**hyperparameters)
        elif model == 'ridge':
            classifier = Ridge(**hyperparameters)
        elif model == "lr":
            classifier = LogisticRegression(**hyperparameters)

        CV_AUCs = []
        for i in range(0, num_inner_trials):
            X_train, X_CV, y_train, y_CV = train_test_split(X_train_CV, y_train_CV, test_size=0.20, random_state=i, stratify=y_train_CV)
            # print(y_train.mean(), y_CV.mean())

            X_train = X_train[:train_size]
            y_train = y_train[:train_size]

            train_AUC, CV_AUC = experiment(X_train, X_CV, y_train[:train_size], y_CV, classifier)
            CV_AUCs.append(CV_AUC)

        if np.mean(CV_AUCs) > max_CV_AUC:
            max_CV_AUC = np.mean(CV_AUCs)
            max_CV_AUC_std = np.std(CV_AUCs)
            best_hyperparmeters = hyperparameters

    return max_CV_AUC, max_CV_AUC_std, classifier, best_hyperparmeters


def nested_cross_validation(X, y, model, hyperparameters_list, train_size, num_outer_trials=5, num_inner_trials=5):
    max_test_AUC = 0.0
    max_test_AUC_std = 0.0
    train_AUCs = []
    test_AUCs = []
    for i in range(0, num_outer_trials):
        print('Outer loop iteration: %i/%i' % (i, num_outer_trials), end='\r')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=i, stratify=y)  # random.seed(datetime.now()))
        # print(y_train.mean(), y_test.mean())

        max_CV_AUC, max_CV_AUC_std, classifier, best_hyperparmeters = inner_cv(X_train, y_train, model,
                                                                               hyperparameters_list, train_size,
                                                                               num_inner_trials=num_inner_trials)

        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

        if model == "SVM":
            classifier = svm.SVC(**best_hyperparmeters)
        elif model == 'ridge':
            classifier = Ridge(**best_hyperparmeters)
        elif model == "lr":
            classifier = LogisticRegression(**best_hyperparmeters)
        # print(best_hyperparmeters)
        train_AUC, test_AUC = experiment(X_train, X_test, y_train, y_test, classifier)
        train_AUCs.append(train_AUC)
        test_AUCs.append(test_AUC)

    return train_AUCs, test_AUCs

def run_experiments(data, labels, train_sizes, model, hyperparameters_list, num_outer_trials=5, num_inner_trials=5):
    for train_size in train_sizes:

        train_AUCs, test_AUCs = nested_cross_validation(data, labels, model, hyperparameters_list, train_size, num_outer_trials=num_outer_trials, num_inner_trials=num_inner_trials)

        print("#############################################################################################")
        print("Train size: {}, Train reconstruction AUC:{:.4f},{:.4f}".format(train_size, np.mean(train_AUCs), np.std(train_AUCs)))
        print("Test size: {}, Test reconstruction AUC:{:.4f},{:.4f}".format(train_size, np.mean(test_AUCs), np.std(test_AUCs)))

def set_SVM_hyperparameters(param_grid = {'C': [0.1, 1, 10, 100, 1000],'gamma': [1, 0.1, 0.01, 0.001, 0.0001],'kernel': ['rbf']}):
    hyper_list = []
    for key in param_grid.keys():
        hyper_list.append(param_grid[key])

    hyperparameters_list = []
    for i in range(len(hyper_list[0])):
        for j in range(len(hyper_list[1])):
            mydict = {}
            mydict['C'] = hyper_list[0][i]
            mydict['gamma'] = hyper_list[1][j]
            mydict['kernel'] = hyper_list[2][0]
            hyperparameters_list.append(mydict)

    return hyperparameters_list

def set_lr_hyperparameters(param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]}):
    hyper_list = []
    for key in param_grid.keys():
        hyper_list.append(param_grid[key])

    hyperparameters_list = []
    for i in range(len(hyper_list[0])):
        mydict = {}
        mydict['C'] = hyper_list[0][i]
        mydict['solver'] = 'liblinear'
        hyperparameters_list.append(mydict)
    hyperparameters_list.append({'penalty': 'none'})

    return hyperparameters_list
