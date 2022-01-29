#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from collections import Counter


def split_data(data, gt, test_size=30):
    """
    split dataset into train and test set
    
    returns: tuple of numpy arrays (train_set, test_set)
    """
    test_set = data.sample(n=test_size)
    test_country_names = list(test_set.index.values)
    train_set = data.drop(labels=test_country_names)
    
    test_gt = gt.loc[test_set.index.values]
    train_gt = gt.drop(labels=test_country_names)
    
    return train_set, test_set, train_gt, test_gt


def n_fold_ceval(reg_model, n, data, gt, test_size, scaling):
    """
    perform n-fold validation
    
    args: number of validations (n), dataset of indicators (data), groundtruth data (gt), size of test set (test size), loss function (loss)
    returns: list of losses for the n models, mean loss, 
             list of arrays with the coeffcients for the n models, the average size for each coefficient (rounded)
    """
    loss_list = []
    coef_list = []

    assert scaling in ["normalize", "standardize", "no_scaling"]

    if scaling == "normalize":
        data[:] = sklearn.preprocessing.normalize(data, axis=0)
    elif scaling == "standardize":
        data[:] = sklearn.preprocessing.StandardScaler().fit_transform(data)
    
    for i in range(0,n):
        train, test, train_gt, test_gt = split_data(data, gt, test_size)
        
        reg = reg_model.fit(train, train_gt)
        test_pred = reg.predict(test)
        loss = sklearn.metrics.mean_squared_error(test_gt, test_pred)
        coefs = reg.coef_
        loss_list.append(loss)
        coef_list.append(coefs)

    # calculate mean loss
    loss_arr = np.array(loss_list)
    mean_loss = loss_arr.mean()

    # calculate and round average coefficients
    avg_coefs = np.around(np.mean(coef_list, axis=0), 4)[0]
        
    return loss_list, mean_loss, coef_list, avg_coefs
