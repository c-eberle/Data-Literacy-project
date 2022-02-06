#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from collections import Counter
from . import vis_utils
from sklearn import preprocessing, linear_model


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


def n_fold_ceval(reg_model, n, data, gt, test_size, scaling, calc_adj_r_squared=False):
    """
    perform n-fold validation
    
    args: number of validations (n), dataset of indicators (data), groundtruth data (gt), size of test set (test size), loss function (loss)
    returns: list of losses for the n models, mean loss, 
             list of arrays with the coeffcients for the n models, the average size for each coefficient (rounded)
    """
    loss_list = []
    train_loss_list = []
    coef_list = []
    adj_r_squared_list =[]
    
    assert scaling in ["normalize", "standardize", "no_scaling"]

    if scaling == "normalize":
        data[:] = sklearn.preprocessing.normalize(data, axis=0)
    elif scaling == "standardize":
        data[:] = sklearn.preprocessing.StandardScaler().fit_transform(data)
    
    for i in range(0,n):
        train, test, train_gt, test_gt = split_data(data, gt, test_size)
        
        if type(reg_model)==type(sklearn.linear_model.Lasso()) or type(reg_model)==type(sklearn.linear_model.LassoCV()):
            reg = reg_model.fit(train, np.ravel(train_gt))
        else:
            reg = reg_model.fit(train, train_gt)    

        test_pred = reg.predict(test)
        train_pred = reg.predict(train)
        loss = sklearn.metrics.mean_squared_error(test_gt, test_pred)
        train_loss = sklearn.metrics.mean_squared_error(train_gt, train_pred)
        coefs = reg.coef_

        loss_list.append(loss)
        train_loss_list.append(train_loss)
        coef_list.append(coefs)

        # calculate adjusted r-squared
        adj_r_squared = 1 - ( 1- reg_model.score(train, train_gt) ) * ( len(train_gt) - 1 ) / ( len(train_gt) - train.shape[1] - 1 )
        adj_r_squared_list.append(adj_r_squared)


    # calculate mean loss
    loss_arr = np.array(loss_list)
    mean_loss = loss_arr.mean()

    # calculate mean train loss
    train_loss_arr = np.array(train_loss_list)
    mean_train_loss = train_loss_arr.mean()

    # calculate and round average coefficients
    avg_coefs = np.around(np.mean(coef_list, axis=0), 4)[0]

    # calculate adjusted r-squared 
    avg_adj_r_squared = np.mean(adj_r_squared_list)


    if calc_adj_r_squared:
        return loss_list, mean_loss, mean_train_loss, coef_list, avg_coefs, avg_adj_r_squared

        
    return loss_list, mean_loss, coef_list, avg_coefs


def print_bad_predictions(reg_model, data, gt, threshold):
    """
    print list of predicted ladder score vs. ground truth for all countries where
    the prediction is worse than a given threshold
    
    returns: list of tuples [(pred_1, gt_1), ... , (pred_n, gt_n)]
    """
    pred_arr = reg_model.predict(data)
    pred_vs_gt = gt.copy(deep=True)
    pred_vs_gt["Prediction"] = pred_arr
    for country in pred_vs_gt.index:
        if abs(pred_vs_gt.loc[country,"Ladder score"] - pred_vs_gt.loc[country,"Prediction"]) > threshold:
            print(pred_vs_gt.loc[country], "\n")
    return

def get_largest_coefs(reg_model, indicators, n):
    """
    get values of largest n coefficients
    """
    if vis_utils.get_title(reg_model) == "Lasso":
        coefs = reg_model.coef_
    else:
        coefs = reg_model.coef_[0]
        
    coef_df = pd.DataFrame(coefs, columns=["Coefficient"], index=indicators)
    coef_df.sort_values("Coefficient", inplace=True, key=abs, ascending=False)
    
    return coef_df.iloc[:n]


def get_largest_coef_pls(pls_model, indicators):
    """
    get values of largest coefficient
    (partial least squares needs its own function because the coefficients are
     a linear combination of indicators)
    """
    sum_components = np.zeros([pls_model.x_weights_.shape[0]])
    # take weighted sum of components
    for i in range(0,4): 
        sum_components += pls_model.y_weights_[0,i] * pls_model.x_weights_[:,i]
        
    df_sum_comp = pd.DataFrame(sum_components, columns=["Coefficient"], index=indicators)
    df_sum_comp.sort_values("Coefficient", inplace=True, key=abs, ascending=False)
    
    return df_sum_comp
