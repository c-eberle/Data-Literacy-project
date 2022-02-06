#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from collections import Counter
import vis_utils
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
    
    
def corr_counter_old(corr, threshold=0.85, verbose=False):
    corr_dict = {}
    for name, values in corr.iteritems():
        if verbose:
            print()
            print('\nTarget indicator: ', name)
            print('Correlated Indicators:')
        corr_count = 0
        for i in range(0, corr.shape[1]):   
            if threshold < abs(values[i]) < 1:
                name = corr.columns[i]
                if verbose:
                    print('{name}: {value}'.format(name=name, value=values[i]))
                corr_count += 1

        corr_dict[name] = corr_count
    return corr_dict

def corr_counter(corr):
    corr_dict = {}
    for name, values in corr.iteritems():
        for i in range(0, corr.shape[1]):   
            corr_dict[name] = sum(abs(values))
    return corr_dict


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

"""
                            remove before submission
                                ↓↓↓↓↓↓↓↓↓↓↓
############################### some testing ######################################

from sklearn import linear_model
import vis_utils

wb_data = pd.read_csv("../data/wb_data.csv", index_col="Country Name")
wb_data_short = pd.read_csv("../data/wb_data_short.csv", index_col="Country Name")
whr_data = pd.read_csv("../data/whr_data.csv", index_col="Country name")

test_size=30
lasso = sklearn.linear_model.Lasso()

loss_list, mean_loss, coef_list, avg_coefs = n_fold_ceval(reg_model=lasso, n=1000, data=wb_data, gt=whr_data, test_size=test_size, scaling="normalize")

vis_utils.visualize_coefs(lasso, wb_data.columns.values, 10)

from sklearn import linear_model

wb_data = pd.read_csv("../data/wb_data.csv", index_col="Country Name")
wb_data_short = pd.read_csv("../data/wb_data_short.csv", index_col="Country Name")
whr_data = pd.read_csv("../data/whr_data.csv", index_col="Country name")

test_size=1
ridge = sklearn.linear_model.Ridge()
loss_list, mean_loss, coef_list, avg_coefs = n_fold_ceval(reg_model=ridge, n=1000, data=wb_data, gt=whr_data, test_size=test_size, scaling="normalize")
"""
from sklearn.cross_decomposition import PLSRegression
pls = pls = PLSRegression(n_components=3)

wb_data = pd.read_csv("../data/wb_data.csv", index_col="Country Name")
wb_data_short = pd.read_csv("../data/wb_data_short.csv", index_col="Country Name")
whr_data = pd.read_csv("../data/whr_data.csv", index_col="Country name")

wb_data_red = wb_data.copy(deep=True)
test_size=30

for i in range(0, 10):
    _, pls_mean_loss, _, _, pls_adj_r_sq = n_fold_ceval(reg_model=pls, n=1000, data=wb_data_red, gt=whr_data, 
                                                                        test_size=test_size, scaling="normalize", calc_adj_r_squared=True)
    largest_coef = get_largest_coefs(pls, wb_data_red.columns.values, 1).index.values
    print("Mean loss: ", pls_mean_loss)
    print("Adjusted R-Squared: ", pls_adj_r_sq)
    print("Removing coefficient ", largest_coef, "\n")
    wb_data_red = wb_data_red.drop(largest_coef, axis=1)

