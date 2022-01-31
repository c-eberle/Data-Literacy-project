#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
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
        
        if type(reg_model)==type(sklearn.linear_model.Lasso()) or type(reg_model)==type(sklearn.linear_model.LassoCV()):
            reg = reg_model.fit(train, np.ravel(train_gt))
        else:
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


def visualize_predictions(reg_model, data, gt):
    """
    create scatterplot with predicted latter scores on x-axis and ground truth on y-axis
    """
    
    pred_vals = reg_model.predict(data)
    gt_vals = gt.loc[:,"Ladder score"]
    
    title=get_title(reg_model) + ", alpha = " + str(reg_model.alpha_)
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.scatter(pred_vals, gt_vals)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c="red")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Ladder score")
    plt.xlim([2,8])
    plt.ylim([2,8])
    plt.title(title)
    return

def visualize_coefs(reg_model, indicators, n):
    """
    plot values of largest n coefficients
    """
    coef_df = pd.DataFrame(reg_model.coef_, columns=["Coefficient"], index=indicators)   
    coef_df.sort_values("Coefficient", inplace=True)
    coef_df.iloc[:n,:].plot(kind="barh")
    plt.axvline(x=0, color="grey")
    plt.xlim([-0.02, 0.02]) #xlim is based on a first test with LassoCV, range may need to be adjusted
    plt.title(get_title(reg_model))
    
    return

def get_title(reg_model):
    """
    helper function to plot titles
    """
    if type(reg_model)==type(sklearn.linear_model.LinearRegression()):
        title = "Least Squares"
    elif type(reg_model)==type(sklearn.linear_model.Ridge()) or type(reg_model)==type(sklearn.linear_model.RidgeCV()):
        title = "Ridge"
    elif type(reg_model)==type(sklearn.linear_model.Lasso()) or type(reg_model)==type(sklearn.linear_model.LassoCV()):
        title = "Lasso"
    else:
        title = ""
        
    return title

"""
                            remove before submission
                                ↓↓↓↓↓↓↓↓↓↓↓
############################### some testing ######################################
from sklearn import linear_model

wb_data = pd.read_csv("../data/wb_data.csv", index_col="Country Name")
wb_data_short = pd.read_csv("../data/wb_data_short.csv", index_col="Country Name")
whr_data = pd.read_csv("../data/whr_data.csv", index_col="Country name")

test_size=1
alphas = [0.01, 0.1, 1, 10]
lasso_cv = sklearn.linear_model.LassoCV(alphas=alphas, normalize=True)

loss_list, mean_loss, coef_list, avg_coefs, test_country_list = n_fold_ceval(reg_model=lasso_cv, n=1000, data=wb_data, gt=whr_data, test_size=test_size, scaling="no_scaling")
"""