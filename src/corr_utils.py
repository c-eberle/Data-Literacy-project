#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from collections import Counter
from . import ana_utils

def corr_counter(corr, sum_of_squares=False):
    corr_dict = {}
    if sum_of_squares:
            for name, values in corr.iteritems():
                for i in range(0, corr.shape[1]):   
                    corr_dict[name] = sum(values**2)
    else:
        for name, values in corr.iteritems():
            for i in range(0, corr.shape[1]):   
                corr_dict[name] = sum(abs(values))
    return corr_dict

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


def coef_removal_sim(num_indicators, sample_reps, model, n, data, gt, test_size=30, scaling="normalize",verbose=True):
    mean_errors0 = []
    mean_errors1 = []
    mean_errors5 = []
    for j in range(0, sample_reps):
        wb_data_rand_reduced = data.sample(num_indicators, axis=1)

        for i in range(0, 6):
            loss_list, mean_loss, mean_train_loss, coef_list, avg_coefs, adjusted_r_squared = ana_utils.n_fold_ceval(reg_model=model, n=n, data=wb_data_rand_reduced, gt=gt, test_size=test_size, scaling=scaling, calc_adj_r_squared=True)
            largest_coef = ana_utils.get_largest_coefs(model, wb_data_rand_reduced.columns.values, 1).index.values
            wb_data_rand_reduced = wb_data_rand_reduced.drop(largest_coef, axis=1)

            if i == 0:
                mean_errors0.append(mean_loss)
            if i == 1:
                mean_errors1.append(mean_loss)
            if i == 5:
                mean_errors5.append(mean_loss)

    print(np.mean(mean_errors0))
    print(np.mean(mean_errors1))
    print(np.mean(mean_errors5))

    


def ind_removal_sim(num_indicators_list, sample_reps, model, n, data, gt, test_size=30, scaling="normalize"):
    mean_loss_list, std_list = [], []
    mean_train_loss_list = []
    for num_indicators in num_indicators_list:
        mean_errors = []
        mean_train_errors = []
        mean_coef_size = []
        for j in range(0, sample_reps):
            wb_data_rand_reduced = data.sample(num_indicators, axis=1)
            loss_list, mean_loss, mean_train_loss, coef_list, avg_coefs, adjusted_r_squared = ana_utils.n_fold_ceval(reg_model=model, n=n, data=wb_data_rand_reduced, gt=gt, test_size=test_size, scaling=scaling, calc_adj_r_squared=True)
            
            mean_errors.append(mean_loss)
            mean_train_errors.append(mean_train_loss)

            mean_abs_coef = np.mean(abs(avg_coefs)) * len(avg_coefs)
            mean_coef_size.append(mean_abs_coef)

        mean_loss_list.append(np.mean(mean_errors))
        mean_train_loss_list.append(np.mean(mean_train_errors))

        std_list.append(np.std(mean_errors))
        print("Number of indicators", num_indicators)
        print("Avg. Loss", np.mean(mean_errors))
        print("Avg. Train loss", np.mean(mean_train_errors))
        print("Loss STD", np.std(mean_errors))
        print("Avg. Total Coef. Size", np.mean(mean_coef_size), "\n")

    return mean_loss_list, mean_train_loss_list, std_list


def pearsons_reduction(data, target_size, sum_of_squares = False): 
    reduced_data = data.copy(deep=True)
    remove_limit = len(data.columns) - target_size
    corr_sorted_column_list = []
    
    for i in range(0, remove_limit):
        indicator_corr = reduced_data.corr(method="pearson")
        corr_dict = corr_counter(indicator_corr)

        most_correlated_indicator = max(corr_dict, key=corr_dict.get)
        #print(max(corr_dict.values()))
        reduced_data.drop(columns=most_correlated_indicator, inplace=True)
        corr_sorted_column_list.append(most_correlated_indicator)

    return corr_sorted_column_list


def multi_ceval(num_indicator_list, data, verbose, **ceval_kwargs):
    pearson_mean_loss_list = []
    pearson_mean_train_loss_list = []
    pearson_std_list = []

    for i in num_indicator_list:

        ceval_kwargs["data"] = data.iloc[:,:i]
        loss_list, mean_loss, mean_train_loss, coef_list, avg_coefs, adj_r_squared  = ana_utils.n_fold_ceval(**ceval_kwargs)
        pearson_mean_loss_list.append(mean_loss)
        pearson_mean_train_loss_list.append(mean_train_loss)
        pearson_std_list.append(np.std(loss_list))
        if verbose:#
            print("Number of indicators:", i)
            print("Mean loss:", mean_loss)
            print("Mean train loss:", mean_train_loss)
            print("STD of the Loss:", np.std(loss_list))
            print("Adjusted R-Squared: ", adj_r_squared)
            print("The average size of the coefficients:", np.mean(abs(avg_coefs)), "\n")

    return pearson_mean_loss_list, pearson_mean_train_loss_list, pearson_std_list

    

def sklearn_vif(exogs, data):

    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}


    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        np.seterr(divide='ignore') # ignore division by zero errors in case of perfect correlation

        vif = 1/(1 - r_squared)
        vif_dict[exog] = round(vif, 1)

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = round(tolerance,4)

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif


def vif_reduction(data, target_size):
    reduced_data = data.copy(deep=True)
    remove_limit = len(data.columns) - target_size
    vif_sorted_column_list = []
    
    for i in range(0, remove_limit):
        wb_vif = sklearn_vif(reduced_data.columns, reduced_data)
        ind_to_drop = wb_vif["VIF"].idxmax()
        reduced_data.drop(columns=ind_to_drop, inplace=True)
        #print(ind_to_drop)
        vif_sorted_column_list.append(ind_to_drop)

    return vif_sorted_column_list



"""
############ Testing and debugging area, remove before submission ############
wb_data = pd.read_csv("../data/wb_data.csv", index_col="Country Name")
wb_data_short = pd.read_csv("../data/wb_data_short.csv", index_col="Country Name")
whr_data = pd.read_csv("../data/whr_data.csv", index_col="Country name")
wb_data_pear_sorted = pd.read_csv("../data/wb_data_pear_sorted.csv").set_index(wb_data.index, inplace=True)
wb_data_vif_sorted = pd.read_csv("../data/wb_data_vif_sorted.csv").set_index(wb_data.index, inplace=True)

ridge = sklearn.linear_model.Ridge()
test_size = 30
num_indicator_list = [10, 20, 30, 40, 50, 75, 100, 121]
ceval_kwargs = {"reg_model" : ridge, "n" : 2000, "gt" : whr_data, "test_size" : test_size, "scaling" : "normalize", "calc_adj_r_squared" : True}


pearson_mean_loss_list, _ = multi_ceval(num_indicator_list, wb_data_pear_sorted, True, **ceval_kwargs)
"""
