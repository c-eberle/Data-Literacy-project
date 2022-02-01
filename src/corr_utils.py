#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from collections import Counter

def corr_counter(corr):
    corr_dict = {}
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


def ind_removal_sim(num_indicators_list, sample_reps, model, n, data, gt, test_size=30, scaling="normalize"):
    for num_indicators in num_indicators_list:
        mean_errors = []
        mean_coef_size = []
        for j in range(0, sample_reps):
            wb_data_rand_reduced = data.sample(num_indicators, axis=1)
            loss_list, mean_loss, coef_list, avg_coefs = ana_utils.n_fold_ceval(reg_model=model, n=n, data=wb_data_rand_reduced, gt=gt, test_size=test_size, scaling=scaling)
            
            mean_errors.append(mean_loss)

            mean_abs_coef = np.mean(abs(avg_coefs)) * len(avg_coefs)
            mean_coef_size.append(mean_abs_coef)

        print("Number of indicators", num_indicators)
        print("Avg. Loss", np.mean(mean_errors))
        print("Loss STD", np.std(mean_errors))
        print(" Avg. Total Coef. Size", np.mean(mean_coef_size), "\n")


def pearsons_reduction(data, target_size): 
    reduced_data = data.copy(deep=True)
    remove_limit = len(data.columns) - target_size
    
    for i in range(0, remove_limit):
        indicator_corr = reduced_data.corr(method="pearson")
        corr_dict = corr_counter(indicator_corr)

        most_correlated_indicator = max(corr_dict, key=corr_dict.get)
        #print(max(corr_dict.values()))
        reduced_data.drop(columns=most_correlated_indicator, inplace=True)

    return reduced_data


def sklearn_vif(exogs, data):

    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}
    epsilon = 1e6

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared + epsilon)
        vif_dict[exog] = round(vif, 1)

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = round(tolerance,4)

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif


def vif_reduction(data, drop_batch, target_size):
    reduced_data = data.copy(deep=True)
    iterations = round((len(data.columns) - target_size) / drop_batch)
    
    for i in range(0, iterations):
        wb_vif = sklearn_vif(reduced_data.columns, reduced_data)

        drop_list = list(wb_vif["VIF"].sort_values()[-drop_batch:].index)
        reduced_data.drop(columns=drop_list, inplace=True)

    return reduced_data



