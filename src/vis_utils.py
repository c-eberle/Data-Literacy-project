#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 16:48:31 2022

@author: christian
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from collections import Counter

def visualize_predictions(reg_model, data, gt):
    """
    create scatterplot with predicted latter scores on x-axis and ground truth on y-axis
    """
    
    pred_vals = reg_model.predict(data)
    gt_vals = gt.loc[:,"Ladder score"]
    
    #title=get_title(reg_model) + ", alpha = " + str(reg_model.alpha_)
    title=get_title(reg_model)
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
    if get_title(reg_model) == "Lasso":
        coefs = reg_model.coef_
    else:
        coefs = reg_model.coef_[0]
        
    coef_df = pd.DataFrame(coefs, columns=["Coefficient"], index=indicators)
    coef_df.sort_values("Coefficient", inplace=True, key=abs, ascending=False)
    
    title=get_title(reg_model)
    coef_df.iloc[:n,:].plot(kind="barh")
    plt.axvline(x=0, color="grey")
    plt.title(title)
    fname = title + "_coefs.pdf"
    plt.savefig(fname=fname, format="pdf")
    

def get_title(reg_model):
    """
    helper function to plot title string
    """
    if type(reg_model)==type(sklearn.linear_model.LinearRegression()):
        title = "Least Squares"
    elif type(reg_model)==type(sklearn.linear_model.Ridge()) or type(reg_model)==type(sklearn.linear_model.RidgeCV()):
        title = "Ridge"
    elif type(reg_model)==type(sklearn.linear_model.Lasso()) or type(reg_model)==type(sklearn.linear_model.LassoCV()):
        title = "Lasso"
    elif type(reg_model)==type(sklearn.cross_decomposition.PLSRegression()):
        title = "Partial Least Squares"
    else:
        title = ""
        
    return title

def visualize_alphas(alphas, mean_losses):
    """
    plot mean loss for different alpha values
    
    Params: alphas (list), mean_losses(list)
    Note: when comparing multiple models, mean_losses can be a list of lists, where each list contains the mean losses for one model
    """
    fig = plt.figure(figsize=(8,6))
    

    plt.scatter(alphas, mean_losses[0])
    plt.plot(alphas, mean_losses[0], linestyle="solid", color="red", label="Ridge, test loss")

    plt.scatter(alphas, mean_losses[1])
    plt.plot(alphas, mean_losses[1], linestyle="dotted", color="red", label="Ridge, train loss")
    
    plt.scatter(alphas, mean_losses[2])
    plt.plot(alphas, mean_losses[2], linestyle="solid", color="blue", label="Lasso, test loss")
    
    plt.scatter(alphas, mean_losses[3])
    plt.plot(alphas, mean_losses[3], linestyle="dotted", color="blue", label="Lasso, train loss")
    
    plt.xlabel("alpha")
    plt.ylabel("Mean Squared Error")
    
    plt.legend()
    plt.xscale("log")
    plt.show()
    
