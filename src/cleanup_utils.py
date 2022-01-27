#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter


def count_uniquely_sparse_countries(df, sparsity_thresh):
    """
    a function that iterates through the indicators and counts a country 
    if it is the only one without data for a given indicator.
    
    returns: the occurences
    """
    sparse_countries = []
    for i in range(0, len(df.columns)):
        num_values = sum(df[df.columns[i]]=="..")
        if num_values == sparsity_thresh:
            sparse_countries.append(list(df[df[df.columns[i]]==".."].index.values))
            
    # merges the list of lists into single list        
    sparse_countries_unified = sum(sparse_countries, [])
    
    #counts and ouputs occurances of countries
    print(Counter(sparse_countries_unified))



def delete_sparse_indicators(df, sparsity_thresh):
    """
    a function that deletes indicators from the data, if they are missing data 
    from more countries than specified by the sparsity_threshold.
    
    returns: the dense data
    """
    sparse_indicators = []
    for i in range(0, len(df.columns)):
        num_values = sum(df[df.columns[i]]=="..")
        if num_values > sparsity_thresh:
            sparse_indicators.append(df.columns[i])
            
    dense_df = df.drop(columns=sparse_indicators)
    
    return dense_df