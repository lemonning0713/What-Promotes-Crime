#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:24:45 2018

@author: yxy
"""

import pandas as pd
import numpy as np
import seaborn as sns

def corr(x):
    # correlation table
    corr = plt.corr()
    ax = plt.axes()
    sns.heatmap(corr,
                annot=True,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    ax.set_title("Correlation Heat Map", size = 15)
    name1="Correlation"+str(x)
    plt.savefig(name1)
    plt.close
    
    # subplots of scatterplots
    sns.pairplot(x)
    name2="Scatterplots"+str(x)
    plt.savefig(name2)
    plt.close
    


def main():
# Correlation Plots
    df=pd.read_csv('crimedata_final_part3.csv')
    df2=df[['Assault','Burglary','Death','Drug','Fraud','Other','Robbery','Sexual','Theft']]
    Var_Corr = df2.corr()
    plt.subplots(figsize=(20,15))
    # plot the heatmap and annotation on it
    sns.heatmap(Var_Corr, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)