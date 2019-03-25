#encoding=utf-8

import codecs
import sklearn
import os
import sklearn.preprocessing
from os.path import join
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import tensorflow as tf
from itertools import chain
from sklearn import linear_model
import statsmodels.api as sm

__author__ = "future_chi"

def select_gene(gene_names, df, method, feature_sel, save=False):
    if method == 'random':
        random_genes = np.random.choice(gene_names, (feature_sel)).tolist()
    elif method == 'HVG1':
        random_genes = HVG1(df, feature_sel)
    # elif method == 'HVG2':
    else:
        random_genes = gene_names
    if save:
        random_genes_path = os.path.join("./{}/{}/select_gene_names.csv".format(method, feature_sel))
        with codecs.open(random_genes_path, "w", encoding="utf-8") as f:
            for la in random_genes:
                f.write(str(la) + "\n")
    return random_genes

def HVG1(df, feature_sel):
    vars = np.var(df.values, axis=0)
    standard_vars = np.sqrt(vars)
    mean = np.mean(df.values, axis=0)
    x = np.divide(standard_vars, mean)
    top_vars_index = x.argsort()[-feature_sel:][::-1]
    return df.columns[top_vars_index]



def sizefactor(df):
    log = np.log(df+1)
    col_mean = np.mean(log, axis=0)
    col_mean = np.expand_dims(np.exp(col_mean), 0)
    div = np.divide(np.exp(log), col_mean)
    sf = np.median(div, axis=1)
    sf = np.expand_dims(sf, 1)
    div = np.log(np.divide(df, sf)+1)
    return div

def row_normal(data, factor=1e6):
    row_sum = np.sum(data, axis=1)
    row_sum = np.expand_dims(row_sum, 1)
    div = np.divide(data, row_sum)
    div = np.log(1 + factor * div)
    return div

