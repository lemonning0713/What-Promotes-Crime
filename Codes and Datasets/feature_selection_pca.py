#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 22:11:07 2018

@author: yxy
"""
import pandas as pd
import sklearn.datasets as ds
import numpy as np
from sklearn.decomposition import PCA
#from sklearn.datasets import load_iris
#from sklearn.preprocessing import scale
import scipy
import matplotlib.pyplot as plt
from numpy import *
import sklearn
from sklearn import preprocessing

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy import stats

from scipy.sparse import *
from scipy import *

from sklearn.decomposition import PCA

import seaborn as sns; sns.set()  # for plot styling
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing
from sklearn import multiclass

import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import MNLogit

###
def DBScanClustering(DFArray, eps, min_samples=10):
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    DBSCAN_fit = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(DFArray)
    core_samples_mask = np.zeros_like(DBSCAN_fit.labels_, dtype=bool)
    core_samples_mask[DBSCAN_fit.core_sample_indices_] = True
    labels = DBSCAN_fit.labels_
    #print(labels)
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  
    ## Plot the result
    ## Plot code from here:
    ## http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = DFArray[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=2)

        xy = DFArray[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('Mean Robbery Occurence')
    plt.ylabel('School Rate')
    plt.show()
    plt.close()

    ss=metrics.silhouette_score(DFArray, labels, metric='euclidean')
    print("Silouette score of this DBscan is",ss)

def kmeansClustering(DFArray, DistanceMeasure="L2", k=3):
    DF=DFArray
    kmeansResults=KMeans(n_clusters=k,init='k-means++', verbose=1, algorithm="full")
    kmeansResults.fit(DF)
    #print("Centers: ", kmeansResults.cluster_centers_)  
    #print("Labels: ", kmeansResults.labels_)
    #print("Intertia (L2norm dist):", kmeansResults.inertia_)


def MakeBlobs_kmeans(X,k):
    ## Call kmeans on the blobs
    kmeansResultsB=KMeans(n_clusters=k, verbose=1, precompute_distances=True)
    kmeansResultsB.fit(X)
    y_kmeans = kmeansResultsB.predict(X)
    #print("Centers: ", kmeansResultsB.cluster_centers_)  
    #print("Labels: ", kmeansResultsB.labels_)
    #print("Intertia (L2norm dist):", kmeansResultsB.inertia_)
    ## VIS
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeansResultsB.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.title("Plot of Kmeans")
    plt.show()
    plt.close()
    #return X
    
    labels=kmeansResultsB.labels_
    ss=metrics.silhouette_score(X, labels, metric='euclidean')
    print("Silouette score of Kmeans is",ss)

def Makeanarray(x1,x2):
    a0 =[]
    for i in range(len(x1)):
        a0.append([x1[i],x2[i]])
    a0=np.asarray(a0)
    return a0

def ward(X):
    linkage_matrix = linkage(X, 'ward')
    figure = plt.figure(figsize=(7.5, 5))
    dendrogram(
            linkage_matrix,
            color_threshold=0,
    )
    plt.title('Hierarchical Clustering Dendrogram (Ward)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.show()
"""
data=pd.read_csv('crimedata_moon_1130_new.csv')
data["as_cnt"] = np.round(data['Assault']* data['astotal'])
data["bur_cnt"] = np.round(data['Burglary']* data['burtotal'])
data["death_cnt"] = np.round(data["Death"]* data['detotal'])
data["drug_cnt"] = np.round(data['Drug']* data['drugtotal'])
data["fraud_cnt"] = np.round(data['Fraud']* data['fratotal'])
data["rob_cnt"]= np.round(data['Robbery']* data['robtotal'])
data["sex_cnt"]= np.round(data['Sexual']* data['sextotal'])
data["theft_cnt"]= np.round(data['Theft']* data['thetotal'])

#x=data[['Moon_Illumination','MaxTemperature','MinTemperature','Day','Month','Weekend']]
#y=data[["as_cnt","bur_cnt","death_cnt","drug_cnt","fraud_cnt","rob_cnt","sex_cnt","theft_cnt"]]


pca=sklearn.decomposition.PCA(n_components=2)
pca.fit_transform(x)
    #pca.fit_transform(Z)
#plt.semilogy(pca.explained_variance_ratio_, '--o')
#plt.semilogy(pca.explained_variance_ratio_.cumsum(), '-o');

data_scaled = pd.DataFrame(preprocessing.scale(x),columns = x.columns) 
a=pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2'])
a.to_csv('pca_fts.csv')

print ("variance:",pca.explained_variance_)
print ('variance ratio',pca.explained_variance_ratio_)
print ('ratio sum:',pca.explained_variance_ratio_.cumsum())
print ('Score',pca.score(x))
#pca.explained_variance_ratio_
    #pd.tools.plotting.scatter_matrix(x)
"""
df=pd.read_csv('crimedata_final_part3.csv')
df['Total_Sch_Cnt'] = df['Pri_Sch_Cnt'] + df['Pub_Sch_Cnt'] + df['Uni_Cnt']
df['Total_Sch_Pop'] = df['Pri_Sch_Pop'] + df['Pub_Sch_Pop'] + df['Uni_Pop']
df['Total_Sch_Rate'] = df['Total_Sch_Cnt']/df['Total_Sch_Pop']
cr_bar_l=[]
for i in range(0,110248):
    cr_bar_l.append(np.mean(df.iloc[i,5:13]))
#print(cr_bar_l)
df['cr_bar']=cr_bar_l

"""
x=df[['Day','Month','Weekend','Moon_Illumination','MaxTemperature','MinTemperature','Total_Sch_Rate']]
pca=sklearn.decomposition.PCA(n_components=2)
pca.fit_transform(x)

data_scaled = pd.DataFrame(preprocessing.scale(x),columns = x.columns) 
a=pd.DataFrame(pca.components_,columns=data_scaled.columns,index = ['PC-1','PC-2'])
a.to_csv('pca_fts.csv')

print ("variance:",pca.explained_variance_)
print ('variance ratio',pca.explained_variance_ratio_)
print ('ratio sum:',pca.explained_variance_ratio_.cumsum())
print ('Score',pca.score(x))
"""

###
ar1=df.iloc[:,:].groupby(df['Date']).mean()

# K-means
ark=Makeanarray(ar1['Assault'],ar1['Moon_Illumination'])
MakeBlobs_kmeans(ark,2)

# DBScan
df_Detroit=df[df['City']=='Detroit']
df_Detroit=df_Detroit[df_Detroit['Robbery']!=0]
df_Detroit=df_Detroit.reset_index()
#ar1=Makeanarray(df_Detroit['Robbery'],df_Detroit['Pub_Sch_Pop'])
arr1=Makeanarray(ar1['Robbery'],ar1['Total_Sch_Rate'])
DBScanClustering(arr1,0.001)

# Hierarchical
arh=Makeanarray(ar1['Drug'],ar1['MaxTemperature'])
ward(arh)
