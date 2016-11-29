# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ### Import libraries



from __future__ import print_function
import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.cross_validation import train_test_split

# <markdowncell>

# ### Read Data



df = pd.read_csv("test.csv")
df = df._get_numeric_data()
df_columns = df.columns

#Fill NA in data with mean/median
df_mean = df.mean().astype(int)
df = df.fillna(df_mean)

#Features to be considered for clustering
features = df.columns
X = df[features]

# <markdowncell>

# ### clustering



df_out = pd.DataFrame() #output dataFrame
CLUSTERING = 'kmeans'
if CLUSTERING == 'kmeans':
  from sklearn.cluster import KMeans
  y_pred = KMeans(n_clusters=5,init='k-means++').fit_predict(X)
  df_out['kmeans'] = y_pred
if CLUSTERING == 'spectral':
  from sklearn.cluster import SpectralClustering
  y_pred = SpectralClustering(n_clusters=5).fit_predict(X)
  df_out['spectral'] = y_pred
if CLUSTERING == 'DBSCAN':
  from sklearn.cluster import DBSCAN
  y_pred = DBSCAN().fit_predict(X)
  df_out['DBSCAN'] = y_pred
if CLUSTERING == 'hierarchical':
  from sklearn.cluster import AgglomerativeClustering
  y_pred = AgglomerativeClustering(n_clusters=5).fit_predict(X)
  df_out['hierarchical'] = y_pred
if CLUSTERING == 'AffinityPropagation':
  from sklearn.cluster import AffinityPropagation
  y_pred = AffinityPropagation().fit_predict(X)
  df_out['AffinityPropagation'] = y_pred



print(df_out)




