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



#print(df.columns)
print(df_columns)

# <markdowncell>

# ###Data preparation for classification



#Features that are being considered
f1 = [u'SubConId', u'SubDuration', u'SubNumTotIssues', u'SubNumTotObs', u'SubNumSafeIssues', u'SubNumQAQC', u'SubNumSafe', u'SubNumQAQCHighNComplete', u'SubNumQAQCMediumNComplete', u'SubNumQAQCLowNComplete', u'SubNumQAQCNCompleteIssues', u'SubNumQAQCHighNManuf', u'SubNumQAQCMediumNManuf', u'SubNumQAQCLowNManuf', u'SubNumQAQCNManufIssues', u'SubNumQAQCHighDef', u'SubNumQAQCMediumDef', u'SubNumQAQCLowNDef', u'SubNumQAQCNDefIssues', u'SubNumQAQCHighNConDoc', u'SubNumQAQCMediumNConDoc', u'SubNumQAQCLowNConDoc', u'SubNumQAQCNConDocIssues', u'SubNumQAQCHigh', u'SubNumQAQCMedium', u'SubNumQAQCLow', u'SubNumQAQCIssues', u'ProjectZip', u'ProjDuration', u'ProjNumTotIssues']

f2 = ['SubQualityRate']

##Train Test Split
train, test = train_test_split(df, test_size = 0.2) #split data into train set and test set in 80%:20% ratio
y_train = train["SubQualityRate"]
x_train = train[f1]
y_test = test["SubQualityRate"]
x_test = test[f1]

## Feature Normalization
train_mean = train[f1].mean()
train_sd = train[f1].std()
train_meanmaxdiff = train[f1].max() - train[f1].min()

NORMALIZATION_TYPE = "MaxMin"
if NORMALIZATION_TYPE == "MaxMin":
  #Normalize train and test data - MaxMin
  x_train = (x_train - train_mean)/train_meanmaxdiff
  x_test = (x_test - train_mean)/train_meanmaxdiff
if NORMALIZATION_TYPE == "STD":
  #Normalize train and test data - STD
  x_train = (x_train - train_mean)/train_sd
  x_test = (x_test - train_mean)/train_sd

x_train = x_train.as_matrix()
y_train = list(y_train)

# <markdowncell>

# ### Data Augmentation (if some classes are under represented)



## find the data that requires duplication
def findDataForDuplication(x_train, y_train, rating):
  x_temp = []
  y_temp = []
  for i in range(0,len(y_train)):
    if(y_train[i] == rating):
      x_temp.append(x_train[i])
      y_temp.append(y_train[i])
  return [x_temp, y_temp]

## Calculate Rating's histogram
def HistCalc(y_train):
  count_train = {}
  for i in range(0, len(y_train)):
    if y_train[i] in count_train:
      count_train[y_train[i]] += 1
    else:
      count_train[y_train[i]] = 1
  return (count_train)

## Attach duplicate data
def addData(x_train, y_train, x_temp, y_temp, count):
  for i in range(0, count):
    x_train = np.concatenate((x_train, x_temp), axis = 0)
    y_train = np.concatenate((y_train, y_temp), axis = 0)
  return [x_train, y_train]

## Augment Data
def augmentData(x_train, y_train):
  dist = HistCalc(y_train)
  for i in dist:
    x_temp = []
    y_temp = []
    rating = i
    count = int(2000 / dist[i])
    #print(count)
    [x_temp, y_temp] = findDataForDuplication(x_train, y_train, rating)
    [x_train, y_train] = addData(x_train, y_train, x_temp, y_temp, count)
  return [x_train, y_train]
  

print(HistCalc(y_train))
[x_train, y_train] = augmentData(x_train, y_train)
print(HistCalc(y_train))

# <markdowncell>

# ### Classification



CLASSIFICATION = 'rf'
################################################CLASSIFICATION#########
if CLASSIFICATION == 'logit':
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression()
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)

if CLASSIFICATION == 'svm':
  from sklearn import svm
  clf = svm.SVC(kernel='linear')
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'mnb':
  from sklearn.naive_bayes import MultinomialNB
  clf = MultinomialNB()
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'bnb':
  from sklearn.naive_bayes import BernoulliNB
  clf = BernoulliNB()
  clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'gnb':
  from sklearn.naive_bayes import GaussianNB
  clf = GaussianNB()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'rf':
  from sklearn.ensemble import RandomForestClassifier
  clf = RandomForestClassifier(n_estimators=100, criterion='entropy', oob_score=True)
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)

if CLASSIFICATION == 'gbt':
  from sklearn.ensemble import GradientBoostingClassifier
  clf = GradientBoostingClassifier()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)
  
if CLASSIFICATION == 'lda':
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  clf = LinearDiscriminantAnalysis()
  #from sklearn.lda import LDA
  #clf = LDA()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'qda':
  from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
  clf = QuadraticDiscriminantAnalysis()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'dt':
  from sklearn.tree import DecisionTreeClassifier
  clf = DecisionTreeClassifier(max_depth=5)
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'knn':
  from sklearn.neighbors import KNeighborsClassifier
  clf = KNeighborsClassifier(n_neighbors= 10)
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)


if CLASSIFICATION == 'ada':
  from sklearn.ensemble import AdaBoostClassifier
  clf = AdaBoostClassifier()
  clf = clf.fit(x_train, y_train)
  result = clf.predict(x_test)

# <markdowncell>

# ### Error Calculation



from __future__ import division
from sklearn.metrics import mean_squared_error
print("RMSE: ", mean_squared_error(y_test, result))
from sklearn.metrics import accuracy_score
print("ACCURACY :", accuracy_score(y_test, result))
from sklearn.metrics import precision_recall_fscore_support
print("precision_recall_fscore_support: ",precision_recall_fscore_support(y_test, result))
from sklearn.metrics import precision_score
print("precision_score : ", precision_score(y_test, result, average=None))
from sklearn.metrics import recall_score
print("recall_score : ", recall_score(y_test, result, average=None))
from sklearn.metrics import f1_score
print("f1_score : ", f1_score(y_test, result, average=None))

# <markdowncell>

# ### Feature Importance



imp = clf.feature_importances_
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
ax.plot(range(len(imp)),imp)
plt.show()

featureImp = {}
for i in range(0, len(imp)):
  featureImp[f1[i]] = imp[i]
#print(featureImp)
sortedFeatureNames = sorted(featureImp, key=featureImp.get, reverse=True)
sortedFeatureValues = []
#print(sortedFeatureNames)
for w in sortedFeatureNames:
  print(w, "\t : \t", featureImp[w],"#")
  sortedFeatureValues.append(featureImp[w])




