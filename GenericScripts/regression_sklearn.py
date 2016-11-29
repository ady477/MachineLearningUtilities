# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ### Import libraries



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# <markdowncell>

# ### Read Data



# read data into a DataFrame
data = pd.read_csv('Advertising.csv', index_col=0)
print(data.head())

# print the shape of the DataFrame
print(data.shape)



# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='Sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='Radio', y='Sales', ax=axs[1])
data.plot(kind='scatter', x='Newspaper', y='Sales', ax=axs[2])

# <markdowncell>

# ### sklearn



# create X and y
feature_cols = data.columns[:-1]
print feature_cols
X = data[feature_cols]
y = data.Sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print lm.intercept_
#print lm.coef_
# pair the feature names with the coefficients
zip(feature_cols, lm.coef_)



# predict for a new observation
lm.predict([100, 25, 25])



# calculate the R-squared
lm.score(X, y)




