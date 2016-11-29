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

# ### 1 Variable



# this is the standard import if you're using "formula notation" (similar to R)
import statsmodels.formula.api as smf
# create a fitted model in one line
lm = smf.ols(formula='Sales ~ TV', data=data).fit()
# print the coefficients
lm.params
# Predictions
X_new = pd.DataFrame({'TV': [50]}) #use the DataFrame that you want to predict
prediction = lm.predict(X_new)
print prediction



#Plotting
# create a DataFrame with the minimum and maximum values of TV
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
# make predictions for those x values and store them
preds = lm.predict(X_new)
# first, plot the observed data
data.plot(kind='scatter', x='TV', y='Sales')
# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)



# print the confidence intervals for the model coefficients
print "confidence interval: \n", lm.conf_int()
# print the p-values for the model coefficients
print "\n Pvalues: \n", lm.pvalues
# print the R-squared value for the model
print "\n R-squared values: \n",lm.rsquared

# <markdowncell>

# ### Multiple Linear Regression



# create a fitted model with all three features
lm = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()

# print the coefficients
lm.params



# print a summary of the fitted model
lm.summary()










