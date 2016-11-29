# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>



from __future__ import print_function
import os
import subprocess
import pandas as pd
import numpy as np



df = pd.read_csv("test.csv")
#df = df._get_numeric_data()
df_columns = df.columns
#Fill NA in data with mean/median
#df_mean = df.mean().astype(int)
df = df.fillna(0)



print((df.columns))

# <markdowncell>

# ## describe dataframe



df.describe()

# <markdowncell>

# ## Merge dataframes



merged = df1.merge(df2, how='inner', left_on="SESSION ID", right_on="Session ID")
merged.describe()

# <markdowncell>

# ## get correlation of all pair of columns in the dataframe



#df.describe()
corr_df = df.corr(method='pearson', min_periods=1)



print(corr_df)




