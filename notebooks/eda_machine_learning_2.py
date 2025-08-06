#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Data handling
import numpy as np
import pandas as pd
import re

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import graphviz
import optuna
import optuna.visualization as vis
get_ipython().run_line_magic('matplotlib', 'inline')

# Stats
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm
import scipy.stats as st
from scipy.stats import shapiro, norm, chi2_contingency

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error

#lib
from lib.clean_data_functions import clean_ext_version, clean_ext_publisher, clean_repo_publisher

#
from wordcloud import WordCloud,STOPWORDS
from ast import literal_eval
from collections import Counter

# os
import os

# time
import time

import warnings
# warnings.filterwarnings("ignore")    # (Optional)

print("Project has been created with Pandas: " ,pd. __version__," And with Numpy: ",np. __version__)


# In[11]:


import yaml

try:
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)
except:
    print("Yaml configuration file not found!")


# ### 1. Loading

# In[12]:


df = pd.read_csv(config["data"]["clean"]["file_data_cleaned"])
# df.head()


# In[13]:


print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns')


# In[14]:


df.info()


# #### Metadata:
# - **_verified_:**                 check, whether extension's security is breached         (boolean)
# - **_ext_categories_:**           categories of extension                                 (obj)   (multi values)
# - **_ext\_install\_count_:**      total number of installations of extension              (int64)
# - **_ext\_rating_:**              rating of extension (avg of stars rating)               (float64)
# - **_repository_:**               url of repository                                       (obj)
# - **_total\_vulners_:**           number of detected vulnerabilities                      (int64)
# - **_critical\_vulners_:**        number of critical(severity) vulnerabilities            (int64)
# - **_high\_vulners_:**            number of high(severity) vulnerabilities                (int64)
# - **_medium\_vulners_:**          number of medium(severity) vulnerabilities              (int64)
# - **_low\_vulners_:**             number of low(severity) vulnerabilities                 (int64)
# - **_repo\_owner_:**              owner of repository (via column repository)             (obj)
# - **_repo\_name_:**               name of repository (via column repository)              (obj)
# - **_repo\_stars_:**              number of stars of repository (via column repository)   (int64)   
# - **_repo\_forks_:**              number of forks of repository (via column repository)   (int64)   
# - **repo\_languages:**            program languages used (via column repository)          (obj)   (multi values)
# 

# ### 3. EDA

# In[15]:


# cols_num = df.select_dtypes(include = ['float','int']).columns.to_list()
# cols_cat = df.select_dtypes(include = ['object', 'category']).columns.to_list()


# In[16]:


# df.select_dtypes("number").nunique().sort_values(ascending=False)


# In[17]:


# df.select_dtypes(exclude="number").nunique().sort_values(ascending=False)


# In[18]:


# languages_counter=Counter(languages_list)
# languages_counter.most_common(12)

