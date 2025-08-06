#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data handling
import numpy as np
import pandas as pd
import re

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')

#lib
from lib.clean_data_functions import clean_ext_version, clean_ext_publisher, clean_repo_publisher

#
from ast import literal_eval
from collections import Counter

# os
import os

# time
import time

import warnings
warnings.filterwarnings("ignore")    # (Optional)

print("Project has been created with Pandas: " ,pd. __version__," And with Numpy: ",np. __version__)


# In[ ]:





# In[2]:


import yaml

try:
    with open("../config.yaml", "r") as file:
        config = yaml.safe_load(file)
except:
    print("Yaml configuration file not found!")


# ### 1. Loading

# In[ ]:


df = pd.read_csv(config["data"]["raw"]["file_ext_repo"])
# df = df.sort_values(by = ["ext_install_count", "ext_rating"], ascending= False)
df.head()


# In[ ]:


print(f'The dataset has {df.shape[0]} rows and {df.shape[1]} columns')


# In[ ]:


df.info()


# #### Metadata:
# - **_verified_:**                 check, whether extension's security is breached         (boolean)
# - **_ext_name_:**                 name of extension                                       (obj)
# - **_ext_publisher_:**            name of extension'S publisher                           (obj)
# - **_ext_version_:**              current version of extension                            (obj)
# - **_ext_categories_:**           categories of extension                                 (obj)   (multi values)
# - **_ext_tags_:**                 keywords related to extension                           (obj)   (multi values)
# - **_ext\_install\_count_:**      total number of installations of extension              (int64)
# - **_ext\_rating_:**              rating of extension (avg of stars rating)               (float64)
# - **_ext\_last\_updated_:**       timestamp of last update                                (obj)
# - **_repo_publisher_:**           publisher of extension                                  (obj)
# - **_repository_:**               url of repository                                       (obj)
# - **_total_vulnerabilities_:**    number of detected vulnerabilities                      (int64)
# - **_critical_:**                 number of critical(severity) vulnerabilities            (int64)
# - **_high_:**                     number of high(severity) vulnerabilities                (int64)
# - **_medium_:**                   number of medium(severity) vulnerabilities              (int64)
# - **_low_:**                      number of low(severity) vulnerabilities                 (int64)
# - **_repo\_owner_:**              owner of repository (via column repository)             (obj)
# - **_repo\_name_:**               name of repository (via column repository)              (obj)
# - **_repo\_stars_:**              number of stars of repository (via column repository)   (int64)   
# - **_repo\_forks_:**              number of forks of repository (via column repository)   (int64)   
# - **_language_:**                 program languages used (via column repository)          (obj)   (multi values)
# - **_topics_:**                   keywords related to repository (via column repository)  (obj)   (multi values)
# - **_error_:**                    log of fetching repository                              (obj)
# 

# ### 2. Cleaning

# In[ ]:


# Check missing values
df.isna().sum()


# Remove record with values (not null) in "error" column due to unavailability of repository

# In[ ]:


df = df[df["error"].isna()]
print(f"After removing extensions with unavailable repository, the dataset has {df.shape[0]} rows and {df.shape[1]} columns")


# #### Clean columns

# Rename and remove columns for noises and avoiding of overfit ["ext\_tags", "repo\_publisher", "error"]

# In[8]:


# Use  "repo\_owner" over "repo\_publisher" due to correctness from fetching infos directly)
df = df.drop(columns=["ext_name","ext_publisher","ext_tags","ext_last_updated", "repo_publisher", "error"]).rename(columns={"total_vulnerabilities": "total_vulners", "critical": "critical__vulners", "high": "high__vulners", "medium": "medium__vulners", "low": "low__vulners", "language": "repo_languages", "topics": "repo_topics"})


# In[ ]:


df


# In[10]:


# df.info()


# #### Clean data inconsistencies

# In[11]:


#ext_categories
df["ext_categories"] = df["ext_categories"].str.lower()
categories = ""
for cate in df["ext_categories"]:
    categories += cate + ";"

categories_list = categories.split(';')
categories_set  = set(categories_list)

# categories_set


# In[12]:


#ext_version

df["ext_version"] = df["ext_version"].apply(clean_ext_version)
df['ext_version'] = pd.to_numeric(df['ext_version'], errors="coerce")


# In[13]:


#ext_rating
df["ext_rating"] = df["ext_rating"].apply(lambda x: round(x,2) if pd.notna(x) else pd.NA )


# In[ ]:


# df[(df["ext_version"] > 10) & (df["ext_rating"] == 0) & (df["ext_install_count"] < 100) & (df["repo_stars"] < 10)]
# df[(df["ext_rating"] == 0) & (df["ext_install_count"] < 10) & (df["repo_stars"] < 10)]
df


# In[15]:


#repo_languages
# eval the obj[] and transform to str
df["repo_languages"] = df["repo_languages"].apply(lambda x: literal_eval(x) if pd.notna(x) else [])
df["repo_languages"] = df["repo_languages"].apply(lambda x: ';'.join(map(str, x)) if x else x)
df["repo_languages"] = df["repo_languages"].str.lower()



# prog_languages = languages_set


# In[16]:


languages = ""
for lang in df["repo_languages"]:
    if not pd.isna(lang):
        languages += lang + ";"

languages_list = languages.split(';')
languages_set  = set(languages_list)


# In[17]:


#repo_topics
df["repo_topics"] = df["repo_topics"].apply(lambda x: literal_eval(x) if pd.notna(x) else [])
df["repo_topics"] = df["repo_topics"].apply(lambda x: ';'.join(map(str, x)) if x else pd.NA)
df["repo_topics"] = df["repo_topics"].str.lower()


# In[18]:


# df["repo_languages"] = df["repo_languages"].where(~(pd.isna(df["repo_languages"]) & ~pd.isna(df["repo_topics"]) & df["repo_topics"].isin(list(languages_set))), df["repo_topics"])


# In[ ]:


df


# In[20]:


def filter_string(s, valid_set):
    if pd.isna(s):
        return np.nan
    return ';'.join([x for x in s.split(';') if x in valid_set])

# Fill NA in language where topics is not NA with parts of topics in languages
mask = pd.isna(df['repo_languages']) & ~pd.isna(df['repo_topics'])
df.loc[mask, 'repo_languages'] = df.loc[mask, 'repo_topics'].apply(lambda x: filter_string(x, languages_set))


# In[ ]:


df["repo_languages"].isna().sum()/df["repo_languages"].shape[0]
# sorted(languages_set)


# NA in column "repo_languages" is 24,6%. We will try to fillna with mode() of languages\_list or unknown ??

# #### Handle missing/na values

# In[22]:


df["repo_languages"] = df["repo_languages"].fillna("unknown")


# #### Handle duplicating
# 

# In[23]:


# df.duplicated().sum()
df = df[~df.duplicated(subset=["repo_owner","repo_stars", "repo_forks"])]


# #### Transform

# In[ ]:


# df[df["repo_languages"].isna()]
df = df.drop(columns="repo_topics")
# df[(df["repo_languages"].isna()) & df["verified"] == True]
df


# In[25]:


df.to_csv(config["data"]["clean"]["file_data_cleaned"],index=False )

