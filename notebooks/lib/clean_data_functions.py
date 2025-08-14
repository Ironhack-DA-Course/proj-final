import pandas as pd
import numpy as np
import re
import requests
import time
from typing import List, Dict

def clean_ext_publisher(x):
    if isinstance(x, str):
        text = str(x).strip().lower()
        if text.startswith("ms"):
            return "microsoft"
        else:
            return x.lower()
    return x

def clean_repo_publisher(x):
    if isinstance(x, str):
        return str(x).strip().lower().replace(" ", "")
    return x

def clean_ext_version(x):
    if isinstance(x, str):
        text = str(x).strip().lower()
        if "." in text:
            return text.split(".")[0]
    return x

def filter_string(s, valid_set):
    if pd.isna(s):
        return np.nan
    return ';'.join([x for x in s.split(';') if x in valid_set])