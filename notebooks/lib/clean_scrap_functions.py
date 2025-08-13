import pandas as pd
import numpy as np
import re
import requests
import time
from typing import List, Dict

def split_url(x):
    '''
    GIT Expression 
    - '(\w+://)(.+@)*([\w\d\.]+)(:[\d]+){0,1}/*(.*)'
    - /^((?<protocol>https?|ssh|git|ftps?):\/\/)?((?<user>[^\/@]+)@)?(?<hostname>[^\/:]+)[\/:](?<pathHead>[^\/:]+)\/(?<pathTail>(.+)(.git)?\/?)$/gm


    https://stackoverflow.com/questions/2514859/regular-expression-for-git-repository
    '''
    if isinstance(x, str):  # Check if the input is a string
        git_url = str(x).strip()
        url = git_url[:-4] if git_url.endswith(".git") else git_url
        
        parts = re.split(r'//|@', url)
        if len(parts) > 1:
            second_part = parts[1].replace(":", "/") if ":" in parts[1] else parts[1]
            return f"https://{second_part}"
        else:
            return x
    return x  #
    
   

def clean_ext_publisher(x):
    text = str(x).strip().lower()
    if text.startswith("ms"):
        return "microsoft"
    else:
        return x.lower()

def clean_repo_publisher(x):
    return str(x).strip().lower().replace(" ", "")
    