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

def get_github_user_type(username: str, token: str = None) -> Dict:
    """
    Get GitHub user/org information to determine if it's individual or organization
    
    Args:
        username: GitHub username or organization name
        token: GitHub personal access token (optional but recommended)
    
    Returns:
        Dictionary with user info including type
    """
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'
    
    url = f'https://api.github.com/users/{username}'
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        return {
            'username': username,
            'type': data.get('type', 'Unknown'),  # 'User' or 'Organization'
            'name': data.get('name'),
            'company': data.get('company'),
            'public_repos': data.get('public_repos'),
            'followers': data.get('followers'),
            'created_at': data.get('created_at'),
            'is_organization': data.get('type') == 'Organization'
        }
    except requests.exceptions.RequestException as e:
        return {
            'username': username,
            'type': 'Error',
            'error': str(e),
            'is_organization': None
        }

    