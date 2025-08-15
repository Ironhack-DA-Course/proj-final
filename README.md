# VSCode Extensions security

## Introduction
VSCode holds a leading position in the IDE market, commanding approximately 40% of the market share. Its lightweight design, open-source, and versatile features make it a top choice for software developers.
The growing number of new extensions for VSCode reflects a vibrant and active community, but it also introduces potential cybersecurity risks. In this project, we will analyze VSCode extensions available on the marketplace and predict the risks associated with their metrics.

- 🛠 GitHub Repository: [Final-Project](https://github.com/Ironhack-DA-Course/proj-final)
- 📋 Trello Board: [Project Tasks & Timeline](https://trello.com/b/73Tt6BOE/final) 
- 🎬 Presentation : [VSCode Extension Vulnerabilities](https://docs.google.com/presentation/d/1K6rmBP8EbodVEAzXBknHKRE_Xcf9jUkD/view) 

## Datasets

We use 1 file with raw data.
-  **extensions\_repos\_full:**   Engagement, Activity like categories, install_count, repository,... of extensions.

The other 3 files are raw data for exercise by scraping more derived datas
- **VSCode Extensions\_Verified**: Validation, whether the extension is secured (scanning through Snyk) 
- **VSCode Extensions\_Vulnerabilities**: Count of vulnerabilities with different weight
- **vscode\_extensions\_scraped_0825**: total available extensions in VS-Code Market place until 08.2025

A Metadata(dictionary) is provided to help us understand the content of the columns in each file and guide us through the analysis

|                                |                                                        |         |
|--------------------------------|--------------------------------------------------------|---------|
| **_verified_:**                | check, whether extension's security is breached        | (boolean) |
| **_ext_categories_:**          | categories of extension                                | (obj)   (multi values) |
| **_ext\_install\_count_:**     | total number of installations of extension             | (int64) |
| **_ext\_rating_:**             | rating of extension (avg of stars rating)              | (float64) |
| **_repository_:**              | url of repository                                      | (obj)   | 
| **_total\_vulners_:**          | number of detected vulnerabilities                     | (int64) |
| **_critical\_vulners_:**       | number of critical(severity) vulnerabilities           | (int64) |
| **_high\_vulners_:**           | number of high(severity) vulnerabilities               | (int64) |
| **_medium\_vulners_:**         | number of medium(severity) vulnerabilities             | (int64) |
| **_low\_vulners_:**            | number of low(severity) vulnerabilities                | (int64) |
| **_repo\_owner_:**             | owner of repository (via column repository)            | (obj)   |
| **_repo\_name_:**              | name of repository (via column repository)             | (obj)   |
| **_repo\_stars_:**             | number of stars of repository (via column repository)  | (int64) |   
| **_repo\_forks_:**             | number of forks of repository (via column repository)  | (int64) |   
| **repo\_languages:**           | program languages used (via column repository)         | (obj)   (multi values) |


**Comments on the Data:**
- Multi values as value
- Duplicated
- Missing values
- Inconsistent data

##  Business Problem 
Predict the vulnerabilities in VS Code extensions based on the project's metrics to enhance security practices against cyber threats.
Developing strategic investments to mitigate risks, protect intellectual property, and maintain customer trust.


## Hypothesis
* Extension with **high installation count** has _high_ proability of being breached 
* Extension with **high rating and forks of repository** also attracts the attackers
* Extension with **high rating** attracts the exploiation

## **Conclusion:**  
* Extension with **high installation count** has _high_ proability of being breached : checked
* Extension with **high rating and forks of repository** also attracts the attackers : checked
* Extension with **high rating** attracts the exploiation* : unchecked

**_Business recommendation_**: 
- Create an online service/software to validate the proability of vulnerabilities based on the extension's metrics and project's status of repos.
- Integrate prediction into the vulnerability scanning (e.g., Snyk, Veracode, SonarQube) tools
- Publishing report service to market niche, (e.g. extensions for IDE, Browser, Software, ...)

## Methodology

Our methodology involved several key steps, focusing on data collection, data wrangling, EDA, data preprocessing,  ML-Model selection, Model training , Model evaluation, and -tuning

**1. Data Collection:**
- Scraping the site https://marketplace.visualstudio.com/vscode to collect infos of of VSCode extensions
- Getting the report https://docs.google.com/spreadsheets/d/12GIzrSzzU-_Ok4pPigUJYSxKO2ZYSmDwr1OJy6T2X40/edit for status of VSCode extensions vulnarabilities

**2. Data Wrangling:**

* **Data Exploration:** 
  - Understanding the data’s structure, quality, and potential issues by exploring it through summary statistics, visualizations, and basic queries.
    -> Metadata

* **Data Cleaning:**
  - Removing or correcting errors, inconsistencies, and inaccuracies in the data. 
  This includes handling missing values, outliers, duplicates, and data entry errors.

* **Data Transformation:**
  - Structuring: Changing the format or structure of the data, such as pivoting tables, splitting or merging columns, and normalizing data. Enhancing the dataset with additional information, such as adding derived features, merging external datasets, or enriching with domain-specific knowledge.
  - Combining: This step also includes data integration, where data from multiple sources are combined.
  - 
  - Ensuring the transformed data meets the required quality and accuracy standards before using it in further analysis or modelling.
  - Remove non-relevant columns for model training ext\_id, ...

**3. EDA**  
* **Univariate:**
  * Overview: 
    - ext_categories: over 20 categories, the most common are not categorized ones, Programming Languages
    - ext_rating:  4.17 median
    - ext\_installation\_count: 2147 median
    - repo_rating: 3* median  
    - repo_languages: most used programming languages for extensions are Typescript, Javascript
    - repo_fork: 1 median
    - ...
  * Distribution: 
    - Most of them are not normal distribution -> smoothing/transforming them later for model training    
    - **_critical_vulners has only 1 value '0'_** 
  * Outlier:
    - Most of them have high positive high skew -> right tails with extrem high values     

* **Bivariate:**
  * Correlation/Association:
    - Most of them are correlated/ associated with each others but weak or medium relationship
    - **_'total\_vulners' strong correlation with high\_, medium\_, low\_vulners_**
    - repo\_stars strong relationship with repo\_forks

**4. Machine Learning:**

* **Data Preprocessing:**
  - Data Cleaning and Transformation: 
    * Binning 'ext\_rating', 'ext\_version' columns with high cardinality
  - Outlier Handling: Replace outliers with bounders
  - Split data into Train, Test set on features and target

* **Feature Engineering:**
  - One hot encoding: nominal categorical 
  - Binning/Grouping: ordinal categorical
  - power-transform on all numerical columns into  normal distribution    
  - Recombine Train, Test set after applying feature engineering

  - Feature selection (Dimensionality Reduction): drop feature columns ('total_vulners') with high correlation to other features

* **Imbalanced handling:**
  - Use SMOTE to transform into balanced dataset

* **Model selection:** 
  - Ensemble:
    * Random Forest
    * Gradient Boosting
    * Ada Boost
    * Bagging Classifier    

  - Trees:
    * Extra Trees
    * Decision Trees
    * XGBoost
    * LightGBM
    * CatBoost

  - Kernel-Based:
    * SVC

  - Probabilistic
    * BernoulliNB
    * GaussianNB

  - Linear
    * Logistic Regression

  - Instance-Based
    * KNN

* **Define Metrics:**
  |Metric|Definition|	Meaning in Attrition Context|
  | ----------- | ----------- | ----------- |
  | Recall|  TP/(TP+FN) | 	Most important – how many true leavers you can catch|
  | Precision	 |TP/(TP+FP) |	Among predicted leavers, how many are actually correct
  | Accuracy |(TP+TN)/Total |	Can be misleading with imbalanced data (e.g., <20% attrition) |
  | F1-score |2⋅Precision⋅Recall/(Precision+Recall) |Balanced trade-off between Precision and Recall|
  | AUC-ROC	|Area under ROC Curve |	Measures ability to distinguish leavers vs. stayers at all thresholds|

* **Model training:**  
  - Train train_dataset and predict test ones for each model
  for classification

* **Model evaluation:** 
Evaluate metrics for classification (target is category)
  - Accuracy
  - Recall
  - Prediction
  - F1
-> Focus on "Recall" due to imbalanced in target "verified"
-> Best performing model overall: _BaggingClassifier(with default DecisionTreeClassifier)_

**6. Model tuning:**
  * Tuning with GridSearchCV and RanddomSearchCV

**7. Insight:**
  * Model 'SVC' takes huge computation and is slow
  * Balance treatment improves performance of model with weak performances in imbalanced
  * DecisionTree/ Bagging are among the best -> decide for Bagging due to best overall in many score-categories in both balanced/imblanced dataset(Acc, Precision, Recall, F1)

**8. Limitation:**
  * Exploding -> data leakage -> explode after split training/test data

## Data Analysis Tools and Libraries:

* __Python__: The primary programming language for data manipulation and analysis.
* __Pandas__: Essential for data loading, cleaning, and transformation.
* __Matplotlib / Seaborn__: Used for creating various visualizations (bar charts, line graphs).
* __Optuna__: Visualizations ML results
* __Scikit-learn__: Machine learning training/tuning
* __Pickle__: Export/Import model files
* __imblearn__: ML lib for imbalanced 
* __statmodel__: statistical  lib


## Notebooks Usage:
* There are maybe 2 model files. One is with imbalanced and one is with balanced/SMOTE dataset. Check id of {X\_train}, {y\_train}
* Execute notebooks in orders *\_1.ipynb, *\_2.ipynb, *\_3.ipynb

##  Repository Structure
```
proj-final/
├── data/                        # Raw and cleaned CSV files
├── figures/                     # Sketching of structures in EDA, and ML
├── models/                      # Pickle files of preprocessing, training
├── notebooks/                   # Python notebooks with analysis
    |── lib/                     # Python functions
├── README.md                    # This file
└── slides                       # Url of presentation
```
## __Author__
__*Robert*__