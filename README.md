.venv/Scripts/jupyter nbconvert --to script ./notebooks/eda_machine_learning_2.ipynb
.venv/Scripts/ipython main.py 

import subprocess

subprocess.run(["python", "./notebooks/data_preprocessing_1.py"])
subprocess.run(["python", "./notebooks/eda_machine_learning_2.py"])


Exploding -> data leakage -> explode after split training/test data

Aggregate Post-Explosion: If you explode but want to avoid duplication, group by 'ID' and aggregate the exploded 'V' values (e.g., as a list or concatenated string) before training.

# VSCode Extensions security

## Introduction


- 🛠 GitHub Repository: [Final-Project](https://github.com/Ironhack-DA-Course/proj-final)
- 📋 Trello Board: [Project Tasks & Timeline](https://trello.com/b/73Tt6BOE/final) 
- Presentation: [Extension_Security](https://docs.google.com/presentation/d/../view) 

## Datasets Used

We use 1 file with raw data.
 *  **Stroke data (healthcare-dataset):** Demographics like age, gender, hypertension,... of clients.

A Metadata(dictionary) is provided to help us understand the content of the columns in each file and guide us through the analysis

- **_verified_:**                 check, whether extension's security is breached         (boolean)
- **_ext_categories_:**           categories of extension                                 (obj)   (multi values)
- **_ext\_install\_count_:**      total number of installations of extension              (int64)
- **_ext\_rating_:**              rating of extension (avg of stars rating)               (float64)
- **_repository_:**               url of repository                                       (obj)
- **_total\_vulners_:**           number of detected vulnerabilities                      (int64)
- **_critical\_vulners_:**        number of critical(severity) vulnerabilities            (int64)
- **_high\_vulners_:**            number of high(severity) vulnerabilities                (int64)
- **_medium\_vulners_:**          number of medium(severity) vulnerabilities              (int64)
- **_low\_vulners_:**             number of low(severity) vulnerabilities                 (int64)
- **_repo\_owner_:**              owner of repository (via column repository)             (obj)
- **_repo\_name_:**               name of repository (via column repository)              (obj)
- **_repo\_stars_:**              number of stars of repository (via column repository)   (int64)   
- **_repo\_forks_:**              number of forks of repository (via column repository)   (int64)   
- **repo\_languages:**            program languages used (via column repository)          (obj)   (multi values)


**Comments on the Data:**

##  Business Problem & Hypothesis
Predict the vulnerabilities in VS Code extensions based on the project's metrics to enhance security practices against cyber threats.
Developing strategic investments to mitigate risks, protect intellectual property, and maintain customer trust.

* **Question:** 

* **Conclusion:**  

**_Business recommendation_**: 

## Methodology

Our methodology involved several key steps, focusing on data scraping, data cleaning, data preprocessing, EDA, ML-Model selection, Model training , Model evaluation, and -tuning

**1. Data preprocessing:** 
* Datasets were downloaded from kaggle.
* Data Cleaning: 
    * maping categorical values to numerical, drop "id" column, not considering gender "Other"
    * fillna bmi with average value-> other approach ??
    * reduce some outliers on age , gender, bmi
   

**2. EDA**
* generic EDA on following columns:
  * Age: Older patients have a significantly higher risk.
  * Hypertension & Heart Disease: Strong positive correlation with stroke.
  * Glucose Level & BMI: Higher values may indicate risk but with some variability.
  * Smoking Status: Formerly smoked and smokes groups show increased risk.   
* power transformer on all numerical columns or glucose level for observing the distribution
* Check relationship with target column "stroke": heatmap for nummerical, chi test for categorical 

**3. Model selection:** 
* KNN 
* Logistic Regression
* Random Forest

**4. Model training:**  
* Trained train_dataset and predict test ones for every model
for classification

**5. Model evaluation:** 
Evaluate metrics for classification (target is category)
* Accuracy
* Recall
* Prediction
* F1
-> Focus on F1 due to imbalanced in target "verified"
-> Best performing model: 

**6. Model tuning:**
Grid search vs Random Search

**7. Insights:**
* Older adults with medical conditions should be prioritized.
* Lifestyle factors matter — smoking still plays a role.
* Medical metrics like glucose and BMI are useful but work best when combined with age or disease.
* Tree-based models performed best and are interpretable.

## Data Analysis Tools and Libraries:**
* __Python__: The primary programming language for data manipulation and analysis.
* __Pandas__: Essential for data loading, cleaning, and transformation.
* __Matplotlib / Seaborn__: Used for creating various visualizations (bar charts, line graphs).
* __Graphviz /Optuna__: Visualizations ML results
* __Scikit-learn__: Machine learning training/tuning

##  Repository Structure

```
proj-final/
├── data/                        # Raw and cleaned CSV files
├── figures/                     # Sketching of structures in EDA, and ML
├── notebooks/                   # Python notebooks with analysis
   |__ lib/                      # Python functions
├── README.md                    # This file
└── slides                       # Url of presentation
```
## 👥 Team Members
__*Robert*__