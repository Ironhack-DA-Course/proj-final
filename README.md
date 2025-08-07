.venv/Scripts/jupyter nbconvert --to script ./notebooks/eda_machine_learning_2.ipynb
.venv/Scripts/ipython main.py 

import subprocess

subprocess.run(["python", "./notebooks/data_preprocessing_1.py"])
subprocess.run(["python", "./notebooks/eda_machine_learning_2.py"])