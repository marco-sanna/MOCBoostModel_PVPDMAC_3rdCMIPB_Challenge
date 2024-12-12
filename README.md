# MOCBoostModel_PVPDMAC_3rdCMIPB_Challenge

This repository contains 2 Python scripts developed for predictive modeling tasks in the 3rd CMI-PB Challenge together with the input datasets.

- MOCBoost.py was used to predict the main chalenge tasks. 
- MOCBoost_bonus.py was used to implement bonus task prediction. 

The scripts contain:
1. Preprocessing steps to obtain input Training and Test data
2. Modelling steps to train a MultiOutputRegressor with CatBoost as internal estimator
3. Prediction steps on validation data
    

## Repository contents:

- data (dir): 
    - bonus_task_data
    - harmonized
    - legacy
    - raw
    - subject_metadata (contains subjects .tsv files extracted from raw (dir))
    
- Scripts (dir):
    - MOCBoost.py 
    - MOCBoost_bonus.py
    
- requirements.txt: list of required dependancies

## Requirements:

- Python 3.12.7

This project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- requests
- scipy
- future
- scikit-learn
- xgboost
- catboost
- pyHSICLasso
- hyperopt

These can be installed using `pip` or by creating a virtual environment (recommended).

## Usage:

1. Clone or download the repository
2. Access the directory
3. Create and activate a virtual environment (suggested)
4. Install the dependencies: pip install -r requirements.txt
5. Run the script on terminal: python ./Scripts/MOCBoost.py

