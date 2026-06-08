# Generalisability of machine learning models for rock type classification on XRF drill core scan data at the Rävliden North VMS deposit, Skellefte district, Sweden

This repository contains code and dummy data for the paper titled 'Generalisability of machine learning models for rock type classification on XRF drill core scan data at the Rävliden North VMS deposit, Skellefte district, Sweden'. 

## ML_classifiers.py

This file contains Python base code for running training and testing of RF, SVM and MLP models.
Pre-processing of data is done separately and decided by the user.

## requirements.txt

This file lists the require libraries and versions for running the Python script.

## Note

The comments are available in the script to follow the process.
Data are dummy data that have been pre-processed. Raw data is proprietary to Boliden and not made available.

## Tutorial and user guide

### 1. Overview
This Python script contains the workflow for training a multi-layer perceptron (MLP) classifier to predict geological targets such as precursors, alteration types or rock types. The process includes hyperparameter tuning, final model training, evaluation on test sets, uncertainty estimation using Monte Carlo Dropout, and feature importance analysis with SHAP.

Code Structure

•	Imports: Load libraries and check versions.

•	Global Setup: Set paths, seeds, and constants.

•	Data Loading: Load training/test datasets.

•	Set the training strategy (1-5) to choose different pre-processing and cross-validation setups.

•	Helper functions: Set up for Python scode to export metrics in desirable format.

•	Class ConditionalSMOTE: Defined as a guardrail for rare classes, particularly important for grouped cross validation per drill hole.

•	Training of RF, SVM and MLP models. Note importance that SMOTE is applied within a pipeline so SMOTE is only applied to training folds and not to validation folds.

•	Model evaluation: Runs on test set for RF, SVM and MLP. This is done for each drill hole separately.

### 2. Input Data

Input data Format
•	CSV files with columns for features and targets.

•	Features should end with the suffix ‘_pct_lrEM_clr_z’, ‘_pct_lrEM’ or ‘_pct_half_DL_clr_z’ (or update FEATURE_SUFFIX in the script). These are the different pre-processing variants.

•	The target columns should be named as ‘Final_v5_labels_Rock_type_FS’. (or update TARGET in script accordingly).

•	Have a ‘Hole_ID’ column to ensure that stratified group K-fold cross-validation works correctly.


### 3. Configuration
Script Constants

Edit these at the top of the script:

•	model_strategy: Set value 1-5 to choose which pre-processing and generalisability setup to use.

•	df_train: Training dataset filename.

•	df_test_DH10: Test dataset 1 filename.

•	df_test_DH14: Test dataset 2 filename.

•	df_test_DH18: Test dataset 3 filename.

•	df_test_total: Test dataset with all drill holes included.

Paths

By default, all paths are relative to the script’s location. Adjust if running from a different directory.

### 4. Running the Script
Step-by-Step
1.	Open a terminal in your project directory.
2.	Activate a virtual environment if using one.
3.	Using a Jupyter extension is recommended.
4.	Run the script: python script.py
5.	Monitor progress: The script prints real-time updates (e.g., fold assignments, model training, evaluation).
6.	Save locations are printed upon completion.

Expected Output

•	Results are saved with prefix depending on specified training strategy (Mod1_, Mod2_, etc.)

•	Cross-validation results are saved including best hyperparamteres from tuning for all three models (RF, SVM and MLP).

•	Test results for all three drills hole individually and the entire test set (all three test drill holes together) are saved.

•	Confusion matrices as figures and tables saved for each model.



## Citing this work

Please cite this work.
```bibtex
    @article {},
	author = {Filip Simán, Nils Jansson, Foteini Simistira Liwicki, Hamam Mokayed, Christian Günther, Paul McDonnell, Tobias Hermansson},
	title = {Generalisability of machine learning models for rock type classification on XRF drill core scan data at the Rävliden North VMS deposit, Skellefte district, Sweden},
	elocation-id = {},
	year = {},
	doi = {},
	publisher = {},
	URL = {},
	eprint = {},
	journal = {}
    }

```

