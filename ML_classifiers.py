# Python script for "Generalisability of machine learning models for rock type classification on XRF drill core scan data at the Rävliden North VMS deposit, Skellefte district, Sweden"
# Journal: Natural Resources Research
# Filip Simán1*, Nils Jansson1, Foteini Simistira Liwicki3, Hamam Mokayed3, Christian Günther3, Paul McDonnell2, Tobias Hermansson2
# *Corresponding author: filip.siman@ltu.se
# 1Swedish School of Mines, Luleå University of Technology, Department of Civil and Environmental Engineering, Luleå, Sweden
# 2Boliden Mineral AB, Exploration Department, Boliden, Sweden
# 3Luleå University of Technology, Department of Computer Science, Electrical and Space Engineering, Luleå, Sweden

#---
# ML models include Random Forest (RF), Support Vector Machine (SVM) and Multilayer Perceptron (MLP).
# Data is pre-processed in previous scripts
# Different model strategies to test here:
# 1. Baseline models with shuffle split CV with lrEM VBDL imputation, CLR-z normalization and SMOTE.
# 2. Baseline variants of models with shuffle split CV:
#    a) No SMOTE
#    b) No CLR-z normalization 
#    c) Half-detenction limit replacement instead of lrEM VBDL imputation
# 3. Comparison models with stratified k-groups CV with lrEM VBDL imputation, CLR-z normalization and SMOTE.

#---
# Author: Filip Siman
# Lulea university of technology
# Contact: filip.siman@ltu.se
# 2026-01-29

# %% Load some useful libraries
print("Loading some useful libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, KFold
from scipy.stats import randint
from scipy.stats import loguniform
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_predict, cross_validate
import time
import joblib


# %% Set working directory to script location (even in interactive mode)
import os
from pathlib import Path

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path().resolve()

os.chdir(script_dir)
print("Working directory set to:", os.getcwd())

#%% Load data
# The training data.
df_train = pd.read_csv("3_preprocessed_training_data_v5.csv")

# Load the test data as well, for later evaluation.
df_test_DH10 = pd.read_csv("3_preprocessed_DH10_test_data_v5.csv")
df_test_DH14 = pd.read_csv("3_preprocessed_DH14_test_data_v5.csv")
df_test_DH18 = pd.read_csv("3_preprocessed_DH18_test_data_v5.csv")
df_test_total = pd.read_csv("3_preprocessed_total_test_data_v5.csv")

# %% Select model strategy parameters.

# 1. Baseline models, shuffle split CV with SMOTE, CLR-z normalization and lrEM VBDL imputation.
# 2. Basline models no SMOTE
# 3. Baseline models no CLR-z normalization
# 4. Baseline models half-detection limit imputation
# 5. Stratified k-groups CV models

# Set model strategy here:
model_strategy = 1

# %% Prepare feature matrix X and target vector y
# Check model_strategy and prepare data accordingly.
if model_strategy == 1:
    feature_cols = [
        'Si_pct_lrEM_clr_z', 'S_pct_lrEM_clr_z', 'K_pct_lrEM_clr_z', 
        'Ca_pct_lrEM_clr_z', 'Ti_pct_lrEM_clr_z', 'Mn_pct_lrEM_clr_z', 
        'Fe_pct_lrEM_clr_z', 'Cu_pct_lrEM_clr_z', 'Zn_pct_lrEM_clr_z', 
        'Sr_pct_lrEM_clr_z', 'Zr_pct_lrEM_clr_z'
    ]

    RESULTS_PREFIX = "Mod1_"
    do_smote = True
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    groups = None
    
elif model_strategy == 2:
    feature_cols = [
        'Si_pct_lrEM_clr_z', 'S_pct_lrEM_clr_z', 'K_pct_lrEM_clr_z', 
        'Ca_pct_lrEM_clr_z', 'Ti_pct_lrEM_clr_z', 'Mn_pct_lrEM_clr_z', 
        'Fe_pct_lrEM_clr_z', 'Cu_pct_lrEM_clr_z', 'Zn_pct_lrEM_clr_z', 
        'Sr_pct_lrEM_clr_z', 'Zr_pct_lrEM_clr_z'
    ]
    
    RESULTS_PREFIX = "Mod1a_"
    do_smote = False
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    groups = None

elif model_strategy == 3:
    feature_cols = [
        'Si_pct_lrEM', 'S_pct_lrEM', 'K_pct_lrEM', 
        'Ca_pct_lrEM', 'Ti_pct_lrEM', 'Mn_pct_lrEM',
        'Fe_pct_lrEM', 'Cu_pct_lrEM', 'Zn_pct_lrEM',
        'Sr_pct_lrEM', 'Zr_pct_lrEM'
    ]

    RESULTS_PREFIX = "Mod1b_"
    do_smote = True
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    groups = None

elif model_strategy == 4:
    feature_cols = [
        'Si_pct_half_DL_clr_z', 'S_pct_half_DL_clr_z', 'K_pct_half_DL_clr_z',
        'Ca_pct_half_DL_clr_z', 'Ti_pct_half_DL_clr_z', 'Mn_pct_half_DL_clr_z',
        'Fe_pct_half_DL_clr_z', 'Cu_pct_half_DL_clr_z', 'Zn_pct_half_DL_clr_z',
        'Sr_pct_half_DL_clr_z', 'Zr_pct_half_DL_clr_z'
    ]

    RESULTS_PREFIX = "Mod1c_"
    do_smote = True
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=42)
    groups = None

elif model_strategy == 5:
    feature_cols = [
        'Si_pct_lrEM_clr_z', 'S_pct_lrEM_clr_z', 'K_pct_lrEM_clr_z', 
        'Ca_pct_lrEM_clr_z', 'Ti_pct_lrEM_clr_z', 'Mn_pct_lrEM_clr_z', 
        'Fe_pct_lrEM_clr_z', 'Cu_pct_lrEM_clr_z', 'Zn_pct_lrEM_clr_z', 
        'Sr_pct_lrEM_clr_z', 'Zr_pct_lrEM_clr_z'
    ]

    RESULTS_PREFIX = "Mod2_"
    do_smote = True
    cv_strategy = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
    groups = df_train['Hole_ID']

# Select only those columns for features
x = df_train[feature_cols]
y = df_train['Final_v5_labels_Rock_type_FS']

# Ensure results folder exists when script is run; if it already exists, this is a no-op.
try:
    results_dir = script_dir / f"{RESULTS_PREFIX}results"
except NameError:
    results_dir = Path.cwd() / f"{RESULTS_PREFIX}results"
results_dir.mkdir(parents=True, exist_ok=True)
print("Results directory ensured at:", results_dir)

#%% Define some helper functions
# ---------------------------------------------------------------------------
# Helper functions: export cross-validated evaluation metrics to CSV
# ---------------------------------------------------------------------------
def export_cv_metrics(estimator, x, y, cv, prefix, groups=None):
  # Run cross_val_predict on estimator and export classification report,
  # confusion matrix and some overall metrics to CSV files with the given prefix.
  # estimator: estimator or pipeline (must implement fit/predict)
  # x, y: data
  # cv: int or cross-validation splitter
  # prefix: filename prefix (string)
  # groups: group labels for GroupKFold or StratifiedGroupKFold (optional)

    print(f"Exporting CV metrics for: {prefix}")
    # Create folder for cross-val results if not existing.
    # Name accroding to RESULTS_PREFIX: "ModX_results" where X = RESULTS_PREFIX
    # If folder exists named "ModX_results" where X = RESULTS_PREFIX, use that.
    try:
        out_dir = Path(script_dir) / f"{RESULTS_PREFIX}results"
    except NameError:
        out_dir = Path.cwd() / f"{RESULTS_PREFIX}results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Prepare a single, consistent full_prefix for all outputs in this function
    full_prefix = f"{RESULTS_PREFIX}{prefix}" if not str(prefix).startswith(RESULTS_PREFIX) else str(prefix)

    # Get cross-validated predictions
    y_pred = cross_val_predict(estimator, x, y, cv=cv, groups=groups)

    # Classification report -> DataFrame
    rpt = classification_report(y, y_pred, output_dict=True)
    df_rpt = pd.DataFrame(rpt).transpose()
    df_rpt.to_csv(str(out_dir / f"{full_prefix}_classification_report.csv"), index=True)

    # Confusion matrix -> DataFrame (rows=true, cols=pred)
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(str(out_dir / f"{full_prefix}_confusion_matrix.csv"))
    # Also save a plotted confusion matrix figure (eps + png)
    try:
        save_confusion_matrix_figure(df_cm, out_dir / f"{full_prefix}_confusion_matrix") # Our own funtion defined below.
    except NameError:
        # Function not yet defined or import issue; skip plotting
        pass

    # Overall scalar metrics
    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average='macro')
    # Also compute cross-validated mean and std for the final estimator
    cv_stats = {}
    try:
        # cross_validate returns dict with keys 'test_<metric>'
        scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'cohen_kappa': make_scorer(cohen_kappa_score)}
        cv_res = cross_validate(estimator, x, y, cv=cv, scoring=scoring, groups=groups, error_score='raise')
        # compute mean/std
        for metric in scoring.keys():
            key = f"test_{metric}"
            vals = cv_res.get(key)
            if vals is not None:
                cv_stats[f"{metric}_cv_mean"] = float(np.mean(vals))
                cv_stats[f"{metric}_cv_std"] = float(np.std(vals, ddof=0))
            else:
                cv_stats[f"{metric}_cv_mean"] = None
                cv_stats[f"{metric}_cv_std"] = None
    except Exception as e:
        print(f"Warning: cross_validate failed for final estimator: {e}")
        cv_stats = {
            "accuracy_cv_mean": None,
            "accuracy_cv_std": None,
            "f1_macro_cv_mean": None,
            "f1_macro_cv_std": None,
            "cohen_kappa_cv_mean": None,
            "cohen_kappa_cv_std": None,
        }

    # Compute single-run Cohen's kappa (safe)
    try:
        cv_kappa_single = cohen_kappa_score(y, y_pred)
    except Exception:
        cv_kappa_single = None

    overall = pd.DataFrame({
        'accuracy': [acc],
        'f1_macro': [f1m],
        'cohen_kappa': [cv_kappa_single],
        **{k: [v] for k, v in cv_stats.items()}
    })
    overall.to_csv(str(out_dir / f"{full_prefix}_overall_metrics.csv"), index=False)

    print(f"Saved: {out_dir / f'{full_prefix}_classification_report.csv'}, {out_dir / f'{full_prefix}_confusion_matrix.csv'}, {out_dir / f'{full_prefix}_overall_metrics.csv'}")


def run_on_test_set_and_export(estimator, x_test, y_test, prefix):
    # Run the given estimator on test set and export the results as CSV files.

    print(f"Exporting test set metrics for: {prefix}")
    # Create folder for cross-val results if not existing.
    # Name accroding to RESULTS_PREFIX: "ModX_results" where X = RESULTS_PREFIX
    # If folder exists named "ModX_results" where X = RESULTS_PREFIX, use that
    try:
        out_dir = Path(script_dir) / f"{RESULTS_PREFIX}results"
    except NameError:
        out_dir = Path.cwd() / f"{RESULTS_PREFIX}results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predict on test set (timed) with error handling
    try:
        start_pred = time.time()
        y_pred = estimator.predict(x_test)
        pred_time = time.time() - start_pred
        print(f"{prefix} test prediction time: {pred_time:.2f}s")
    except Exception as e:
        print(f"Prediction failed for {prefix}: {e}")
        return

    # Important! Also save raw test data with predictions appended
    try:
        # Prepare a DataFrame from x_test (works if x_test is DataFrame or ndarray)
        if isinstance(x_test, pd.DataFrame):
            df_out = x_test.copy()
        else:
            # Try to infer column names from feature_cols if present, else use integer columns
            try:
                cols = feature_cols
            except NameError:
                cols = None
            df_out = pd.DataFrame(x_test, columns=cols)

        # True label column name
        true_col_name = y_test.name if hasattr(y_test, 'name') and y_test.name else 'true_rock_type'
        df_out[true_col_name] = list(y_test)

        # Determine suffix based on prefix (common cases)
        pl = prefix.lower() if isinstance(prefix, str) else ''
        if 'rf' in pl:
            suffix = '_RF'
        elif 'svm' in pl:
            suffix = '_SVM'
        elif 'mlp' in pl:
            suffix = '_MLP'
        else:
            # fallback: uppercase prefix without non-alnum
            safe = ''.join(ch for ch in str(prefix) if ch.isalnum())
            suffix = f"_{safe.upper()}"

        pred_col = f"predicted_rock_type{suffix}"
        df_out[pred_col] = list(y_pred)

        # Save combined CSV (build filename locally to avoid relying on outer full_prefix binding)
        try:
            if isinstance(prefix, str) and prefix.startswith(RESULTS_PREFIX):
                fname = f"{prefix}_testset_with_predictions.csv"
            else:
                fname = f"{RESULTS_PREFIX}{prefix}_testset_with_predictions.csv"
            out_file = out_dir / fname
            df_out.to_csv(str(out_file), index=False)
            print(f"Saved combined test predictions to: {out_file}")
        except Exception as e:
            print(f"Failed to save combined predictions CSV: {e}")
    except Exception as e:
        print(f"Failed to build/save predictions dataframe for {prefix}: {e}")

    # Classification report -> DataFrame
    rpt = classification_report(y_test, y_pred, output_dict=True)
    df_rpt = pd.DataFrame(rpt).transpose()
    full_prefix = f"{RESULTS_PREFIX}{prefix}" if not prefix.startswith(RESULTS_PREFIX) else prefix
    df_rpt.to_csv(str(out_dir / f"{full_prefix}_testset_classification_report.csv"), index=True)

    # Confusion matrix -> DataFrame (rows=true, cols=pred)
    labels = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    df_cm.to_csv(str(out_dir / f"{full_prefix}_testset_confusion_matrix.csv"))
    # Also save a plotted confusion matrix figure (eps + png)
    try:
        save_confusion_matrix_figure(df_cm, out_dir / f"{full_prefix}_testset_confusion_matrix") # Our own funtion defined below.
    except NameError:
        # Function not yet defined or import issue; skip plotting
        pass

    # Overall scalar metrics (include Cohen's kappa)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    try:
        kappa = cohen_kappa_score(y_test, y_pred)
    except Exception:
        kappa = None

    overall = pd.DataFrame({
        'accuracy': [acc],
        'f1_macro': [f1m],
        'cohen_kappa': [kappa]
    })
    overall.to_csv(str(out_dir / f"{full_prefix}_testset_overall_metrics.csv"), index=False)

    print(f"Saved: {out_dir / f'{full_prefix}_testset_classification_report.csv'}, {out_dir / f'{full_prefix}_testset_confusion_matrix.csv'}, {out_dir / f'{full_prefix}_testset_overall_metrics.csv'}")


def export_search_cv_results(search_obj, prefix):
    # Export RandomizedSearchCV / GridSearchCV cv_results_ dict to a CSV for analysis.
    # If no cv_results_ attribute exists the function will return silently.

    # Create folder for hyperparameter search cv results if not existing.
    # Name accroding to RESULTS_PREFIX: "ModX_results" where X = RESULTS_PREFIX
    # If folder exists named "ModX_results" where X = RESULTS_PREFIX, use that

    if not hasattr(search_obj, 'cv_results_'):
        print(f"No cv_results_ available on object {search_obj}; skipping export.")
        return

    cvres = search_obj.cv_results_
    # Determine output folder: prefer script location if available, else cwd
    try:
        out_dir = Path(script_dir) / f"{RESULTS_PREFIX}results"
    except NameError:
        out_dir = Path.cwd() / f"{RESULTS_PREFIX}results"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Convert to DataFrame where possible
    try:
        df_cvres = pd.DataFrame(cvres)
        full_prefix = f"{RESULTS_PREFIX}{prefix}" if not prefix.startswith(RESULTS_PREFIX) else prefix
        df_cvres.to_csv(str(out_dir / f"{full_prefix}_hyparam_cv_results.csv"), index=False)
        print(f"Saved: {out_dir / f'{full_prefix}_hyparam_cv_results.csv'}")
    except Exception as e:
        print(f"Failed to export cv_results_: {e}")


def save_confusion_matrix_figure(df_cm, out_path_base, fmt_list=("eps", "png"), cmap="viridis"):
    # Create and save a nice confusion-matrix figure from a DataFrame.
    # df_cm: pandas.DataFrame with index=actual, columns=predicted (values are ints)
    # out_path_base: path-like or str without extension where to save the figure; extensions will be appended
    # fmt_list: tuple of formats to save (e.g. ('eps','png'))
    # cmap: colormap name for seaborn

    try:
        # Create a figure scaled to the number of classes
        n = max(3, len(df_cm))
        fig, ax = plt.subplots(figsize=(max(6, n * 0.6), max(5, n * 0.6)))

        sns.heatmap(
            df_cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            square=True,
            cbar_kws={"shrink": 0.6},
            ax=ax,
        )

        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

        ax.set_xlabel("Predicted", fontsize=14, labelpad=20)
        ax.set_ylabel("Actual", fontsize=14)

        fig.tight_layout()

        # Save requested formats
        out_base = Path(out_path_base)
        for fmt in fmt_list:
            out_file = out_base.with_suffix(f".{fmt}")
            fig.savefig(str(out_file), format=fmt, dpi=300, bbox_inches="tight")

        plt.close(fig)
        print(f"Saved confusion matrix figures: {[str(out_base.with_suffix('.' + f)) for f in fmt_list]}")
    except Exception as e:
        print(f"Failed to save confusion matrix figure for {out_path_base}: {e}")

# Build a conditional SMOTE helper function
# This is needed because with StratifiedGroupKFold some classes may have very few or no samples in some folds.
class ConditionalSMOTE:
    # Apply SMOTE only to classes that have enough samples in the current fold.
    # This exposes a fit_resample(X, y) method so it can be used inside an
    # imblearn.pipeline. For classes with too few examples (<=1) the sampler
    # leaves them unchanged.

    def __init__(self, k_neighbors=5, random_state=None, sampling_strategy='not majority'):
        self.k_neighbors = int(k_neighbors)
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X, y):
        y_arr = np.asarray(y)
        counts = pd.Series(y_arr).value_counts()
        if counts.empty:
            return X, y

        majority_count = int(counts.max())

        # Build sampling_strategy only for classes that have >=2 samples
        sampling_strategy = {}
        for cls, cnt in counts.items():
            if (int(cnt) >= 2) and (int(cnt) < majority_count):
                # target to bring minority up to majority (can be adjusted)
                sampling_strategy[cls] = majority_count

        if len(sampling_strategy) == 0:
            # Nothing to do — return original arrays
            return X, y

        # Ensure safe k_neighbors for SMOTE (needs at least k_neighbors+1 samples)
        min_cnt = min([int(counts[c]) for c in sampling_strategy.keys()])
        safe_k = max(1, min(self.k_neighbors, min_cnt - 1))

        sm = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=safe_k, random_state=self.random_state)
        X_res, y_res = sm.fit_resample(X, y_arr)
        return X_res, y_res


# ---------------------------------------------------------------------------

#%% Training Random Forest (RF).
# Check first if do_smote is set to True or False
if do_smote:
    print("Conditional SMOTE applied in pipeline.")
    # Pipeline: SMOTE -> RandomForest
    # (SMOTE applies *within* each training fold; held-out data remains untouched)
    rf_pipeline = make_pipeline(
    ConditionalSMOTE(k_neighbors=5, random_state=42, sampling_strategy='not majority'),
    RandomForestClassifier(random_state=42)
    )
    
else:
    print("SMOTE NOT applied in pipeline.")
    # Pipeline: RandomForest only
    rf_pipeline = make_pipeline(
    RandomForestClassifier(random_state=42)
    )

# Hyperparameter space (prefix with the auto-generated step name 'randomforestclassifier__')
param_dist = {
    'randomforestclassifier__n_estimators': randint(100, 500),
    'randomforestclassifier__max_depth': [10, 20, 30, None],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}

# Set up the random search for hyperparameter tuning with random CV=10
random_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=cv_strategy, # Defined earlier based on model_strategy
    scoring='f1_macro',
    random_state=42,
    verbose=1
)

# IMPORTANT: Fit on the original data. The pipeline handles SMOTE inside CV.
if groups is not None:
    random_search.fit(x, y, groups=groups)
else:
    random_search.fit(x, y)

best_rf = random_search.best_estimator_
# Training complete for Random Forest (details saved to Mod1_results)
# Export Random Forest CV metrics and search results
export_cv_metrics(best_rf, x, y, cv=cv_strategy, prefix='CV_rf', groups=groups)
export_search_cv_results(random_search, 'CV_rf')
try:
    joblib.dump(best_rf, results_dir / f"{RESULTS_PREFIX}rf_best_estimator.joblib")
    print(f"Saved best RF estimator to: {results_dir / f'{RESULTS_PREFIX}rf_best_estimator.joblib'}")
except Exception as e:
    print(f"Failed to save RF estimator: {e}")


# %% Training Support Vector Machine (SVM)
# Check first if do_smote is set to True or False
if do_smote:
    print("Conditional SMOTE applied in pipeline.")
    # Pipeline: SMOTE -> SVC (RBF)
    svm_pipeline = make_pipeline(
        ConditionalSMOTE(k_neighbors=5, random_state=42, sampling_strategy='not majority'),
        SVC(kernel='rbf', probability=False, random_state=42)
    )
else:
    print("SMOTE NOT applied in pipeline.")
    # Pipeline: SVC (RBF) only
    svm_pipeline = make_pipeline(
        SVC(kernel='rbf', probability=False, random_state=42)
    )

# Parameter distributions (note the 'svc__' prefix)
param_dist_svm = {
    'svc__C': loguniform(1e-2, 1e2),
    'svc__gamma': loguniform(1e-5, 1e0),
}

# Random CV (same as other models)
random_search_svm = RandomizedSearchCV(
    estimator=svm_pipeline,
    param_distributions=param_dist_svm,
    n_iter=20,
    cv=cv_strategy,
    scoring='f1_macro',
    random_state=42,
    verbose=1
)
if groups is not None:
    random_search_svm.fit(x, y, groups=groups)
else:
    random_search_svm.fit(x, y)
best_svm = random_search_svm.best_estimator_

# Training complete for SVM (details saved to Mod1_results)
# Export SVM CV metrics and search results
export_cv_metrics(best_svm, x, y, cv=cv_strategy, prefix='CV_svm', groups=groups)
export_search_cv_results(random_search_svm, 'CV_svm')
try:
    joblib.dump(best_svm, results_dir / f"{RESULTS_PREFIX}svm_best_estimator.joblib")
    print(f"Saved best SVM estimator to: {results_dir / f'{RESULTS_PREFIX}svm_best_estimator.joblib'}")
except Exception as e:
    print(f"Failed to save SVM estimator: {e}")


# %% Trainilayer perceptron (MLP)
# Check first if do_smote is set to True or False
if do_smote:
    print("Conditional SMOTE applied in pipeline.")
    # Pipeline: SMOTE -> MLP
    mlp_pipeline = make_pipeline(
        ConditionalSMOTE(k_neighbors=5, random_state=42, sampling_strategy='not majority'),
        MLPClassifier(
            max_iter=1000,
            early_stopping=False,
            n_iter_no_change=20,
            random_state=42
        )
    )
else:
    print("SMOTE NOT applied in pipeline.")
    # Pipeline: MLP only
    mlp_pipeline = make_pipeline(
        MLPClassifier(
            max_iter=1000,
            early_stopping=False,
            n_iter_no_change=20,
            random_state=42
        )
    )

# Hyperparameter distributions (prefix with 'mlpclassifier_')
hidden_layer_options = [
    (50,), (100,), #(200,),
    (100, 50), (150, 75), #(200, 100),
    (100, 100)#, (150, 100, 50)
]

param_dist_mlp = {
    'mlpclassifier__hidden_layer_sizes': hidden_layer_options,
    'mlpclassifier__activation': ['relu'],
    'mlpclassifier__alpha': loguniform(1e-3, 1e-2),              # L2 regularization
    'mlpclassifier__learning_rate_init': loguniform(1e-4, 1e-3), # step size
    'mlpclassifier__solver': ['adam']
}

# Random CV (same as other models)
random_search_mlp = RandomizedSearchCV(
    estimator=mlp_pipeline,
    param_distributions=param_dist_mlp,
    n_iter=20,
    cv=cv_strategy,
    scoring='f1_macro',
    random_state=42,
    verbose=1
)

# Fit on original data; if do_smote true then SMOTE & scaling happen inside CV folds
if groups is not None:
    random_search_mlp.fit(x, y, groups=groups)
else:
    random_search_mlp.fit(x, y)

best_mlp = random_search_mlp.best_estimator_
# Training complete for MLP (details saved to Mod1_results)
# Export MLP CV metrics and search results
export_cv_metrics(best_mlp, x, y, cv=cv_strategy, prefix='CV_mlp', groups=groups)
export_search_cv_results(random_search_mlp, 'CV_mlp')

try:
    joblib.dump(best_mlp, results_dir / f"{RESULTS_PREFIX}mlp_best_estimator.joblib")
    print(f"Saved best MLP estimator to: {results_dir / f'{RESULTS_PREFIX}mlp_best_estimator.joblib'}")
except Exception as e:
    print(f"Failed to save MLP estimator: {e}")


#%% Evaluate best models on the test sets and export results
# Select only those columns for features
x_test_DH10 = df_test_DH10[feature_cols]
y_test_DH10 = df_test_DH10['Final_v5_labels_Rock_type_FS']

# Run the best models on the test set and export results
run_on_test_set_and_export(best_rf, x_test_DH10, y_test_DH10, prefix='DH10_test_rf')
run_on_test_set_and_export(best_svm, x_test_DH10, y_test_DH10, prefix='DH10_test_svm')
run_on_test_set_and_export(best_mlp, x_test_DH10, y_test_DH10, prefix='DH10_test_mlp')


# Select only those columns for features
x_test_DH14 = df_test_DH14[feature_cols]
y_test_DH14 = df_test_DH14['Final_v5_labels_Rock_type_FS']

# Run the best models on the test set and export results
run_on_test_set_and_export(best_rf, x_test_DH14, y_test_DH14, prefix='DH14_test_rf')
run_on_test_set_and_export(best_svm, x_test_DH14, y_test_DH14, prefix='DH14_test_svm')
run_on_test_set_and_export(best_mlp, x_test_DH14, y_test_DH14, prefix='DH14_test_mlp')


# Select only those columns for features
x_test_DH18 = df_test_DH18[feature_cols]
y_test_DH18 = df_test_DH18['Final_v5_labels_Rock_type_FS']

# Run the best models on the test set and export results
run_on_test_set_and_export(best_rf, x_test_DH18, y_test_DH18, prefix='DH18_test_rf')
run_on_test_set_and_export(best_svm, x_test_DH18, y_test_DH18, prefix='DH18_test_svm')
run_on_test_set_and_export(best_mlp, x_test_DH18, y_test_DH18, prefix='DH18_test_mlp')

# %% End of script