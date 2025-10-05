# Core libraries and model APIs
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def clean_data(df: pd.DataFrame):
    """
    Cleans the input dataframe by:
    1. Removing empty rows and columns.
    2. Keeping only the specified features and the target.
    3. Removing duplicate rows.
    4. Dropping rows with missing target values.
    5. Splitting into features (X) and target (y).
    
    Args:
        df (pd.DataFrame): The raw dataframe.
    
    Returns:
        tuple: (X, y) where
            X (pd.DataFrame): Feature dataframe (restricted to chosen features).
            y (pd.Series): Target column ('NSP').
    """
    # Step 1: Make a working copy of the dataframe
    cleaned = df.copy()
    
    # Step 2: Drop any rows or columns that are completely empty
    cleaned = cleaned.dropna(axis=0, how='all').dropna(axis=1, how='all')
    
    # Step 3: Define target and allowed features
    target_col = 'NSP'
    allowed_features = {'ASTV', 'DP', 'ALTV', 'Median', 'Variance', 'AC', 'UC', 'Mode'}
    
    # Step 4: Keep only allowed features + target (with validation)
    available_features = [c for c in allowed_features if c in cleaned.columns]
    
    # Warn if target column is missing
    if target_col not in cleaned.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")
    
    # Warn about missing features (optional but helpful)
    missing_features = allowed_features - set(available_features)
    if missing_features:
        print(f"Warning: The following features are missing: {missing_features}")
    
    keep_cols = available_features + [target_col]
    cleaned = cleaned[keep_cols]
    
    # Step 5: Remove duplicate rows
    cleaned = cleaned.drop_duplicates()
    
    # Step 6: Drop rows where target is missing
    cleaned = cleaned.dropna(subset=[target_col])
    
    # Step 7: Handle missing values in features (optional - you might want to impute)
    # For now, dropping rows with any missing feature values
    initial_rows = len(cleaned)
    cleaned = cleaned.dropna()
    dropped_rows = initial_rows - len(cleaned)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing feature values")
    
    # Step 8: Separate features and target
    X = cleaned.drop(columns=[target_col])
    y = cleaned[target_col]

    # Add at the end before returning:
    # Reset index to avoid gaps after dropping rows
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    print(f"Final dataset: {len(X)} samples, {len(X.columns)} features")
    
    return X, y


df = pd.read_excel('ctg.xlsx', sheet_name = 'Data', header = 1)

X, y = clean_data(df)

########################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import make_scorer, balanced_accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import lightgbm as lgb


def evaluate_with_smote_cv_lightgbm_optimal2(
    X, y, 
    test_size=0.2, 
    n_splits=5, 
    random_state=42,
    use_early_stopping=True,
    early_stopping_rounds=50,
    use_class_weights=False,
    smote_strategy='auto',
    verbose=True
):
    """
    Enhanced LightGBM training with SMOTE + Stratified K-Fold CV.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target column
        test_size (float): Proportion for test split
        n_splits (int): Number of folds for cross-validation
        random_state (int): Random seed
        use_early_stopping (bool): Whether to use early stopping for final model
        early_stopping_rounds (int): Number of rounds for early stopping
        use_class_weights (bool): Use class_weight='balanced' instead of/with SMOTE
        smote_strategy (str): SMOTE sampling strategy ('auto', 'not majority', 'all')
        verbose (bool): Print detailed output
    
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, cv_scores, 
                test_balanced_accuracy, feature_importance, hyperparameters, predictions)
    """
    # --- 1. Train/test split with stratification ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    if verbose:
        print("=" * 60)
        print("TRAINING CONFIGURATION")
        print("=" * 60)
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Class distribution (train): {dict(y_train.value_counts().sort_index())}")
        print(f"Class distribution (test): {dict(y_test.value_counts().sort_index())}")
        print()
    
    # --- 2. Best parameters from Optuna (Updated) ---
    best_params = {
        'learning_rate': 0.015148189601668145,
        'num_leaves': 24,
        'max_depth': 3,
        'min_child_samples': 25,
        'min_split_gain': 0.44871424998167664,
        'lambda_l1': 0.799084665275944,
        'lambda_l2': 0.7644980529191521,
        'feature_fraction': 0.9596036089327271,
        'bagging_fraction': 0.9098171071119672,
        'bagging_freq': 1,
        'extra_trees': False,
        'n_estimators': 295,
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': random_state,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Add class weights if requested
    if use_class_weights:
        best_params['class_weight'] = 'balanced'
        if verbose:
            print("Using class_weight='balanced'")
    
    # --- 3. Stratified CV evaluation with SMOTE ---
    if verbose:
        print("=" * 60)
        print("CROSS-VALIDATION EVALUATION")
        print("=" * 60)
    
    pipeline = Pipeline(steps=[
        ('smote', SMOTE(random_state=random_state, sampling_strategy=smote_strategy)),
        ('lgbm', LGBMClassifier(**best_params))
    ])
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scorer = make_scorer(balanced_accuracy_score)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=scorer, n_jobs=-1)
    
    if verbose:
        print(f"Balanced Accuracy per fold: {np.round(cv_scores, 4)}")
        print(f"Mean Balanced Accuracy: {np.round(np.mean(cv_scores), 4)} ± {np.round(np.std(cv_scores), 4)}")
        print()
    
    # --- 4. Train final model with SMOTE applied on training set ---
    if verbose:
        print("=" * 60)
        print("FINAL MODEL TRAINING")
        print("=" * 60)
    
    smote = SMOTE(random_state=random_state, sampling_strategy=smote_strategy)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    if verbose:
        print(f"After SMOTE - Train size: {len(X_train_bal)}")
        print(f"Class distribution (balanced): {dict(pd.Series(y_train_bal).value_counts().sort_index())}")
        print()
    
    model = LGBMClassifier(**best_params)
    
    # Fit with or without early stopping
    if use_early_stopping:
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model.fit(
            X_train_bal, y_train_bal,
            eval_set=[(X_test, y_test)],
            callbacks=callbacks
        )
        if verbose and hasattr(model, 'best_iteration_'):
            print(f"Best iteration: {model.best_iteration_} (out of {best_params['n_estimators']})")
    else:
        model.fit(X_train_bal, y_train_bal)
    
    # --- 5. Feature importance ---
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # --- 6. Test set evaluation (before retraining) ---
    y_pred = model.predict(X_test)
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    
    if verbose:
        print(f"\nTest Set Balanced Accuracy: {np.round(test_balanced_accuracy, 4)}")
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred))
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
    
    # Save predictions before retraining
    test_predictions = y_pred.copy()
    
    # --- 7. Retrain on full dataset ---
    if verbose:
        print("\n" + "=" * 60)
        print("RETRAINING ON FULL DATASET (TRAIN + TEST)")
        print("=" * 60)
    
    # Combine train and test
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)
    
    # Apply SMOTE to full dataset
    smote_full = SMOTE(random_state=random_state, sampling_strategy=smote_strategy)
    X_full_bal, y_full_bal = smote_full.fit_resample(X_full, y_full)
    
    if verbose:
        print(f"Full dataset size: {len(X_full_bal)}")
        print(f"Class distribution: {dict(pd.Series(y_full_bal).value_counts().sort_index())}")
    
    # Retrain model on everything
    final_model = LGBMClassifier(**best_params)
    final_model.fit(X_full_bal, y_full_bal)
    
    if verbose:
        print("✓ Model retrained on full dataset")
        print("=" * 60)
    
    # Return final_model with test predictions from before retraining
    return (final_model, X_train, X_test, y_train, y_test, cv_scores, 
            test_balanced_accuracy, feature_importance, best_params, test_predictions)


# Example usage:
"""
model, X_train, X_test, y_train, y_test, cv_scores, test_balanced_accuracy, feature_importance, hyperparameters, predictions = evaluate_with_smote_cv_lightgbm_optimal2(
    X, y,
    test_size=0.2,
    n_splits=5,
    random_state=42,
    use_early_stopping=True,
    early_stopping_rounds=50,
    use_class_weights=False,
    verbose=True
)

# Now you can use variables directly
print(f"Model: {model}")
print(f"CV Scores: {cv_scores}")
print(f"Test Accuracy: {test_balanced_accuracy}")
"""



model, X_train, X_test, y_train, y_test, cv_scores, test_balanced_accuracy, feature_importance, hyperparameters, predictions = evaluate_with_smote_cv_lightgbm_optimal2(
    X, y,
    test_size=0.2,
    n_splits=5,
    random_state=42,
    use_early_stopping=True,
    early_stopping_rounds=50,
    use_class_weights=False,
    verbose=True
)

# Now you can use variables directly
print(f"Model: {model}")
print(f"CV Scores: {cv_scores}")
print(f"Test Accuracy: {test_balanced_accuracy}")

# ============================================================
# Save the final model to disk for later testing
# ============================================================
import joblib

# Save the trained model
joblib.dump(model, "lightgbm_ctg_model.pkl")
print("\n✅ Model saved as 'lightgbm_ctg_model.pkl' in current directory.")
