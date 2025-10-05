import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

def test_model_on_excel(excel_path):
    """
    Tests a trained LightGBM model on a sample Excel file containing CTG data.
    Displays per-row predictions, class probabilities, balanced accuracy, and macro F1 score.

    Args:
        excel_path (str): Path to the Excel file (e.g., 'CTG_first10_combined.xlsx')

    Returns:
        pd.DataFrame: DataFrame comparing true vs predicted NSP values and probabilities
    """

    # --- Step 1: Load trained model ---
    model_path = "lightgbm_ctg_model.pkl"
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.\n")

    # --- Step 2: Load Excel file ---
    df = pd.read_excel(excel_path)

    # --- Step 3: Define expected columns ---
    target_col = 'NSP'
    selected_features = ['Variance', 'DP', 'AC', 'UC', 'Mode', 'Median', 'ASTV', 'ALTV']

    # --- Step 4: Check for missing or extra columns ---
    missing = [col for col in selected_features if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in Excel: {missing}")

    extra = [col for col in df.columns if col not in selected_features + [target_col]]
    if extra:
        print(f"⚠️ Warning: Extra columns in Excel ignored: {extra}\n")

    # --- Step 5: Extract features & target ---
    X_sample = df[selected_features]
    y_true = df[target_col]

    # --- Step 6: Align feature order with model ---
    if hasattr(model, "feature_name_"):
        model_features = list(model.feature_name_)
        if model_features != list(X_sample.columns):
            print("⚠️ Reordering columns to match model training feature order...")
            X_sample = X_sample[model_features]
    else:
        print("⚠️ Model has no feature_name_ attribute (may cause mismatch).")

    # --- Step 7: Predict ---
    y_pred = model.predict(X_sample)
    y_proba = model.predict_proba(X_sample)

    # --- Step 8: Build results DataFrame ---
    proba_df = pd.DataFrame(
        y_proba,
        columns=[f"P(NSP={cls+1})" for cls in range(y_proba.shape[1])]
    )

    results_df = pd.concat([
        pd.DataFrame({'True_NSP': y_true, 'Predicted_NSP': y_pred}),
        proba_df
    ], axis=1)

    # --- Step 9: Compute metrics ---
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    # --- Step 10: Display results ---
    pd.set_option('display.float_format', lambda x: f"{x:.3f}")
    print("=== Model Predictions with Class Probabilities ===")
    print(results_df.to_string(index=False))
    print("\n=== Evaluation Metrics ===")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")

    # --- Step 11: Save output ---
    results_df.to_excel("test_results_output.xlsx", index=False)
    print("\n✅ Results saved as 'test_results_output.xlsx'.")

    return results_df, balanced_acc, macro_f1


# Example run
if __name__ == "__main__":
    results, balanced_acc, macro_f1 = test_model_on_excel("CTG_first10_combined.xlsx")
