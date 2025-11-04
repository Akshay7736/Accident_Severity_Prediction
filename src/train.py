import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from imblearn.over_sampling import SMOTE
import joblib
import os


# =========================
# Load Data
# =========================
def load_data():
    file_path = os.path.join("data", "processed", "Accident_Severity_no_outliers.csv")
    df = pd.read_csv(file_path)
    print("Data Loaded. Shape:", df.shape)
    return df


# =========================
# Label Encoding for Categorical Columns
# =========================
def build_label_mappings(train_df):
    mappings = {}
    cat_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        unique_vals = list(train_df[col].astype(str).unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        mapping["_unknown_"] = len(unique_vals)
        mappings[col] = mapping
    return mappings


def apply_mappings(df, mappings):
    df_out = df.copy()
    for col, mapping in mappings.items():
        if col in df_out.columns:
            df_out[col] = df_out[col].astype(str).map(lambda x: mapping.get(x, mapping["_unknown_"]))
    return df_out


# =========================
# Evaluate Model
# =========================
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\nModel: {name}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-Score : {f1:.3f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    return acc


# =========================
# Main Training Logic
# =========================
def main():
    df = load_data()
    target_col = "Accident_Severity"
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print("Train/Test Split:", X_train.shape, X_test.shape)

    # Encode categorical variables
    mappings = build_label_mappings(X_train)
    X_train_enc = apply_mappings(X_train, mappings)
    X_test_enc = apply_mappings(X_test, mappings)

    # =========================
    # ✅ Handle Class Imbalance with SMOTE
    # =========================
    print("\nApplying SMOTE to balance minority classes...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)
    print("After SMOTE Resampling:")
    print(y_train_bal.value_counts())

    # =========================
    # Define Models
    # =========================
    models = {
        "1": ("Random Forest", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # ✅ Makes it sensitive to minority classes
        )),
        "2": ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight='balanced')),
        "3": ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced')),
        "4": ("Gradient Boosting", GradientBoostingClassifier(random_state=42))
    }

    # =========================
    # Train & Evaluate Models
    # =========================
    scores = {}
    for key, (name, model) in models.items():
        print(f"\n{'='*10} Training {name} {'='*10}")
        model.fit(X_train_bal, y_train_bal)
        acc = evaluate_model(name, model, X_test_enc, y_test)
        scores[key] = (name, acc)

    # =========================
    # Model Comparison
    # =========================
    print("\nModel Accuracy Comparison:")
    print("--------------------------------")
    for key, (name, acc) in scores.items():
        print(f"{key}. {name:<20} Accuracy: {acc:.3f}")

    # =========================
    # Save Best Model
    # =========================
    choice = input("\nEnter the model number to save (e.g. 1): ").strip()
    if choice not in models:
        print("Invalid choice. Exiting without saving.")
        return

    chosen_name, chosen_model = models[choice]
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "model.pkl")
    mapping_path = os.path.join("models", "label_mappings.pkl")

    joblib.dump(chosen_model, model_path)
    joblib.dump(mappings, mapping_path)

    print(f"\nModel '{chosen_name}' saved as: {model_path}")
    print(f"Label mappings saved to: {mapping_path}")


# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    main()
