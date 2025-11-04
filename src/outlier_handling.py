import pandas as pd
import os

def cap_outliers_iqr(df, columns):
    """
    Caps outliers in given numeric columns using IQR method.
    """
    df_capped = df.copy()
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_capped[col] = df[col].clip(lower, upper)
    return df_capped


def main():
    # Input file — from preprocess step
    processed_path = os.path.join("data", "processed", "Accident_Severity_classification_ready.csv")
    df = pd.read_csv(processed_path)

    # Columns to cap — numeric features in our dataset
    numeric_cols = ["Number_of_Casualties", "Number_of_Vehicles", "Speed_limit"]

    print("Capping outliers using IQR for columns:", numeric_cols)
    df_capped = cap_outliers_iqr(df, numeric_cols)

    # Save cleaned version
    output_path = "data/processed/Accident_Severity_no_outliers.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_capped.to_csv(output_path, index=False)
    print(f"\nOutlier-handled dataset saved to: {output_path}")
    print("Final shape:", df_capped.shape)


if __name__ == "__main__":
    main()
