import pandas as pd
import os
def load_and_clean_dataset(file_path):
    df = pd.read_csv(file_path)

    # Drop unnecessary columns
    cols_to_drop = ["Accident_Index", "Latitude", "Longitude"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Parse Accident Date
    df["Accident Date"] = pd.to_datetime(df["Accident Date"], errors="coerce")
    df["Day"] = df["Accident Date"].dt.day
    df["Month"] = df["Accident Date"].dt.month
    df = df.drop(columns=["Accident Date"])

    # Clean numeric columns
    numeric_cols = ["Number_of_Casualties", "Number_of_Vehicles", "Speed_limit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing essential fields
    essential_cols = ["Local_Authority_(District)", "Junction_Detail"]
    df.dropna(subset=essential_cols, inplace=True)

    return df


def fill_missing_values(df, group_cols=["Month", "Day_of_Week", "Urban_or_Rural_Area"]):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Fill missing numeric values
    for col in numeric_cols:
        df[col] = df.groupby(group_cols)[col].transform(lambda x: x.fillna(x.mean()))

    # Fill missing categorical values, including target
    cat_fill_cols = [
        "Day_of_Week", "Time", "Road_Type", "Weather_Conditions",
        "Light_Conditions", "Urban_or_Rural_Area", "Accident_Severity"
    ]

    for col in cat_cols:
        if col in cat_fill_cols or col in df.columns:
            df[col] = df.groupby(group_cols)[col].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
            )

    return df


def main():
    print("Starting preprocessing...")

    df = load_and_clean_dataset("data/raw/accidents_2022.csv")
    df = fill_missing_values(df)

    # Move Month, Day to front for readability
    for col in ["Month", "Day"]:
        if col in df.columns:
            df.insert(0, col, df.pop(col))

    print("\nPreprocessing done. Final shape:", df.shape)
    print(df.head())

    output_path = "data/processed/Accident_Severity_classification_ready.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset exported to: {output_path}")


if __name__ == "__main__":
    main()
