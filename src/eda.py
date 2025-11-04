import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    # Load processed dataset (after outlier handling)
    processed_path = os.path.join("data", "processed", "Accident_Severity_no_outliers.csv")
    df = pd.read_csv(processed_path)
    print("Dataset loaded successfully. Shape:", df.shape)
    print(df.head())

    # Basic info
    print("\n=== INFO ===")
    print(df.info())
    print("\n=== DESCRIPTION ===")
    print(df.describe())

    # Target variable distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x="Accident_Severity", data=df, palette="Set2")
    plt.title("Distribution of Accident Severity")
    plt.xlabel("Accident Severity")
    plt.ylabel("Count")
    plt.show()

    # Severity vs Day_of_Week
    plt.figure(figsize=(8,4))
    sns.countplot(x="Day_of_Week", hue="Accident_Severity", data=df, palette="coolwarm")
    plt.title("Accident Severity by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.show()

    # Severity vs Weather Conditions
    plt.figure(figsize=(10,5))
    sns.countplot(x="Weather_Conditions", hue="Accident_Severity", data=df, palette="magma")
    plt.title("Accident Severity by Weather Conditions")
    plt.xticks(rotation=45)
    plt.show()

    # Severity vs Road Type
    plt.figure(figsize=(8,4))
    sns.countplot(x="Road_Type", hue="Accident_Severity", data=df, palette="viridis")
    plt.title("Accident Severity by Road Type")
    plt.xticks(rotation=45)
    plt.show()

    # Severity vs Light Conditions
    plt.figure(figsize=(8,4))
    sns.countplot(x="Light_Conditions", hue="Accident_Severity", data=df, palette="Set1")
    plt.title("Accident Severity by Light Conditions")
    plt.xticks(rotation=45)
    plt.show()

    # Severity vs Urban or Rural Area
    plt.figure(figsize=(6,4))
    sns.countplot(x="Urban_or_Rural_Area", hue="Accident_Severity", data=df, palette="cubehelix")
    plt.title("Accident Severity by Area Type")
    plt.show()

    # Correlation heatmap (numeric features only)
    plt.figure(figsize=(8,6))
    numeric_cols = df.select_dtypes(include="number").columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.show()

    # Boxplots for numeric variables
    for col in numeric_cols:
        plt.figure(figsize=(6,3))
        sns.boxplot(x=df[col], color="skyblue")
        plt.title(f"Boxplot - {col}")
        plt.show()

    # Distributions of numeric variables
    df[numeric_cols].hist(figsize=(12,8), bins=20)
    plt.suptitle("Distribution of Numeric Features")
    plt.show()

if __name__ == "__main__":
    main()
