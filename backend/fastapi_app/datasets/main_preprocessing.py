import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import preprocess_loan_data

def main():
    """
    Main function to preprocess loan application data and save processed files.
    """
    # Load the data files
    try:
        # Try loading your merged dataset first
        df_merged = pd.read_csv("synthetic_loan_applications_org.csv")
        print(f"Loaded merged dataset with {len(df_merged)} rows and {len(df_merged.columns)} columns")
    except FileNotFoundError:
        try:
            # If that fails, try loading and merging the separate files
            df_applicant = pd.read_csv("applicant_dataset_org.csv")
            df_credit = pd.read_csv("third_party_dataset_org.csv")
            print(f"Loaded applicant dataset with {len(df_applicant)} rows")
            print(f"Loaded credit dataset with {len(df_credit)} rows")
            
            # Merge the datasets
            df_merged = pd.merge(df_applicant, df_credit, on="applicant_id", how="inner")
            print(f"Merged dataset has {len(df_merged)} rows")
        except FileNotFoundError:
            print("Data files not found. Please ensure the CSV files are in the current directory.")
            return
    
    # Run preprocessing on the merged dataset
    print("Preprocessing data...")
    processed_df = preprocess_loan_data(df_merged)
    
    # Save the preprocessed data
    processed_df.to_csv("processed_loan_applications.csv", index=False)
    print(f"Saved preprocessed data with {len(processed_df)} rows to 'processed_loan_applications.csv'")
    
    # Split into train/test sets for model training
    train_df, test_df = train_test_split(processed_df, test_size=0.2, random_state=42)
    
    # Save the split datasets
    train_df.to_csv("train_loan_applications.csv", index=False)
    test_df.to_csv("test_loan_applications.csv", index=False)
    print(f"Created and saved train ({len(train_df)} rows) and test ({len(test_df)} rows) datasets")
    
    # Display summary statistics to verify preprocessing
    print("\nSummary statistics for key columns in preprocessed data:")
    print(processed_df[['annual_income', 'credit_score', 'DTI', 'approved']].describe())
    
    # Calculate and display approval rate
    approval_rate = processed_df['approved'].mean() * 100
    print(f"\nApproval rate: {approval_rate:.2f}%")
    
    # Display noise analysis - check for outliers after preprocessing
    noise_check = {
        'Income outliers': len(processed_df[(processed_df['annual_income'] < 20000) | 
                                          (processed_df['annual_income'] > 200000)]),
        'Credit score outliers': len(processed_df[(processed_df['credit_score'] < 300) | 
                                                (processed_df['credit_score'] > 900)]),
        'Negative months employed': len(processed_df[processed_df['months_employed'] < 0]),
        'Months employed > 600': len(processed_df[processed_df['months_employed'] > 600])
    }
    print("\nNoise check after preprocessing:")
    for check, count in noise_check.items():
        print(f"{check}: {count}")
    
    # Check for missing values after preprocessing
    missing_values = processed_df.isna().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print("\nMissing values after preprocessing:")
        print(missing_cols)
    else:
        print("\nNo missing values after preprocessing")

if __name__ == "__main__":
    main()