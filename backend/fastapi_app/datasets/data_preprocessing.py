import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

def preprocess_loan_data(df, handle_missing=True, handle_noise=True):
    """
    Preprocesses loan application data to increase model efficiency
    
    Parameters:
    df (pandas.DataFrame): The input loan application dataframe
    handle_missing (bool): Whether to handle missing values
    handle_noise (bool): Whether to handle outliers and noise
    
    Returns:
    pandas.DataFrame: Preprocessed dataframe
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # --- 1. Handle missing values ---
    if handle_missing:
        # Identify numeric and categorical columns
        numeric_cols = processed_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove 'applicant_id' from lists if present
        if 'applicant_id' in numeric_cols:
            numeric_cols.remove('applicant_id')
        if 'applicant_id' in categorical_cols:
            categorical_cols.remove('applicant_id')
            
        # Handle missing numeric values using KNN imputation
        if numeric_cols:
            # Scale data for KNN imputation
            scaler = StandardScaler()
            numeric_data = processed_df[numeric_cols].copy()
            
            # Only scale columns with no missing values for fitting
            cols_for_scaling = [col for col in numeric_cols if not numeric_data[col].isna().any()]
            
            if cols_for_scaling:
                numeric_data_scaled = pd.DataFrame(
                    scaler.fit_transform(numeric_data[cols_for_scaling]),
                    columns=cols_for_scaling,
                    index=numeric_data.index
                )
                
                # Replace scaled columns in numeric_data
                for col in cols_for_scaling:
                    numeric_data[col] = numeric_data_scaled[col]
                    
            # Apply KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            imputed_data = imputer.fit_transform(numeric_data)
            
            # Replace values in original dataframe
            for i, col in enumerate(numeric_cols):
                processed_df[col] = imputed_data[:, i]
                
        # Handle categorical missing values with mode
        for col in categorical_cols:
            if processed_df[col].isna().any():
                mode_value = processed_df[col].mode()[0]
                processed_df[col].fillna(mode_value, inplace=True)
    
    # --- 2. Handle noise and outliers ---
    if handle_noise:
        # Fix annual income outliers
        # Modified ranges: original was 20000-200000, data generation now restricts at generation time
        income_mask = (processed_df['annual_income'] < 20000) | (processed_df['annual_income'] > 200000)
        median_income = processed_df.loc[~income_mask, 'annual_income'].median()
        processed_df.loc[income_mask, 'annual_income'] = median_income
        
        # Fix credit score outliers
        # Original range 300-900, data generation has modified distribution parameters but same range
        score_mask = (processed_df['credit_score'] < 300) | (processed_df['credit_score'] > 900)
        median_score = processed_df.loc[~score_mask, 'credit_score'].median()
        processed_df.loc[score_mask, 'credit_score'] = median_score
        
        # Fix months_employed outliers (negative values)
        employed_mask = processed_df['months_employed'] < 0
        processed_df.loc[employed_mask, 'months_employed'] = 0
        
        # Cap months_employed to 600
        over_cap_mask = processed_df['months_employed'] > 600
        processed_df.loc[over_cap_mask, 'months_employed'] = 600
        
        # Fix requested_amount if it exceeds modified upper limit (30000 instead of 50000)
        if 'requested_amount' in processed_df:
            req_amt_mask = processed_df['requested_amount'] > 30000
            processed_df.loc[req_amt_mask, 'requested_amount'] = 30000
    
    # --- 3. Recalculate derived fields ---
    # Monthly income
    processed_df['monthly_income'] = processed_df['annual_income'] / 12.0
    
    # Self-reported debt ratio - handling the modified factor range (0.075-0.25)
    # This is normally part of data generation but we can check if values are outside this range
    if 'self_reported_debt' in processed_df.columns and 'monthly_income' in processed_df.columns:
        debt_ratio = processed_df['self_reported_debt'] / processed_df['monthly_income']
        # Fix debt ratios that are outside the modified range
        invalid_debt = (debt_ratio < 0.075) | (debt_ratio > 0.25)
        if invalid_debt.any():
            # Reset to median of valid range
            median_ratio = 0.1625  # middle of 0.075-0.25
            processed_df.loc[invalid_debt, 'self_reported_debt'] = processed_df.loc[invalid_debt, 'monthly_income'] * median_ratio
    
    # Estimated debt - using modified factor of 0.025 instead of 0.03
    if 'total_credit_limit' in processed_df.columns and 'credit_utilization' in processed_df.columns:
        processed_df['estimated_debt'] = (
            processed_df['total_credit_limit'] * 
            (processed_df['credit_utilization'] / 100.0) * 
            0.025  # Modified from 0.03 to 0.025
        )
    
    # DTI calculation - using modified factor of 0.015 instead of 0.03
    if 'self_reported_debt' in processed_df.columns and 'estimated_debt' in processed_df.columns:
        monthly_debt = processed_df['self_reported_debt'] + processed_df['estimated_debt']
        monthly_income = processed_df['annual_income'] / 12.0
        processed_df['DTI'] = (
            (monthly_debt + (processed_df['requested_amount'] * 0.015)) / 
            monthly_income * 
            100.0  # Modified from 0.03 to 0.015
        )
    
    # --- 4. Apply approval rules based on modified criteria ---
    if all(col in processed_df.columns for col in ['credit_score', 'DTI', 'payment_history']):
        processed_df['approved'] = processed_df.apply(
            lambda row: 
                1 if (
                    row['credit_score'] >= 660 and 
                    row['DTI'] <= 40.0 and 
                    row['payment_history'].lower() in ["ontime", "on-time", "on_time"]
                ) 
                else (
                    0 if (
                        row['credit_score'] < 500 or 
                        row['DTI'] > 50.0 or 
                        row['credit_utilization'] > 80.0
                    ) 
                    else np.random.choice([0, 1])
                ),
            axis=1
        )
    
    # --- 5. Calculate approved_amount and interest_rate if needed ---
    if 'approved' in processed_df.columns and not 'approved_amount' in processed_df.columns:
        # Create temporary function to calculate theoretical limit
        def compute_theoretical_limit(row):
            score = row['credit_score']
            inc = row['annual_income']
            dti = row['DTI']
            util = row['credit_utilization']
            
            # Base limit by score
            if score >= 660:
                base = 0.50 * inc
            elif score >= 500:
                base = 0.25 * inc
            else:
                base = 0.10 * inc
                
            # DTI adjustments
            if dti > 40.0:
                base *= 0.5
            elif dti > 30.0:
                base *= 0.75
                
            # Credit utilization adjustment
            if util > 50.0:
                base *= 0.8
                
            # Employment bonus
            if 'employment_status' in row and 'months_employed' in row:
                emp = str(row['employment_status']).lower()
                months_emp = row['months_employed']
                if emp in ["full-time", "full_time"] and months_emp >= 12:
                    base *= 1.10
                    
            # Payment penalty
            if 'payment_history' in row:
                pmt = str(row['payment_history']).lower()
                if pmt in ["late>60", "late>60", "late60+", "60+"]:
                    base *= 0.5
                    
            # Credit score caps
            if score >= 750:
                cap = 25000
            elif score >= 660:
                cap = 15000
            elif score >= 500:
                cap = 10000
            else:
                cap = 5000
                
            return min(base, cap)
        
        # Create temporary function to compute interest rate
        def compute_interest_rate(row):
            # Base rate in range 3.0 to 6.75 (modified from 3.0-9.25)
            base_rate = np.random.uniform(3.0, 6.75)
            score = row['credit_score']
            
            # Score-based adjustments
            if score >= 750:
                base_rate -= 1.0
            elif score < 500:
                base_rate += 4.0
            elif score < 660:  # 500..659
                base_rate += 2.0
                
            # DTI adjustment
            if row['DTI'] > 30.0:
                base_rate += 1.0
                
            # Payment history adjustment
            if 'payment_history' in row:
                pmt = str(row['payment_history']).lower()
                if pmt in ["late>60", "late>60", "late60+", "60+"]:
                    base_rate += 2.0
                    
            # Num open accounts adjustment
            if 'num_open_accounts' in row and row['num_open_accounts'] > 5:
                base_rate += 1.0
                
            # Clamp to range [3, 15]
            return min(max(base_rate, 3.0), 15.0)
        
        # Apply calculations based on approval status
        approved_amounts = []
        interest_rates = []
        
        for _, row in processed_df.iterrows():
            if row['approved'] == 0:
                approved_amounts.append(0.0)
                interest_rates.append(0.0)
            else:
                limit = compute_theoretical_limit(row)
                amt = min(row['requested_amount'], limit)
                rate = compute_interest_rate(row)
                approved_amounts.append(amt)
                interest_rates.append(rate)
                
        processed_df['approved_amount'] = approved_amounts
        processed_df['interest_rate'] = interest_rates
    
    return processed_df