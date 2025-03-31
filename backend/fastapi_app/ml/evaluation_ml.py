import pandas as pd
import numpy as np
import os
import joblib
import sys
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    mean_absolute_error, recall_score, precision_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def generate_fresh_evaluation_data(num_samples=1000, output_file=None):
    """
    Generate a fresh dataset for evaluation by calling the synthetic data generation script
    
    Args:
        num_samples: Number of samples to generate
        output_file: Path to save the generated data
        
    Returns:
        Path to the generated dataset file
    """
    print("Generating fresh evaluation dataset...")
    
    # Set default output file if not provided
    if output_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(script_dir), 'datasets', 'data')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'evaluation_loan_applications.csv')
    
    # Import the synthetic data generation module
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
    
    try:
        # Try to import and use the synthetic_data module directly
        import synthetic_data_new as synthetic_data
        
        # Generate applicant dataset
        df_applicant = synthetic_data.generate_applicant_dataset(num_rows=num_samples)
        
        # Generate credit dataset
        df_credit = synthetic_data.generate_credit_dataset(num_rows=num_samples)
        
        # Merge datasets
        df_merged = synthetic_data.create_merged_dataset(df_applicant, df_credit)
        
        # Add estimated debt
        df_merged = synthetic_data.add_estimated_debt(df_merged)
        
        # Compute approval
        df_merged = synthetic_data.finalize_approval(df_merged)
        
        # Compute approved_amount and interest
        df_merged = synthetic_data.finalize_approved_amount_and_interest(df_merged)
        
        # Introduce missingness and noise (optional)
        df_merged = synthetic_data.introduce_missingness(df_merged, missing_frac=0.02)
        df_merged = synthetic_data.introduce_noise(
            df_merged, noise_frac=0.05, 
            columns=["annual_income", "credit_score", "months_employed"]
        )
        
        # Save the dataset
        df_merged.to_csv(output_file, index=False)
        
        print(f"Fresh evaluation dataset with {num_samples} samples saved to {output_file}")
        
    except ImportError:
        # If direct import fails, run the script as a subprocess
        print("Falling back to subprocess method for data generation...")
        
        # Create a temporary Python script to generate the data
        temp_script = os.path.join(os.path.dirname(output_file), 'temp_generate_eval_data.py')
        
        with open(temp_script, 'w') as f:
            f.write(f"""
import sys
import os
sys.path.append('{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
from datasets import synthetic_data

# Generate applicant dataset
df_applicant = synthetic_data.generate_applicant_dataset(num_rows={num_samples})

# Generate credit dataset
df_credit = synthetic_data.generate_credit_dataset(num_rows={num_samples})

# Merge datasets
df_merged = synthetic_data.create_merged_dataset(df_applicant, df_credit)

# Add estimated debt
df_merged = synthetic_data.add_estimated_debt(df_merged)

# Compute approval
df_merged = synthetic_data.finalize_approval(df_merged)

# Compute approved_amount and interest
df_merged = synthetic_data.finalize_approved_amount_and_interest(df_merged)

# Introduce missingness and noise
df_merged = synthetic_data.introduce_missingness(df_merged, missing_frac=0.02)
df_merged = synthetic_data.introduce_noise(
    df_merged, noise_frac=0.05, 
    columns=["annual_income", "credit_score", "months_employed"]
)

# Save the dataset
df_merged.to_csv('{output_file}', index=False)
print(f"Fresh evaluation dataset with {num_samples} samples saved to {output_file}")
""")
        
        # Run the temporary script
        subprocess.run([sys.executable, temp_script], check=True)
        
        # Clean up
        os.remove(temp_script)
    
    return output_file

def load_models_and_artifacts(model_dir=None):
    """
    Load the trained models and preprocessing artifacts
    
    Args:
        model_dir: Directory containing the trained models and artifacts
        
    Returns:
        Dictionary containing the loaded models and artifacts
    """
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    print(f"Loading models from {model_dir}...")
    
    # Load models
    approval_model = joblib.load(os.path.join(model_dir, 'approval_model.pkl'))
    credit_limit_model = joblib.load(os.path.join(model_dir, 'credit_limit_model.pkl'))
    interest_rate_model = joblib.load(os.path.join(model_dir, 'interest_rate_model.pkl'))
    
    # Load preprocessing artifacts
    encoded_feature_columns = joblib.load(os.path.join(model_dir, 'encoded_feature_columns.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    scaled_columns = joblib.load(os.path.join(model_dir, 'scaled_columns.pkl'))
    
    return {
        'approval_model': approval_model,
        'credit_limit_model': credit_limit_model,
        'interest_rate_model': interest_rate_model,
        'encoded_feature_columns': encoded_feature_columns,
        'scaler': scaler,
        'scaled_columns': scaled_columns
    }

def load_and_preprocess_data(file_path=None, generate_new=True, num_samples=1000):
    """
    Load and preprocess the dataset for evaluation
    
    Args:
        file_path: Path to the dataset file
        generate_new: Whether to generate a new dataset
        num_samples: Number of samples to generate if generate_new is True
        
    Returns:
        Tuple containing the preprocessed data and target variables
    """
    if generate_new or file_path is None:
        file_path = generate_fresh_evaluation_data(num_samples=num_samples)
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Print available columns to help with debugging
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Handle missing values
    for col in numeric_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if not df[col].mode().empty:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])
    
    # Define target columns
    target_cols = {
        'approval': 'approved',
        'credit_limit': 'approved_amount',
        'interest': 'interest_rate'
    }
    
    # Define features (exclude target columns and applicant_id)
    features = [col for col in df.columns if col not in target_cols.values() and col != 'applicant_id']
    
    # Update categorical_cols to only include those in features
    categorical_cols = [col for col in categorical_cols if col in features]
    
    # Split data
    X = df[features].copy()
    y_approval = df[target_cols['approval']].copy()
    y_credit_limit = df[target_cols['credit_limit']].copy()
    y_interest = df[target_cols['interest']].copy()
    
    X_train, X_test, y_train_approval, y_test_approval = train_test_split(
        X, y_approval, test_size=0.2, random_state=42
    )
    
    # Get indices for approved loans in test set
    approved_indices = y_test_approval == 1
    
    # Extract credit limit and interest rate only for approved loans
    y_test_credit_limit = df.loc[y_test_approval.index[approved_indices], target_cols['credit_limit']]
    y_test_interest = df.loc[y_test_approval.index[approved_indices], target_cols['interest']]
    
    return {
        'X_test': X_test,
        'y_test_approval': y_test_approval,
        'y_test_credit_limit': y_test_credit_limit,
        'y_test_interest': y_test_interest,
        'categorical_cols': categorical_cols,
        'numeric_cols': numeric_cols,
        'approved_indices': approved_indices,
        'test_indices': y_test_approval.index
    }

def preprocess_test_data(X_test, artifacts, categorical_cols):
    """
    Preprocess the test data using the artifacts from training
    
    Args:
        X_test: Test features DataFrame
        artifacts: Dictionary containing preprocessing artifacts
        categorical_cols: List of categorical columns
        
    Returns:
        Preprocessed test data
    """
    # One-hot encode categorical features
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    # Ensure test set has same columns as train set
    X_test_encoded = X_test_encoded.reindex(columns=artifacts['encoded_feature_columns'], fill_value=0)
    
    # Scale numeric features
    X_test_processed = X_test_encoded.copy()
    X_test_processed[artifacts['scaled_columns']] = artifacts['scaler'].transform(X_test_encoded[artifacts['scaled_columns']])
    
    return X_test_processed

def apply_business_rules(approval, amount, interest_rate, features):
    """
    Apply business rules to ensure compliance with approval conditions, credit limits,
    and interest rate constraints, matching the logic in loan.py.
    
    Parameters:
    -----------
    approval : int
        Initial loan approval status (1 for approved, 0 for denied)
    amount : float
        ML-predicted loan amount
    interest_rate : float
        ML-predicted interest rate
    features : dict or pd.Series
        Features containing applicant information
        
    Returns:
    --------
    tuple
        (final_approval, final_amount, final_interest_rate)
    """
    # Enforce hard approval/denial rules
    if approval == 0:
        return 0, 0.0, 0.0
    
    # Extract necessary features
    credit_score = features['credit_score']
    credit_utilization = features['credit_utilization']
    
    # Extract or calculate DTI
    if 'DTI' in features:
        dti = features['DTI']
    else:
        # Calculate DTI if not present in features
        monthly_income = features['annual_income'] / 12
        total_debt = features['self_reported_debt'] + features['estimated_debt']
        monthly_payment = features['requested_amount'] * 0.03
        dti = (total_debt + monthly_payment) / monthly_income * 100 if monthly_income > 0 else 0
    
    # Deny if any of these conditions are met (regardless of model prediction)
    if (credit_score < 500 or dti > 50 or credit_utilization > 80):
        return 0, 0.0, 0.0
    
    # Base limit based on credit score
    annual_income = features['annual_income']
    if credit_score >= 660:
        base_limit = annual_income * 0.5
    elif credit_score >= 500:
        base_limit = annual_income * 0.25
    else:
        base_limit = annual_income * 0.1
    
    # DTI adjustment
    if dti <= 30:
        dti_factor = 1.0
    elif dti <= 40:
        dti_factor = 0.75
    else:
        dti_factor = 0.5
    
    # Credit score cap
    if credit_score >= 750:
        credit_cap = 25000
    elif credit_score >= 660:
        credit_cap = 15000
    elif credit_score >= 500:
        credit_cap = 10000
    else:
        credit_cap = 5000
    
    # Employment bonus
    employment_status = features['employment_status']
    months_employed = features['months_employed']
    employment_bonus = 1.1 if (employment_status == "Full-time" and months_employed >= 12) else 1.0
    
    # Payment history penalty
    payment_history = features['payment_history']
    payment_penalty = 0.5 if payment_history == "Late >60" else 1.0
    
    # Credit utilization penalty
    utilization_penalty = 0.8 if credit_utilization > 50 else 1.0
    
    # Calculate adjusted credit limit
    adjusted_limit = min(
        base_limit * dti_factor * employment_bonus * payment_penalty * utilization_penalty,
        credit_cap
    )
    
    # Use the lower of ML prediction or rule-based limit
    final_amount = min(amount, adjusted_limit)
    final_amount = min(final_amount, features['requested_amount'])
    
    # Ensure interest rate is within specified range (3-15%)
    final_interest_rate = max(3.0, min(15.0, interest_rate))
    
    return approval, final_amount, final_interest_rate

def evaluate_models(models, X_test_processed, data):
    """
    Evaluate the trained models on the test data
    
    Args:
        models: Dictionary containing the trained models
        X_test_processed: Preprocessed test features
        data: Dictionary containing test target variables
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_proba_approval = models['approval_model'].predict_proba(X_test_processed)[:, 1]
    
    # Use threshold of 0.5 for approval decision
    threshold = 0.5
    y_pred_approval_raw = (y_proba_approval >= threshold).astype(int)
    
    # Initialize arrays to store final predictions after business rules
    X_test_original = data['X_test']
    final_approvals = np.zeros_like(y_pred_approval_raw)
    final_amounts = np.zeros(len(y_pred_approval_raw))
    final_interest_rates = np.zeros(len(y_pred_approval_raw))
    
    # Apply business rules to each prediction
    for i, (idx, row) in enumerate(X_test_original.iterrows()):
        # Only predict amount and interest rate for initially approved loans
        if y_pred_approval_raw[i] == 1:
            amount = models['credit_limit_model'].predict(X_test_processed[i:i+1])[0]
            interest_rate = models['interest_rate_model'].predict(X_test_processed[i:i+1])[0]
        else:
            amount = 0.0
            interest_rate = 0.0
        
        # Apply business rules
        final_approval, final_amount, final_interest = apply_business_rules(
            y_pred_approval_raw[i], amount, interest_rate, row
        )
        
        # Store final predictions
        final_approvals[i] = final_approval
        final_amounts[i] = final_amount
        final_interest_rates[i] = final_interest
    
    # Calculate metrics using final predictions after business rules
    accuracy = accuracy_score(data['y_test_approval'], final_approvals)
    conf_matrix = confusion_matrix(data['y_test_approval'], final_approvals)
    
    # Calculate recall, precision, and F1 score for approvals (class 1)
    recall = recall_score(data['y_test_approval'], final_approvals)
    precision = precision_score(data['y_test_approval'], final_approvals)
    f1 = f1_score(data['y_test_approval'], final_approvals)
    
    # Calculate MAE for credit limit and interest rate
    # We need to match the indices of true and predicted values
    true_approved_indices = data['y_test_approval'] == 1
    pred_approved_indices = final_approvals == 1
    
    # Get the indices in the original test set that are approved in both true and predicted
    common_approved_indices = np.logical_and(true_approved_indices, pred_approved_indices)
    
    if np.sum(common_approved_indices) > 0:
        # Get the indices in the test set
        common_indices = data['test_indices'][common_approved_indices]
        
        # Get the true values for loans that are approved in both true and predicted
        true_credit_limit = data['y_test_credit_limit'][common_indices]
        true_interest = data['y_test_interest'][common_indices]
        
        # Get the predicted values for the same loans
        pred_credit_limit = final_amounts[common_approved_indices]
        pred_interest = final_interest_rates[common_approved_indices]
        
        # Calculate MAE
        mae_credit_limit = mean_absolute_error(true_credit_limit, pred_credit_limit)
        mae_interest = mean_absolute_error(true_interest, pred_interest)
    else:
        mae_credit_limit = None
        mae_interest = None
    
    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'mae_credit_limit': mae_credit_limit,
        'mae_interest': mae_interest,
        'y_pred_approval': final_approvals,
        'y_pred_credit_limit': final_amounts,
        'y_pred_interest': final_interest_rates,
        'true_approved_indices': true_approved_indices,
        'pred_approved_indices': pred_approved_indices,
        'common_approved_indices': common_approved_indices
    }

def visualize_results(metrics, data):
    """
    Create visualizations of the model evaluation results
    
    Args:
        metrics: Dictionary containing evaluation metrics
        data: Dictionary containing test data
        
    Returns:
        Matplotlib figure with visualizations
    """
    plt.figure(figsize=(16, 12))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Denied', 'Approved'], 
                yticklabels=['Denied', 'Approved'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Approval Distribution
    plt.subplot(2, 2, 2)
    labels = ['Denied', 'Approved']
    true_counts = [sum(data['y_test_approval'] == 0), sum(data['y_test_approval'] == 1)]
    pred_counts = [sum(metrics['y_pred_approval'] == 0), sum(metrics['y_pred_approval'] == 1)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True')
    plt.bar(x + width/2, pred_counts, width, label='Predicted')
    plt.xticks(x, labels)
    plt.title('Approval Distribution')
    plt.legend()
    
    # Amount Prediction Error
    if metrics['y_pred_credit_limit'] is not None and len(metrics['y_pred_credit_limit']) > 0:
        plt.subplot(2, 2, 3)
        common_approved_indices = metrics['common_approved_indices']
        if np.sum(common_approved_indices) > 0:
            true_credit_limit = data['y_test_credit_limit'][common_approved_indices]
            pred_credit_limit = metrics['y_pred_credit_limit'][:len(true_credit_limit)]
            error = pred_credit_limit - true_credit_limit
            plt.hist(error, bins=30, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Amount Prediction Error')
            plt.xlabel('Predicted - True Amount ($)')
            plt.ylabel('Count')
    
    # Rate Prediction Error
    if metrics['y_pred_interest'] is not None and len(metrics['y_pred_interest']) > 0:
        plt.subplot(2, 2, 4)
        common_approved_indices = metrics['common_approved_indices']
        if np.sum(common_approved_indices) > 0:
            true_interest = data['y_test_interest'][common_approved_indices]
            pred_interest = metrics['y_pred_interest'][:len(true_interest)]
            error = pred_interest - true_interest
            plt.hist(error, bins=30, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Rate Prediction Error')
            plt.xlabel('Predicted - True Rate (%)')
            plt.ylabel('Count')
    
    plt.tight_layout()
    return plt.gcf()

def print_evaluation_report(metrics):
    """
    Print a detailed evaluation report
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "="*50)
    print("LINE OF CREDIT MODEL EVALUATION REPORT")
    print("="*50)
    
    print("\nAPPROVAL PREDICTION:")
    print(f"Accuracy: {metrics['accuracy']:.4f} (Target: ≥ 0.85) - {'PASS' if metrics['accuracy'] >= 0.85 else 'FAIL'}")
    print(f"Recall (Approvals): {metrics['recall']:.4f} (Target: ≥ 0.90) - {'PASS' if metrics['recall'] >= 0.90 else 'FAIL'}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['conf_matrix'])
    
    if metrics['mae_credit_limit'] is not None:
        print("\nCREDIT LIMIT PREDICTION:")
        print(f"MAE: ${metrics['mae_credit_limit']:.2f} (Target: ≤ $2,000) - {'PASS' if metrics['mae_credit_limit'] <= 2000 else 'FAIL'}")
    
    if metrics['mae_interest'] is not None:
        print("\nINTEREST RATE PREDICTION:")
        print(f"MAE: {metrics['mae_interest']:.4f}% (Target: ≤ 1.0%) - {'PASS' if metrics['mae_interest'] <= 1.0 else 'FAIL'}")
    
    print("\nOVERALL EVALUATION:")
    overall_pass = (
        metrics['accuracy'] >= 0.85 and 
        metrics['recall'] >= 0.90 and 
        (metrics['mae_credit_limit'] is None or metrics['mae_credit_limit'] <= 2000) and 
        (metrics['mae_interest'] is None or metrics['mae_interest'] <= 1.0)
    )
    print(f"Status: {'PASS' if overall_pass else 'FAIL'}")
    print("="*50)

def save_visualization(fig, output_path=None):
    """
    Save the visualization to a file
    
    Args:
        fig: Matplotlib figure
        output_path: Path to save the figure
    """
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, 'loc_model_evaluation.png')
    
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

def main():
    """
    Main function to run the evaluation
    """
    # Generate fresh evaluation data
    # Set generate_new=True to generate a new dataset each time
    # Set num_samples to control the size of the generated dataset
    
    # Load models and artifacts
    models_artifacts = load_models_and_artifacts()
    
    # Load and preprocess data with fresh generation
    data = load_and_preprocess_data(generate_new=True, num_samples=1000)
    
    # Preprocess test data
    X_test_processed = preprocess_test_data(
        data['X_test'], 
        models_artifacts, 
        data['categorical_cols']
    )
    
    # Evaluate models
    metrics = evaluate_models(models_artifacts, X_test_processed, data)
    
    # Print evaluation report
    print_evaluation_report(metrics)
    
    # Create and save visualizations
    fig = visualize_results(metrics, data)
    save_visualization(fig)

if __name__ == "__main__":
    main()