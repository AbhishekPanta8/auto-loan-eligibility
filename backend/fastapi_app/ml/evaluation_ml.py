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
        import synthetic_data
        
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
    y_pred_approval = (y_proba_approval >= threshold).astype(int)
    
    # Get credit limit and interest rate predictions for approved loans
    approved_indices = y_pred_approval == 1
    X_test_approved = X_test_processed[approved_indices]
    
    if len(X_test_approved) > 0:
        y_pred_credit_limit = models['credit_limit_model'].predict(X_test_approved)
        y_pred_interest = models['interest_rate_model'].predict(X_test_approved)
    else:
        y_pred_credit_limit = np.array([])
        y_pred_interest = np.array([])
    
    # Calculate metrics
    accuracy = accuracy_score(data['y_test_approval'], y_pred_approval)
    conf_matrix = confusion_matrix(data['y_test_approval'], y_pred_approval)
    
    # Calculate recall, precision, and F1 score for approvals (class 1)
    recall = recall_score(data['y_test_approval'], y_pred_approval)
    precision = precision_score(data['y_test_approval'], y_pred_approval)
    f1 = f1_score(data['y_test_approval'], y_pred_approval)
    
    # Calculate MAE for credit limit and interest rate
    # We need to match the indices of true and predicted values
    true_approved_indices = data['y_test_approval'] == 1
    
    # Get the indices in the original test set that are approved in both true and predicted
    common_approved_indices = np.logical_and(true_approved_indices, approved_indices)
    
    if np.sum(common_approved_indices) > 0:
        # Get the true values for loans that are approved in both true and predicted
        true_credit_limit = data['y_test_credit_limit'][common_approved_indices]
        true_interest = data['y_test_interest'][common_approved_indices]
        
        # Get the predicted values for the same loans
        pred_credit_limit = y_pred_credit_limit[:len(true_credit_limit)]
        pred_interest = y_pred_interest[:len(true_interest)]
        
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
        'y_pred_approval': y_pred_approval,
        'y_pred_credit_limit': y_pred_credit_limit if len(X_test_approved) > 0 else None,
        'y_pred_interest': y_pred_interest if len(X_test_approved) > 0 else None,
        'true_approved_indices': true_approved_indices,
        'pred_approved_indices': approved_indices,
        'common_approved_indices': np.logical_and(true_approved_indices, approved_indices) if len(X_test_approved) > 0 else None
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