import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_loc_model(model, X_test, y_test, canadian_norms=True):
    """
    Evaluate a Line of Credit ML model against specified accuracy targets.
    
    Args:
        model: Trained ML model with predict and predict_proba methods
        X_test: Test features DataFrame
        y_test: Dictionary or DataFrame containing true values for:
                - 'approved': binary approval status (0/1)
                - 'approved_amount': numeric credit limit amount
                - 'interest_rate': numeric interest rate percentage
        canadian_norms: Whether to check alignment with Canadian financial norms
    
    Returns:
        results_dict: Dictionary with evaluation metrics and pass/fail status
    """
    # Get predictions
    y_pred_approved = model.predict(X_test)
    
    # For regression targets, we need to transform the predictions
    # This assumes a multi-output model that returns all three targets
    # For models that only predict one target, this should be adjusted
    try:
        # Try to get all predictions at once (multi-output model)
        y_pred_all = model.predict(X_test)
        y_pred_approved = y_pred_all[:, 0].astype(int)
        y_pred_amount = y_pred_all[:, 1]
        y_pred_rate = y_pred_all[:, 2]
    except:
        # If that fails, try individual predictions
        try:
            y_pred_approved = model.predict_approved(X_test).astype(int)
        except:
            y_pred_approved = model.predict(X_test).astype(int)
        
        try:
            y_pred_amount = model.predict_amount(X_test)
        except:
            # Skip if not available
            y_pred_amount = None
            
        try:
            y_pred_rate = model.predict_rate(X_test)
        except:
            # Skip if not available
            y_pred_rate = None
    
    # Extract true values (handle both dictionary and DataFrame formats)
    if isinstance(y_test, dict):
        y_true_approved = y_test['approved']
        y_true_amount = y_test.get('approved_amount')
        y_true_rate = y_test.get('interest_rate')
    else:  # Assume DataFrame
        y_true_approved = y_test['approved'].values
        y_true_amount = y_test.get('approved_amount', pd.Series()).values
        y_true_rate = y_test.get('interest_rate', pd.Series()).values
    
    # Initialize results dictionary
    results = {
        'approval': {
            'target': 0.85,  # 85% accuracy
            'value': None,
            'pass': False
        },
        'amount': {
            'target': 2000,  # MAE ≤ $2K
            'value': None,
            'pass': False
        },
        'rate': {
            'target': 1.0,  # MAE ≤ 1%
            'value': None,
            'pass': False
        },
        'canadian_norms': {
            'pass': False,
            'details': {}
        },
        'overall_pass': False
    }
    
    # Evaluate approval accuracy
    accuracy = accuracy_score(y_true_approved, y_pred_approved)
    results['approval']['value'] = accuracy
    results['approval']['pass'] = accuracy >= results['approval']['target']
    
    # Detailed classification metrics
    conf_matrix = confusion_matrix(y_true_approved, y_pred_approved)
    class_report = classification_report(y_true_approved, y_pred_approved, output_dict=True)
    
    # Calculate recall for approvals (class 1)
    approval_recall = class_report['1']['recall']
    results['approval']['recall'] = approval_recall
    results['approval']['recall_target'] = 0.90  # 90% recall
    results['approval']['recall_pass'] = approval_recall >= 0.90
    
    # Evaluate amount prediction (only for approved applications)
    if y_pred_amount is not None and len(y_true_amount) > 0:
        # Filter to only look at approved applications
        approved_indices = y_true_approved == 1
        if sum(approved_indices) > 0:
            mae_amount = mean_absolute_error(
                y_true_amount[approved_indices], 
                y_pred_amount[approved_indices]
            )
            results['amount']['value'] = mae_amount
            results['amount']['pass'] = mae_amount <= results['amount']['target']
    
    # Evaluate interest rate prediction (only for approved applications)
    if y_pred_rate is not None and len(y_true_rate) > 0:
        # Filter to only look at approved applications
        approved_indices = y_true_approved == 1
        if sum(approved_indices) > 0:
            mae_rate = mean_absolute_error(
                y_true_rate[approved_indices], 
                y_pred_rate[approved_indices]
            )
            results['rate']['value'] = mae_rate
            results['rate']['pass'] = mae_rate <= results['rate']['target']
    
    # Check Canadian financial norms (if requested and if we have the necessary data)
    if canadian_norms and hasattr(X_test, 'columns'):
        # Calculate DTI (Debt-to-Income ratio) for approved applications
        if all(col in X_test.columns for col in ['annual_income', 'self_reported_debt']) and 'estimated_debt' in X_test.columns:
            # Calculate monthly income
            monthly_income = X_test['annual_income'] / 12
            
            # Calculate total debt
            total_debt = X_test['self_reported_debt']
            if 'estimated_debt' in X_test.columns:
                total_debt += X_test['estimated_debt']
            
            # Calculate DTI
            dti = (total_debt / monthly_income) * 100
            
            # Check if most approved applications have DTI ≤ 40%
            approved_dti = dti[y_pred_approved == 1]
            pct_approved_under_40 = (approved_dti <= 40).mean() * 100
            
            results['canadian_norms']['details']['dti_under_40_pct'] = pct_approved_under_40
            results['canadian_norms']['pass'] = pct_approved_under_40 >= 60  # At least 60% comply
    
    # Determine overall pass status
    results['overall_pass'] = (
        results['approval']['pass'] and 
        (results['amount']['pass'] if results['amount']['value'] is not None else True) and
        (results['rate']['pass'] if results['rate']['value'] is not None else True) and
        (results['canadian_norms']['pass'] if canadian_norms else True)
    )
    
    # Create visualizations
    plt.figure(figsize=(16, 12))
    
    # Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Denied', 'Approved'], 
                yticklabels=['Denied', 'Approved'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Approval Distribution
    plt.subplot(2, 2, 2)
    labels = ['Denied', 'Approved']
    true_counts = [sum(y_true_approved == 0), sum(y_true_approved == 1)]
    pred_counts = [sum(y_pred_approved == 0), sum(y_pred_approved == 1)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True')
    plt.bar(x + width/2, pred_counts, width, label='Predicted')
    plt.xticks(x, labels)
    plt.title('Approval Distribution')
    plt.legend()
    
    # Amount Prediction Error (if available)
    if y_pred_amount is not None and len(y_true_amount) > 0:
        plt.subplot(2, 2, 3)
        approved_indices = y_true_approved == 1
        if sum(approved_indices) > 0:
            error = y_pred_amount[approved_indices] - y_true_amount[approved_indices]
            plt.hist(error, bins=30, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Amount Prediction Error')
            plt.xlabel('Predicted - True Amount ($)')
            plt.ylabel('Count')
    
    # Rate Prediction Error (if available)
    if y_pred_rate is not None and len(y_true_rate) > 0:
        plt.subplot(2, 2, 4)
        approved_indices = y_true_approved == 1
        if sum(approved_indices) > 0:
            error = y_pred_rate[approved_indices] - y_true_rate[approved_indices]
            plt.hist(error, bins=30, alpha=0.7)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title('Rate Prediction Error')
            plt.xlabel('Predicted - True Rate (%)')
            plt.ylabel('Count')
    
    plt.tight_layout()
    
    # Generate detailed report
    print("\n" + "="*50)
    print("LINE OF CREDIT MODEL EVALUATION REPORT")
    print("="*50)
    
    print("\nAPPROVAL PREDICTION:")
    print(f"Accuracy: {accuracy:.4f} (Target: ≥ {results['approval']['target']:.2f}) - {'PASS' if results['approval']['pass'] else 'FAIL'}")
    print(f"Recall (Approvals): {approval_recall:.4f} (Target: ≥ {results['approval']['recall_target']:.2f}) - {'PASS' if results['approval']['recall_pass'] else 'FAIL'}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    if y_pred_amount is not None and results['amount']['value'] is not None:
        print("\nCREDIT LIMIT PREDICTION:")
        print(f"MAE: ${results['amount']['value']:.2f} (Target: ≤ ${results['amount']['target']}) - {'PASS' if results['amount']['pass'] else 'FAIL'}")
    
    if y_pred_rate is not None and results['rate']['value'] is not None:
        print("\nINTEREST RATE PREDICTION:")
        print(f"MAE: {results['rate']['value']:.4f}% (Target: ≤ {results['rate']['target']}%) - {'PASS' if results['rate']['pass'] else 'FAIL'}")
    
    if canadian_norms and 'dti_under_40_pct' in results['canadian_norms']['details']:
        print("\nCANADIAN FINANCIAL NORMS:")
        print(f"% of Approved with DTI ≤ 40%: {results['canadian_norms']['details']['dti_under_40_pct']:.2f}% - {'PASS' if results['canadian_norms']['pass'] else 'FAIL'}")
    
    print("\nOVERALL EVALUATION:")
    print(f"Status: {'PASS' if results['overall_pass'] else 'FAIL'}")
    print("="*50)
    
    return results, plt.gcf()


def run_evaluation_example():
    """
    Example usage of the evaluation function with a dummy model
    """
    # Create a dummy model class for demonstration
    class DummyLOCModel:
        def predict(self, X):
            # Return 3 columns: approval (0/1), amount, rate
            n_samples = len(X)
            approvals = np.random.binomial(1, 0.7, n_samples)  # 70% approval rate
            
            # Generate reasonable amounts and rates
            amounts = np.zeros(n_samples)
            rates = np.zeros(n_samples)
            
            for i in range(n_samples):
                if approvals[i] == 1:
                    # Amounts between $1,000 and $25,000
                    amounts[i] = np.random.uniform(1000, 25000)
                    # Rates between 3% and 15%
                    rates[i] = np.random.uniform(3, 15)
            
            return np.column_stack((approvals, amounts, rates))
    
    # Create dummy test data
    np.random.seed(42)  # For reproducibility
    n_samples = 1000
    
    # Create feature data
    X_test = pd.DataFrame({
        'annual_income': np.random.lognormal(mean=11, sigma=0.5, size=n_samples),
        'self_reported_debt': np.random.uniform(500, 5000, n_samples),
        'self_reported_expenses': np.random.uniform(500, 5000, n_samples),
        'requested_amount': np.random.uniform(1000, 50000, n_samples),
        'credit_score': np.random.normal(680, 100, n_samples).clip(300, 900),
        'credit_utilization': np.random.beta(2, 5, n_samples) * 100,
        'estimated_debt': np.random.uniform(100, 1000, n_samples)
    })
    
    # Create target data
    y_true_approved = np.random.binomial(1, 0.65, n_samples)  # 65% approval rate
    
    y_true_amount = np.zeros(n_samples)
    y_true_rate = np.zeros(n_samples)
    
    for i in range(n_samples):
        if y_true_approved[i] == 1:
            # Set amounts based on income
            monthly_income = X_test['annual_income'].iloc[i] / 12
            base_limit = min(monthly_income * 5, 25000)
            
            # Adjust based on credit score
            credit_score = X_test['credit_score'].iloc[i]
            if credit_score >= 750:
                credit_factor = 1.0
            elif credit_score >= 660:
                credit_factor = 0.8
            else:
                credit_factor = 0.6
                
            y_true_amount[i] = base_limit * credit_factor
            
            # Set rates based on credit score
            if credit_score >= 750:
                y_true_rate[i] = np.random.uniform(3, 6)
            elif credit_score >= 660:
                y_true_rate[i] = np.random.uniform(5, 9)
            else:
                y_true_rate[i] = np.random.uniform(8, 15)
    
    # Create y_test dictionary
    y_test = {
        'approved': y_true_approved,
        'approved_amount': y_true_amount,
        'interest_rate': y_true_rate
    }
    
    # Create and evaluate the model
    model = DummyLOCModel()
    results, fig = evaluate_loc_model(model, X_test, y_test)
    
    # Display the results
    print("\nEvaluation Results Summary:")
    print(f"Approval Accuracy: {results['approval']['value']:.4f} - {'PASS' if results['approval']['pass'] else 'FAIL'}")
    if results['amount']['value'] is not None:
        print(f"Amount MAE: ${results['amount']['value']:.2f} - {'PASS' if results['amount']['pass'] else 'FAIL'}")
    if results['rate']['value'] is not None:
        print(f"Rate MAE: {results['rate']['value']:.4f}% - {'PASS' if results['rate']['pass'] else 'FAIL'}")
    print(f"Canadian Norms: {'PASS' if results['canadian_norms']['pass'] else 'FAIL'}")
    print(f"Overall: {'PASS' if results['overall_pass'] else 'FAIL'}")
    
    # Save the figure
    fig.savefig('loc_model_evaluation.png')
    print("\nEvaluation plots saved to 'loc_model_evaluation.png'")
    
    return results


if __name__ == "__main__":
    # Run the example
    run_evaluation_example()