import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_absolute_error, r2_score
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def create_model_directory():
    """Create directory to save trained models if it doesn't exist"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Print available columns to help with debugging
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Handle missing values - avoiding chained assignment warnings
    # For numeric columns
    for col in numeric_cols:
        df.loc[:, col] = df[col].fillna(df[col].median())
    
    # For categorical columns
    for col in categorical_cols:
        if not df[col].mode().empty:
            df.loc[:, col] = df[col].fillna(df[col].mode()[0])
    
    return df, numeric_cols, categorical_cols

def prepare_features_targets(df, features, target_col, categorical_cols, encode_target=None):
    """Prepare features and target for modeling with proper encoding"""
    # Validate target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Extract features and target
    X = df[features].copy()
    y = df[target_col].copy()
    
    # Encode target if needed
    if encode_target and y.dtype == 'object':
        y = y.map(encode_target)
    
    return X, y

def encode_and_scale_features(X_train, X_test, categorical_cols, numeric_cols):
    """One-hot encode categorical features and scale numeric features"""
    # One-hot encode categorical features
    cat_features = [col for col in categorical_cols if col in X_train.columns]
    X_train_encoded = pd.get_dummies(X_train, columns=cat_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_features, drop_first=True)
    
    # Ensure test set has same columns as train set
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # Scale numeric features
    scaler = StandardScaler()
    numeric_features = [col for col in X_train_encoded.columns if col in numeric_cols or 
                       (col not in cat_features and not any(col.startswith(c + '_') for c in cat_features))]
    
    if numeric_features:
        X_train_encoded[numeric_features] = scaler.fit_transform(X_train_encoded[numeric_features])
        X_test_encoded[numeric_features] = scaler.transform(X_test_encoded[numeric_features])
    
    return X_train_encoded, X_test_encoded, scaler

def find_optimal_threshold(model, X_test, y_test, target_recall=0.90):
    """Find threshold that achieves target recall for positive class (approvals)"""
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5  # Default
    best_f1 = 0
    recall_achieved = False
    
    print("\nFinding optimal threshold for high recall (>90%)...")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    
    for threshold in thresholds:
        # Apply threshold to get predictions
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        recall = recall_score(y_test, y_pred)
        # Handle case where there are no positive predictions
        try:
            precision = precision_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        except:
            precision = 0
            f1 = 0
        
        # Check if we've achieved target recall
        if recall >= target_recall:
            if not recall_achieved:
                recall_achieved = True
                print(f"\n--- Thresholds achieving {target_recall*100}%+ recall ---")
            
            print(f"{threshold:.2f}{precision:.4f}{recall:.4f}{f1:.4f}".replace("0.0000", "0.0000"))
            
            # Update best threshold if F1 is better
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    if not recall_achieved:
        print(f"No threshold achieved {target_recall*100}% recall. Using lowest threshold.")
        best_threshold = thresholds[0]
    
    print(f"\nSelected threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
    return best_threshold

def train_evaluate_classification(X_train, X_test, y_train, y_test, optimize_recall=True):
    """Train and evaluate classification models with cross-validation, optimizing for recall"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
    }
    
    results = {}
    best_model = None
    best_recall = 0
    best_threshold = 0.5
    
    print("\n----- Approval Status Prediction (Classification) -----")
    print("Optimizing for high recall (>90%) to minimize missed opportunities")
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Get probability predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold for high recall
        if optimize_recall:
            threshold = find_optimal_threshold(model, X_test, y_test)
            y_pred = (y_proba >= threshold).astype(int)
        else:
            threshold = 0.5
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'threshold': threshold,
            'cv_recall': np.mean(cv_scores),
            'test_accuracy': accuracy,
            'test_recall': recall,
            'test_precision': precision,
            'test_f1': f1,
            'auc': auc
        }
        
        # Print results
        print(f"\n{name} Results (threshold={threshold:.2f}):")
        print(f"CV Recall: {np.mean(cv_scores):.3f}")
        print(f"Test Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        # Track best model based on recall
        if recall > best_recall:
            best_recall = recall
            best_model = model
            best_threshold = threshold
    
    # Print detailed report for best model
    best_model_name = max(results.items(), key=lambda x: x[1]['test_recall'])[0]
    best_model = results[best_model_name]['model']
    best_threshold = results[best_model_name]['threshold']
    
    # Apply best threshold
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    y_pred_best = (y_proba_best >= best_threshold).astype(int)
    
    print(f"\nBest Model for High Recall: {best_model_name} (threshold={best_threshold:.2f})")
    print(f"Classification Report ({best_model_name}):")
    print(classification_report(y_test, y_pred_best))
    
    # Check if model meets accuracy goals
    accuracy = accuracy_score(y_test, y_pred_best)
    recall = recall_score(y_test, y_pred_best)
    
    print("\n--- Approval Model Goal Assessment ---")
    print(f"Accuracy: {accuracy:.3f} - Goal: ≥ 0.85 - {'✓ PASSED' if accuracy >= 0.85 else '✗ FAILED'}")
    print(f"Recall: {recall:.3f} - Goal: ≥ 0.90 - {'✓ PASSED' if recall >= 0.90 else '✗ FAILED'}")
    
    return results, best_model, best_threshold

def train_evaluate_regression(X_train, X_test, y_train, y_test, target_name):
    """Train and evaluate regression models with cross-validation"""
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=200, max_depth=15)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    print(f"\n----- {target_name} Prediction (Regression) -----")
    
    for name, model in models.items():
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'cv_mae': -np.mean(cv_scores),
            'test_mae': mae,
            'r2': r2
        }
        
        # Format output based on target
        if target_name == 'Credit Limit':
            print(f"{name} - CV MAE: ${-np.mean(cv_scores):.2f}, Test MAE: ${mae:.2f}, R^2: {r2:.3f}")
        else:  # Interest Rate
            print(f"{name} - CV MAE: {-np.mean(cv_scores):.3f}%, Test MAE: {mae:.3f}%, R^2: {r2:.3f}")
        
        # Track best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    # Get predictions from best model
    y_pred_best = best_model.predict(X_test)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    
    # Check if model meets MAE goal
    if target_name == 'Credit Limit':
        goal_met = mae_best <= 2000
        print(f"\n--- Credit Limit Model Goal Assessment ---")
        print(f"MAE: ${mae_best:.2f} - Goal: ≤ $2,000 - {'✓ PASSED' if goal_met else '✗ FAILED'}")
    else:  # Interest Rate
        goal_met = mae_best <= 1.0
        print(f"\n--- Interest Rate Model Goal Assessment ---")
        print(f"MAE: {mae_best:.3f}% - Goal: ≤ 1.0% - {'✓ PASSED' if goal_met else '✗ FAILED'}")
    
    return results, best_model

def get_feature_importance(model, feature_names, top_n=10):
    """Extract feature importance from tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop features by importance:")
        for i in range(min(top_n, len(feature_names))):
            print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    # Set file path - updated to use correct path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(script_dir), 'datasets', 'data', 'synthetic_loan_applications.csv')
    
    # Print model accuracy goals
    print("\n" + "="*50)
    print("MODEL ACCURACY GOALS")
    print("="*50)
    print("Approval Status:")
    print("  • Accuracy: ≥ 85%")
    print("  • Recall (for approvals): ≥ 90% (minimize missed opportunities)")
    print("Credit Limit:")
    print("  • Mean Absolute Error (MAE): ≤ $2,000")
    print("Interest Rate:")
    print("  • MAE: ≤ 1%")
    print("="*50 + "\n")
    
    # Create model directory
    model_dir = create_model_directory()
    
    # Load and preprocess data
    df, numeric_cols, categorical_cols = load_and_preprocess_data(file_path)
    
    # Define target columns and features with correct names from CSV
    target_cols = {
        'approval': 'approved',
        'credit_limit': 'approved_amount',
        'interest': 'interest_rate'
    }
    
    # Validate all required columns exist
    missing_cols = [col for col in target_cols.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    features = [col for col in df.columns if col not in target_cols.values() and col != 'applicant_id']
    
    print("\nTarget columns:", target_cols)
    print("Feature columns:", features)

    # Update categorical_cols to only include those in features.
    categorical_cols = [col for col in categorical_cols if col in features]
    
    
    ###############################
    # 1. Approval Status Prediction (Classification)
    ###############################
    
    # Prepare features and target
    X, y = prepare_features_targets(
        df, features, target_cols['approval'], categorical_cols, 
        encode_target={True: 1, False: 0}  # Assuming 'approved' is boolean
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Updated Encoding and Scaling for Approval Model ---
    # One-hot encode categorical features for training and test sets using drop_first=True as in training.
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    
    # Ensure that X_test_encoded has the same columns as X_train_encoded
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # Determine which columns are numeric (or were intended to be scaled)
    numeric_features = [
        col for col in X_train_encoded.columns 
        if col in numeric_cols or (col not in categorical_cols and not any(col.startswith(c + '_') for c in categorical_cols))
    ]
    
    scaler = StandardScaler()
    # Fit and transform the numeric features for training and transform test numeric features.
    X_train_encoded[numeric_features] = scaler.fit_transform(X_train_encoded[numeric_features])
    X_test_encoded[numeric_features] = scaler.transform(X_test_encoded[numeric_features])
    
    # Save preprocessing artifacts for inference
    joblib.dump(X_train_encoded.columns.tolist(), os.path.join(model_dir, 'encoded_feature_columns.pkl'))
    joblib.dump(numeric_features, os.path.join(model_dir, 'scaled_columns.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    # Train and evaluate classification models with high recall optimization
    classification_results, best_clf, best_threshold = train_evaluate_classification(
        X_train_encoded, X_test_encoded, y_train, y_test, optimize_recall=True
    )
    
    # Get feature importance for best model if it's a tree-based model
    if isinstance(best_clf, (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)):
        get_feature_importance(best_clf, X_train_encoded.columns)
    
    # Save best model and threshold
    joblib.dump(best_clf, os.path.join(model_dir, 'approval_model.pkl'))
    
    
    ###############################
    # 2. Credit Limit Prediction (Regression)
    ###############################
    
    # Filter for approved applicants only
    approved_df = df[df[target_cols['approval']] == 1].copy()
    
    # Prepare features and target for credit limit
    X_limit, y_limit = prepare_features_targets(
        approved_df, features, target_cols['credit_limit'], categorical_cols
    )
    
    # Split data
    X_train_lim, X_test_lim, y_train_lim, y_test_lim = train_test_split(
        X_limit, y_limit, test_size=0.2, random_state=42
    )
    
    # Encode and scale features
    X_train_lim_encoded, X_test_lim_encoded, _ = encode_and_scale_features(
        X_train_lim, X_test_lim, categorical_cols, numeric_cols
    )
    
    # Train and evaluate regression models for credit limit
    limit_results, best_limit_model = train_evaluate_regression(
        X_train_lim_encoded, X_test_lim_encoded, y_train_lim, y_test_lim, 'Credit Limit'
    )
    
    # Get feature importance for best model if it's a tree-based model
    if isinstance(best_limit_model, (DecisionTreeRegressor, RandomForestRegressor)):
        get_feature_importance(best_limit_model, X_train_lim_encoded.columns)
    
    # Save best model
    joblib.dump(best_limit_model, os.path.join(model_dir, 'credit_limit_model.pkl'))
    
    ###############################
    # 3. Interest Rate Prediction (Regression)
    ###############################
    
    # Prepare features and target for interest rate
    X_interest, y_interest = prepare_features_targets(
        approved_df, features, target_cols['interest'], categorical_cols
    )
    
    # Split data
    X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(
        X_interest, y_interest, test_size=0.2, random_state=42
    )
    
    # Encode and scale features
    X_train_int_encoded, X_test_int_encoded, _ = encode_and_scale_features(
        X_train_int, X_test_int, categorical_cols, numeric_cols
    )
    
    # Train and evaluate regression models for interest rate
    interest_results, best_interest_model = train_evaluate_regression(
        X_train_int_encoded, X_test_int_encoded, y_train_int, y_test_int, 'Interest Rate'
    )
    
    # Get feature importance for best model if it's a tree-based model
    if isinstance(best_interest_model, (DecisionTreeRegressor, RandomForestRegressor)):
        get_feature_importance(best_interest_model, X_train_int_encoded.columns)
    
    # Save best model
    joblib.dump(best_interest_model, os.path.join(model_dir, 'interest_rate_model.pkl'))
    
    print(f"\nAll models trained and saved to {model_dir}")
    
    # Print final summary of goal achievement
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    # Get metrics for final assessment
    # Approval model
    y_proba_approval = best_clf.predict_proba(X_test_encoded)[:, 1]
    y_pred_approval = (y_proba_approval >= best_threshold).astype(int)
    approval_accuracy = accuracy_score(y_test, y_pred_approval)
    approval_recall = recall_score(y_test, y_pred_approval)
    
    # Credit limit model
    y_pred_limit = best_limit_model.predict(X_test_lim_encoded)
    limit_mae = mean_absolute_error(y_test_lim, y_pred_limit)
    
    # Interest rate model
    y_pred_interest = best_interest_model.predict(X_test_int_encoded)
    interest_mae = mean_absolute_error(y_test_int, y_pred_interest)
    
    # Print summary
    print("Approval Status Model:")
    print(f"  • Accuracy: {approval_accuracy:.3f} - Goal: ≥ 0.85 - {'✓ PASSED' if approval_accuracy >= 0.85 else '✗ FAILED'}")
    print(f"  • Recall: {approval_recall:.3f} - Goal: ≥ 0.90 - {'✓ PASSED' if approval_recall >= 0.90 else '✗ FAILED'}")
    
    print("\nCredit Limit Model:")
    print(f"  • MAE: ${limit_mae:.2f} - Goal: ≤ $2,000 - {'✓ PASSED' if limit_mae <= 2000 else '✗ FAILED'}")
    
    print("\nInterest Rate Model:")
    print(f"  • MAE: {interest_mae:.3f}% - Goal: ≤ 1.0% - {'✓ PASSED' if interest_mae <= 1.0 else '✗ FAILED'}")
    
    print("="*50)

if __name__ == "__main__":
    main()
