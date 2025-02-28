import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, mean_absolute_error, r2_score

###############################
# 1. Data Loading and Preprocessing
###############################

# Load the dataset (replace 'loc_data.csv' with the actual file path or DataFrame)
df = pd.read_csv('loc_data.csv')

# Handle missing values:
# - For numerical columns: fill with median.
# - For categorical columns: fill with mode (most frequent).
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_cols:
    if not df[col].mode().empty:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Define target columns
target_cols = ['ApprovalStatus', 'CreditLimit', 'InterestRate']

# Define feature columns (exclude target columns)
features = [col for col in df.columns if col not in target_cols]

# One-hot encode categorical features in features only
df_encoded = pd.get_dummies(df[features], columns=[col for col in categorical_cols if col in features], drop_first=True)

# Standardize numerical features for better model performance
scaler = StandardScaler()
numeric_features = [col for col in df_encoded.columns if col not in df.select_dtypes(include=['object','category','bool']).columns]
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])

###############################
# 2. Approval Status Prediction (Classification)
###############################

# Prepare features (X) and target (y) for approval status
X = df_encoded.copy()
y = df['ApprovalStatus']

# Encode the approval status target if it's categorical (e.g., 'Yes'/'No' -> 1/0)
if y.dtype == 'object':
    y = y.map({'Yes': 1, 'No': 0})

# Split the dataset into training and testing sets for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the classifiers
log_clf = LogisticRegression(max_iter=1000, random_state=42)
dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)

# Train the models
log_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# Predict on the test set
y_pred_log = log_clf.predict(X_test)
y_pred_dt = dt_clf.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)

# Calculate accuracy for each model
acc_log = accuracy_score(y_test, y_pred_log)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Calculate AUC-ROC for each model (need probability estimates for AUC)
y_proba_log = log_clf.predict_proba(X_test)[:, 1]
y_proba_dt = dt_clf.predict_proba(X_test)[:, 1]
y_proba_rf = rf_clf.predict_proba(X_test)[:, 1]
auc_log = roc_auc_score(y_test, y_proba_log)
auc_dt = roc_auc_score(y_test, y_proba_dt)
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Print evaluation metrics
print("----- Approval Status Prediction (Classification) -----")
print(f"Logistic Regression Accuracy: {acc_log:.3f}, AUC: {auc_log:.3f}")
print(f"Decision Tree Accuracy: {acc_dt:.3f}, AUC: {auc_dt:.3f}")
print(f"Random Forest Accuracy: {acc_rf:.3f}, AUC: {auc_rf:.3f}")
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


###############################
# 3. Credit Limit Prediction (Regression)
###############################

# Filter the dataset to include only approved applicants
approved_df = df[df['ApprovalStatus'] == 1].copy()

# Define features and target for credit limit prediction
X_limit = approved_df[features]  # use same feature columns as before
y_limit = approved_df['CreditLimit']

# Split into train and test sets
X_train_lim, X_test_lim, y_train_lim, y_test_lim = train_test_split(X_limit, y_limit, test_size=0.2, random_state=42)

# Handle missing values for the subset (if any) and scale numeric features
for col in numeric_cols:
    median_val = X_train_lim[col].median()
    X_train_lim[col].fillna(median_val, inplace=True)
    X_test_lim[col].fillna(median_val, inplace=True)
for col in categorical_cols:
    if col in X_train_lim.columns and not X_train_lim[col].mode().empty:
        mode_val = X_train_lim[col].mode()[0]
        X_train_lim[col].fillna(mode_val, inplace=True)
        X_test_lim[col].fillna(mode_val, inplace=True)

X_train_lim[numeric_cols] = scaler.fit_transform(X_train_lim[numeric_cols])
X_test_lim[numeric_cols] = scaler.transform(X_test_lim[numeric_cols])

# One-hot encode categoricals for train and test, ensuring same columns
X_train_lim = pd.get_dummies(X_train_lim, columns=[col for col in categorical_cols if col in X_train_lim.columns], drop_first=True)
X_test_lim = pd.get_dummies(X_test_lim, columns=[col for col in categorical_cols if col in X_test_lim.columns], drop_first=True)
X_test_lim = X_test_lim.reindex(columns=X_train_lim.columns, fill_value=0)

# Initialize regression models for credit limit
lin_reg = LinearRegression()
dt_reg = DecisionTreeRegressor(random_state=42)
rf_reg = RandomForestRegressor(random_state=42)

# Train the models on the approved applicants' training data
lin_reg.fit(X_train_lim, y_train_lim)
dt_reg.fit(X_train_lim, y_train_lim)
rf_reg.fit(X_train_lim, y_train_lim)

# Predict credit limit on test set
y_pred_lin = lin_reg.predict(X_test_lim)
y_pred_dt = dt_reg.predict(X_test_lim)
y_pred_rf = rf_reg.predict(X_test_lim)

# Evaluate the models
mae_lin = mean_absolute_error(y_test_lim, y_pred_lin)
mae_dt  = mean_absolute_error(y_test_lim, y_pred_dt)
mae_rf  = mean_absolute_error(y_test_lim, y_pred_rf)
r2_lin  = r2_score(y_test_lim, y_pred_lin)
r2_dt   = r2_score(y_test_lim, y_pred_dt)
r2_rf   = r2_score(y_test_lim, y_pred_rf)

print("\n----- Credit Limit Prediction (Regression) -----")
print(f"Linear Regression MAE: ${mae_lin:.2f}, R^2: {r2_lin:.3f}")
print(f"Decision Tree Regressor MAE: ${mae_dt:.2f}, R^2: {r2_dt:.3f}")
print(f"Random Forest Regressor MAE: ${mae_rf:.2f}, R^2: {r2_rf:.3f}")

###############################
# 4. Interest Rate Prediction (Regression)
###############################

# Define features and target for interest rate prediction (approved applicants only)
X_interest = approved_df[features]  # reuse the filtered approved applicants
y_interest = approved_df['InterestRate']

# Split into train and test sets
X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_interest, y_interest, test_size=0.2, random_state=42)

# Handle missing values and scaling for interest rate prediction
for col in numeric_cols:
    median_val = X_train_int[col].median()
    X_train_int[col].fillna(median_val, inplace=True)
    X_test_int[col].fillna(median_val, inplace=True)
for col in categorical_cols:
    if col in X_train_int.columns and not X_train_int[col].mode().empty:
        mode_val = X_train_int[col].mode()[0]
        X_train_int[col].fillna(mode_val, inplace=True)
        X_test_int[col].fillna(mode_val, inplace=True)

X_train_int[numeric_cols] = scaler.fit_transform(X_train_int[numeric_cols])
X_test_int[numeric_cols] = scaler.transform(X_test_int[numeric_cols])

# One-hot encode categorical features, aligning train and test columns
X_train_int = pd.get_dummies(X_train_int, columns=[col for col in categorical_cols if col in X_train_int.columns], drop_first=True)
X_test_int = pd.get_dummies(X_test_int, columns=[col for col in categorical_cols if col in X_test_int.columns], drop_first=True)
X_test_int = X_test_int.reindex(columns=X_train_int.columns, fill_value=0)

# Initialize regression models for interest rate
lin_reg_int = LinearRegression()
dt_reg_int = DecisionTreeRegressor(random_state=42)
rf_reg_int = RandomForestRegressor(random_state=42)

# Train the models
lin_reg_int.fit(X_train_int, y_train_int)
dt_reg_int.fit(X_train_int, y_train_int)
rf_reg_int.fit(X_train_int, y_train_int)

# Predict interest rate on the test set
y_pred_lin_int = lin_reg_int.predict(X_test_int)
y_pred_dt_int = dt_reg_int.predict(X_test_int)
y_pred_rf_int = rf_reg_int.predict(X_test_int)

# Evaluate the models
mae_lin_int = mean_absolute_error(y_test_int, y_pred_lin_int)
mae_dt_int  = mean_absolute_error(y_test_int, y_pred_dt_int)
mae_rf_int  = mean_absolute_error(y_test_int, y_pred_rf_int)
r2_lin_int  = r2_score(y_test_int, y_pred_lin_int)
r2_dt_int   = r2_score(y_test_int, y_pred_dt_int)
r2_rf_int   = r2_score(y_test_int, y_pred_rf_int)

print("\n----- Interest Rate Prediction (Regression) -----")
print(f"Linear Regression MAE: {mae_lin_int:.3f}%, R^2: {r2_lin_int:.3f}")
print(f"Decision Tree Regressor MAE: {mae_dt_int:.3f}%, R^2: {r2_dt_int:.3f}")
print(f"Random Forest Regressor MAE: {mae_rf_int:.3f}%, R^2: {r2_rf_int:.3f}")
