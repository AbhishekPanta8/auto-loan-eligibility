import shap
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, Any, List

class ExplanationService:
    def __init__(self, model_path: str, scaler_path: str, feature_columns_path: str):
        """
        Initialize the explanation service with model and preprocessing artifacts
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(feature_columns_path)
        
        # Initialize SHAP explainer
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = None
            
    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess the input data similar to how training data was processed
        """
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Store original values for explanation
        self.original_values = input_data.copy()
        
        # Handle Province enum
        if 'province' in df.columns and hasattr(df['province'].iloc[0], 'value'):
            df['province'] = df['province'].apply(lambda x: x.value if hasattr(x, 'value') else x)
            
        # Identify categorical columns (those that are not numeric)
        categorical_cols = []
        payment_history_col = None
        if 'employment_status' in df.columns:
            categorical_cols.append('employment_status')
        if 'payment_history' in df.columns:
            payment_history_col = 'payment_history'
        if 'province' in df.columns:
            categorical_cols.append('province')
        
        # One-hot encode categorical features, but handle payment_history separately
        if categorical_cols:
            df_encoded = pd.get_dummies(df[categorical_cols], columns=categorical_cols, drop_first=True)
        else:
            df_encoded = pd.DataFrame(index=df.index)
            
        # Add payment history encoding without dropping first
        if payment_history_col:
            payment_history_encoded = pd.get_dummies(df[payment_history_col], prefix='payment_history', drop_first=False)
            df_encoded = pd.concat([df_encoded, payment_history_encoded], axis=1)
            
        # Add non-categorical columns
        for col in df.columns:
            if col not in categorical_cols and col != payment_history_col:
                df_encoded[col] = df[col]
        
        # Ensure all expected columns are present
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Reorder columns to match training data
        df_encoded = df_encoded[self.feature_columns]
        
        # Scale numeric features
        numeric_features = [col for col in self.feature_columns 
                          if not any(col.startswith(c + '_') for c in ['employment_status', 'payment_history', 'province'])]
        if numeric_features:
            df_encoded[numeric_features] = self.scaler.transform(df_encoded[numeric_features])
        
        return df_encoded
        
    def explain_prediction(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a prediction
        """
        try:
            # Preprocess input
            processed_input = self.preprocess_input(input_data)
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(processed_input)
            
            # For binary classification, shap_values is a list of two arrays
            # We take the first array (probability of approval)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            # Create explanation dictionary
            feature_contributions = {}
            for i, feature in enumerate(self.feature_columns):
                feature_contributions[feature] = float(shap_values[0][i])
                
            # Sort features by absolute contribution
            sorted_features = sorted(
                feature_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            return {
                "feature_importance": dict(sorted_features),
                "base_value": float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) 
                                  else self.explainer.expected_value[0])
            }
            
        except Exception as e:
            print(f"Error in explain_prediction: {str(e)}")
            raise
        
    def get_rejection_explanation(self, input_data: Dict[str, Any], prediction_probability: float) -> Dict[str, Any]:
        """
        Get explanation for loan rejection with only feature importance and base value
        """
        explanation = self.explain_prediction(input_data)
        
        return {
            "technical_details": explanation,
            "rejection_probability": prediction_probability
        }
        
    def _generate_improvement_suggestions(self, top_factors: List[Dict[str, Any]]) -> List[str]:
        """
        Generate improvement suggestions based on top negative factors
        """
        suggestions = []
        for factor in top_factors:
            if factor["direction"] == "increased":
                feature = factor["feature"].replace("_", " ")
                
                if "credit_score" in factor["feature"]:
                    suggestions.append(f"Work on improving your credit score through timely payments and reducing credit utilization")
                elif "debt" in factor["feature"] or "dti" in factor["feature"].lower():
                    suggestions.append(f"Consider reducing your debt levels or increasing your income to improve debt-to-income ratio")
                elif "utilization" in factor["feature"]:
                    suggestions.append(f"Try to keep your credit utilization below 30% of your total credit limit")
                elif "income" in factor["feature"]:
                    suggestions.append(f"A higher income or additional income sources could improve your application")
                elif "payment_history" in factor["feature"]:
                    suggestions.append(f"Maintain a consistent record of on-time payments")
                    
        return suggestions 
        return suggestions 