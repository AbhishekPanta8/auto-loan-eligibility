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
            
            # Get top contributing factors
            top_factors = []
            for feature, impact in sorted_features[:5]:  # Get top 5 factors
                impact_type = "positive" if impact > 0 else "negative"
                # Clean up feature names for display
                display_feature = feature
                for prefix in ['employment_status_', 'payment_history_', 'province_']:
                    if feature.startswith(prefix):
                        display_feature = feature.replace(prefix, '')
                top_factors.append({
                    "feature": display_feature,
                    "impact": abs(impact),
                    "direction": impact_type
                })
                
            return {
                "feature_importance": dict(sorted_features),
                "base_value": float(self.explainer.expected_value if not isinstance(self.explainer.expected_value, list) 
                                  else self.explainer.expected_value[0]),
                "top_factors": top_factors
            }
        except Exception as e:
            print(f"Error in explain_prediction: {str(e)}")
            raise
        
    def get_rejection_explanation(self, input_data: Dict[str, Any], prediction_probability: float) -> Dict[str, Any]:
        """
        Get a human-readable explanation for loan rejection
        """
        explanation = self.explain_prediction(input_data)
        
        # Define thresholds and severity levels
        THRESHOLDS = {
            'credit_score': 660,  # Minimum for good approval chances
            'DTI': 40,  # Maximum DTI ratio
            'credit_utilization': 30,  # Maximum recommended utilization
            'annual_income': 40000,  # Minimum recommended income
            'months_employed': 12,  # Minimum preferred employment duration
            'estimated_debt_ratio': 0.4  # Maximum ratio of estimated debt to annual income
        }
        
        # Generate natural language explanation
        rejection_reasons = []
        improvement_suggestions = []
        
        # Use original values for explanations
        original_values = input_data  # Now using raw features directly
        
        # Calculate some derived metrics for better context
        annual_income = original_values.get('annual_income', 0)
        monthly_income = annual_income / 12 if annual_income > 0 else 0
        estimated_debt = original_values.get('estimated_debt', 0)
        self_reported_debt = original_values.get('self_reported_debt', 0)
        total_debt = estimated_debt + self_reported_debt
        debt_to_income = total_debt / annual_income if annual_income > 0 else 0
        
        for factor in explanation["top_factors"]:
            feature = factor["feature"].replace("_", " ")
            severity = "significantly " if factor["impact"] > 1.5 else ""
            high_severity = factor["impact"] > 2.0
            
            # Negative SHAP values decrease approval probability
            if factor["direction"] == "negative":
                if "payment_history" in factor["feature"]:
                    if "Late" in factor["feature"] and high_severity:
                        payment_history = original_values.get('payment_history', 'Unknown')
                        rejection_reasons.append(
                            f"Your payment history shows {severity}concerning patterns that strongly affect your approval chances"
                        )
                        improvement_suggestions.append(
                            "Establish a solid record of on-time payments for all credit accounts for at least 12 months"
                        )
                        if payment_history in ["Late<30", "Late>60"]:
                            improvement_suggestions.append(
                                "Address any outstanding late payments and ensure all accounts are current"
                            )
                
                elif "months_employed" in factor["feature"]:
                    months = original_values.get('months_employed', 'N/A')
                    if months != 'N/A':
                        if months < THRESHOLDS['months_employed']:
                            rejection_reasons.append(
                                f"Your employment duration of {months} months is below our preferred minimum "
                                f"of {THRESHOLDS['months_employed']} months"
                            )
                            improvement_suggestions.append(
                                f"Consider reapplying after maintaining stable employment for at least "
                                f"{THRESHOLDS['months_employed']} months"
                            )
                
                elif "debt" in factor["feature"] or "DTI" in factor["feature"]:
                    if debt_to_income > 0:
                        ratio_pct = debt_to_income * 100
                        rejection_reasons.append(
                            f"Your total debt of ${total_debt:,.2f} represents {ratio_pct:.1f}% of your "
                            f"annual income, which is {severity}high"
                        )
                        improvement_suggestions.append(
                            f"Work on reducing your total debt or increasing your income to improve your "
                            f"debt-to-income ratio"
                        )
                
                elif "credit_score" in factor["feature"]:
                    credit_score = original_values.get('credit_score', 'N/A')
                    if credit_score != 'N/A':
                        gap = THRESHOLDS['credit_score'] - credit_score
                        if gap > 0:
                            rejection_reasons.append(
                                f"Your credit score of {credit_score} is {severity}below our threshold "
                                f"of {THRESHOLDS['credit_score']} for automatic approval"
                            )
                            improvement_suggestions.append(
                                f"Work on improving your credit score by at least {gap} points through "
                                f"timely payments and reducing credit utilization"
                            )
                
            # Positive SHAP values increase approval probability
            else:
                if "payment_history" in factor["feature"] and "On Time" in factor["feature"]:
                    # This is a positive factor, so we don't add it to rejection reasons
                    continue
                
                elif "estimated_debt" in factor["feature"] and debt_to_income > THRESHOLDS['estimated_debt_ratio']:
                    rejection_reasons.append(
                        f"While your estimated debt level is within acceptable ranges, your total "
                        f"debt-to-income ratio of {debt_to_income*100:.1f}% is concerning"
                    )
                    improvement_suggestions.append(
                        "Consider reducing your overall debt burden before applying for additional credit"
                    )
                
                elif "credit_score" in factor["feature"]:
                    credit_score = original_values.get('credit_score', 'N/A')
                    if credit_score != 'N/A':
                        rejection_reasons.append(
                            f"Your credit score of {credit_score} is favorable, but is outweighed by "
                            f"other risk factors in your application"
                        )
                
                elif "employment" in factor["feature"]:
                    status = original_values.get('employment_status', 'N/A')
                    months = original_values.get('months_employed', 'N/A')
                    if months != 'N/A':
                        rejection_reasons.append(
                            f"Your {status} employment status for {months} months is positive, but "
                            f"other factors in your application present higher risk"
                        )
                    
        # Add overall risk assessment if rejection probability is high
        if prediction_probability > 0.6:
            rejection_reasons.append(
                f"Overall, your application shows a {prediction_probability*100:.1f}% likelihood of default "
                f"based on our risk assessment model"
            )
            improvement_suggestions.append(
                "Consider working on multiple factors simultaneously: payment history, debt reduction, "
                "and credit score improvement"
            )
                
        return {
            "technical_details": explanation,
            "rejection_probability": prediction_probability,
            "main_factors": rejection_reasons,
            "improvement_suggestions": list(set(improvement_suggestions))  # Remove duplicates
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