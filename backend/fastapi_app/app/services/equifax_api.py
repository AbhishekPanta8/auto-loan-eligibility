import os
import requests
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class EquifaxAPI:
    """Service for interacting with Equifax Pre-Approval of One API"""
    
    def __init__(self):
        # These would typically come from environment variables
        self.api_key = os.getenv("EQUIFAX_API_KEY", "test_api_key")
        self.api_secret = os.getenv("EQUIFAX_API_SECRET", "test_api_secret")
        self.base_url = os.getenv("EQUIFAX_API_URL", "https://api.equifax.com/preapproval/v1")
        self.is_sandbox = os.getenv("EQUIFAX_SANDBOX", "true").lower() == "true"
        
    def get_credit_report(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get credit report from Equifax Pre-Approval of One API
        
        Args:
            user_data: Dictionary containing user information needed for the API call
            
        Returns:
            Dictionary containing credit report data or error information
        """
        try:
            # In a real implementation, this would make an actual API call
            # For now, we'll simulate the API response
            if self.is_sandbox:
                return self._get_sandbox_credit_report(user_data)
            
            # Prepare headers with authentication
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the request payload
            payload = {
                "consumer": {
                    "name": {
                        "firstName": user_data.get("first_name", ""),
                        "lastName": user_data.get("last_name", "")
                    },
                    "dateOfBirth": user_data.get("date_of_birth", ""),
                    "addresses": [
                        {
                            "streetAddress": user_data.get("street_address", ""),
                            "city": user_data.get("city", ""),
                            "state": user_data.get("province", ""),
                            "zipCode": user_data.get("postal_code", "")
                        }
                    ],
                    "socialInsuranceNumber": user_data.get("sin", "")
                },
                "permissiblePurpose": "ACCOUNT_REVIEW",
                "consentId": user_data.get("consent_id", "")
            }
            
            # Make the API call
            response = requests.post(
                f"{self.base_url}/credit-report",
                headers=headers,
                json=payload
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Equifax API error: {response.status_code} - {response.text}")
                return {
                    "error": True,
                    "message": f"Error from Equifax API: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.exception("Error calling Equifax API")
            return {
                "error": True,
                "message": f"Error calling Equifax API: {str(e)}"
            }
    
    def _get_sandbox_credit_report(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a sandbox credit report for testing
        
        Args:
            user_data: Dictionary containing user information
            
        Returns:
            Dictionary containing simulated credit report data
        """
        # Extract the user's self-reported credit score if available
        self_reported_score = user_data.get("credit_score", 700)
        
        # Simulate some variance from the self-reported score
        # In a real scenario, the actual score might differ from what the user reports
        import random
        variance = random.randint(-50, 30)
        actual_score = max(300, min(850, self_reported_score + variance))
        
        # Simulate credit report data
        return {
            "creditReport": {
                "creditScore": actual_score,
                "creditHistoryLength": user_data.get("credit_history_length", 60),  # in months
                "missedPayments": max(0, user_data.get("missed_payments", 0) - 1),  # might be better than reported
                "creditUtilization": user_data.get("credit_utilization", 30),
                "openAccounts": user_data.get("num_open_accounts", 2),
                "creditInquiries": user_data.get("num_credit_inquiries", 1),
                "paymentHistory": user_data.get("payment_history", "On Time"),
                "totalCreditLimit": user_data.get("current_credit_limit", 5000) * 1.2,  # might be higher than reported
                "totalDebt": user_data.get("self_reported_debt", 0) * 1.1,  # might be higher than reported
            },
            "consentStatus": "APPROVED",
            "requestId": "sandbox-request-123456"
        } 