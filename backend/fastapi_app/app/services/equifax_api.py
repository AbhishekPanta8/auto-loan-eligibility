import os
import base64
import requests
import time
from typing import Dict, Optional, Any
import logging
import random
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class EquifaxAPI:
    """Service for interacting with Equifax Pre-Approval of One API"""
    
    def __init__(self):
        # Load and validate required environment variables
        self.api_key = os.getenv("EQUIFAX_API_KEY")
        self.api_secret = os.getenv("EQUIFAX_API_SECRET")
        self.member_number = os.getenv("EQUIFAX_MEMBER_NUMBER", "999XX12345")
        self.customer_code = os.getenv("EQUIFAX_CUSTOMER_CODE", "IAPI")
        self.security_code = os.getenv("EQUIFAX_SECURITY_CODE", "@U2")
        self.model_state = os.getenv("EQUIFAX_MODEL_STATE", "GA")
        # Validate required credentials
        if not self.api_key:
            raise ValueError("EQUIFAX_API_KEY not found in environment variables")
        if not self.api_secret:
            raise ValueError("EQUIFAX_API_SECRET not found in environment variables")
            
        # Environment settings
        self.is_sandbox = os.getenv("EQUIFAX_SANDBOX", "true").lower() == "true"
        
        # Set base URLs based on environment
        base_domain = "api.sandbox.equifax.com" if self.is_sandbox else "api.sandbox.equifax.com"
        self.base_url = f"https://{base_domain}/business/preapproval-of-one/v1"
        self.token_url = f"https://{base_domain}/v2/oauth/token"
        print("base_url", self.base_url)
        # Initialize token state
        self._access_token = None
        self._token_expiry = 0
        
        # Get initial access token for production environment
        if not self.is_sandbox:
            self._initialize_token()
    
    def _initialize_token(self) -> None:
        """Initialize or refresh the access token"""
        token_data = self._get_access_token()
        if token_data:
            self._access_token = token_data.get("access_token")
            # Set token expiry (subtract 5 minutes for safety margin)
            expires_in = token_data.get("expires_in", 3600)  # Default 1 hour if not specified
            self._token_expiry = time.time() + expires_in - 300  # Current time + expiry - 5 minutes
    
    def _is_token_valid(self) -> bool:
        """Check if the current token is valid and not expired"""
        return (
            self._access_token is not None
            and self._token_expiry > time.time()
        )
    
    def _get_access_token(self) -> Optional[Dict[str, Any]]:
        """
        Get OAuth2 access token using client credentials flow
        
        Returns:
            Dict containing token data including access_token and expires_in
        """
        try:
            # Create basic auth header
            credentials = f"{self.api_key}:{self.api_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {
                "grant_type": "client_credentials",
                "scope": "https://api.equifax.com/business/preapproval-of-one/v1"
            }
            
            # Print request details for debugging
            logger.info(f"Token URL: {self.token_url}")
            logger.info(f"Headers: {headers}")
            logger.info(f"Data: {data}")
            
            response = requests.post(
                self.token_url,
                headers=headers,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get access token: {response.status_code} - {response.text}")
                logger.error(f"Request URL: {self.token_url}")
                logger.error(f"Request Headers: {headers}")
                logger.error(f"Request Data: {data}")
                return None
                
        except Exception as e:
            logger.exception("Error getting access token")
            return None
    
    def _ensure_valid_token(self) -> bool:
        """Ensure we have a valid access token, refresh if necessary"""
        if self.is_sandbox:
            return True
            
        if not self._is_token_valid():
            self._initialize_token()
        
        return self._is_token_valid()
    
    def _extract_credit_data(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract credit data from Equifax response
        """
        try:
            credit_report = {}
            
            # Get the consumer credit report
            consumer_reports = response_data.get("consumers", {}).get("equifaxUSConsumerCreditReport", [])
            if not consumer_reports:
                raise ValueError("No consumer credit report found in response")
                
            consumer_report = consumer_reports[0]
            
            # Extract model scores
            models = consumer_report.get("models", [])
            scores = {}
            
            for model in models:
                model_id = model.get("identifier")
                model_type = model.get("type")
                score = model.get("score")
                reject_code = None
                
                # Check for reject codes
                rejects = model.get("rejects", [])
                if rejects:
                    reject_code = rejects[0].get("code")
                    logger.warning(f"Model {model_id} returned reject code: {reject_code}")
                    continue
                
                if score is not None:
                    scores[model_id] = {
                        "score": score,
                        "type": model_type,
                        "reasons": model.get("reasons", []),
                        "riskBasedPricing": model.get("riskBasedPricingOrModel", {})
                    }
            
            # Try to get the best available score in this order:
            # 1. FICO Score (02778)
            # 2. Any other available score
            if "02778" in scores:
                credit_report["creditScore"] = scores["02778"]["score"]
                credit_report["scoreType"] = "FICO"
                credit_report["scoreReasons"] = scores["02778"]["reasons"]
            else:
                # Get first available score
                for model_id, score_data in scores.items():
                    credit_report["creditScore"] = score_data["score"]
                    credit_report["scoreType"] = score_data["type"] or f"Model {model_id}"
                    credit_report["scoreReasons"] = score_data["reasons"]
                    break
            
            if "creditScore" not in credit_report:
                logger.error("No valid credit score found in response")
                raise ValueError("No valid credit score available")
            
            # Extract other credit information
            trades = consumer_report.get("trades", [])
            inquiries = consumer_report.get("inquiries", [])
            
            credit_report.update({
                "creditHistoryLength": consumer_report.get("fileAge", 0),
                "missedPayments": sum(1 for trade in trades if trade.get("paymentHistory", {}).get("code") in ["2", "3", "4"]),
                "creditUtilization": self._calculate_utilization(trades),
                "openAccounts": len([t for t in trades if t.get("status", {}).get("code") == "O"]),
                "creditInquiries": len(inquiries),
                "paymentHistory": self._analyze_payment_history(trades),
                "totalCreditLimit": self._calculate_total_credit(trades),
                "totalDebt": self._calculate_total_debt(trades)
            })
            
            return {
                "creditReport": credit_report,
                "consentStatus": "APPROVED",
                "requestId": response_data.get("requestId")
            }
            
        except Exception as e:
            logger.exception("Error extracting credit data")
            raise
    
    def _calculate_utilization(self, trades: list) -> float:
        total_balance = 0
        total_limit = 0
        for trade in trades:
            if trade.get("accountType", {}).get("code") in ["R", "C"]:  # Revolving or Credit Card
                total_balance += float(trade.get("currentBalance", 0))
                total_limit += float(trade.get("creditLimit", 0))
        return (total_balance / total_limit * 100) if total_limit > 0 else 0
    
    def _analyze_payment_history(self, trades: list) -> str:
        # Count late payments in the last 24 months
        late_payments = 0
        for trade in trades:
            payment_pattern = trade.get("paymentPattern", "")
            if payment_pattern:
                late_payments += sum(1 for p in payment_pattern[:24] if p in ["2", "3", "4"])
        
        if late_payments == 0:
            return "Excellent"
        elif late_payments <= 2:
            return "Good"
        elif late_payments <= 5:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_total_credit(self, trades: list) -> float:
        return sum(float(trade.get("creditLimit", 0)) for trade in trades)
    
    def _calculate_total_debt(self, trades: list) -> float:
        return sum(float(trade.get("currentBalance", 0)) for trade in trades)
    
    def get_credit_report(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get credit report from Equifax Pre-Approval of One API
        
        Args:
            user_data: Dictionary containing user information needed for the API call
            
        Returns:
            Dictionary containing credit report data or error information
        """
        try:
            # In sandbox mode, return simulated data
            if self.is_sandbox:
                return self._get_sandbox_credit_report(user_data)
            
            # Ensure we have a valid token
            if not self._ensure_valid_token():
                return {
                    "error": True,
                    "message": "Failed to obtain access token"
                }
            
            # Prepare headers with OAuth2 bearer token
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Generate a unique reference identifier
            reference_id = f"{int(time.time())}-{user_data.get('sin', '')[-4:]}"
            
            # Prepare the request payload according to Equifax API spec
            payload = {
                "consumers": {
                    "name": [
                        {
                            "identifier": "current",
                            "firstName": user_data["first_name"].upper(),
                            "lastName": user_data["last_name"].upper()
                        }
                    ],
                    "socialNum": [
                        {
                            "identifier": "current",
                            "number": user_data["sin"]
                        }
                    ],
                    "addresses": [
                        {
                            "identifier": "current",
                            "houseNumber": user_data["house_number"],
                            "streetName": user_data["street_name"].upper(),
                            "streetType": user_data["street_type"],
                            "city": user_data["city"].upper(),
                            "state": user_data["province"],
                            "zip": user_data["postal_code"]
                        }
                    ]
                },
                "customerReferenceidentifier": f"TD-{int(time.time())}",
                "customerConfiguration": {
                    "equifaxUSConsumerCreditReport": {
                        #"pdfComboIndicator": "Y",
                        # "memberNumber": self.member_number,
                        # "securityCode": self.security_code,
                        # "customerCode": self.customer_code,
                        "multipleReportIndicator": "1",
                        "models": [
                            {
                                "identifier": "02778",
                                "modelField": [
                                    "3",
                                    user_data["province"]
                                ]
                            },
                            {
                                "identifier": "05143"
                            },
                            {
                                "identifier": "02916"
                            }
                        ],
                        "ECOAInquiryType": "Individual"
                    }
                }
            }
            print("payload $$$$", payload)
            # Log request details in debug mode
            logger.debug(f"Making request to: {self.base_url}/report-requests")
            logger.debug(f"Headers: {headers}")
            logger.debug(f"Payload: {payload}")
            
            # Make the API call
            response = requests.post(
                f"{self.base_url}/report-requests",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                return self._extract_credit_data(response_data)
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
            Dictionary containing simulated credit report data that matches production format
        """
        # Extract the user's self-reported credit score if available
        self_reported_score = user_data.get("credit_score", 700)
        
        # Simulate some variance from the self-reported score
        variance = random.randint(-50, 30)
        actual_score = max(300, min(850, self_reported_score + variance))
        
        # Generate realistic reason codes
        reasons = []
        if actual_score < 700:
            reasons = [
                {"code": "32", "description": "Proportion of balances to credit limits is too high"},
                {"code": "21", "description": "Amount of time credit has been established"}
            ]
        
        # Simulate credit report data in the same format as the real API response
        return {
            "consumers": {
                "equifaxUSConsumerCreditReport": [{
                    "models": [
                        {
                            "identifier": "02778",
                            "type": "FICO",
                            "score": actual_score,
                            "modelField": ["3", "GA"],
                            "reasons": reasons,
                            "riskBasedPricingOrModel": {
                                "percentage": "85",
                                "lowRange": "300",
                                "highRange": "850"
                            }
                        },
                        {
                            "identifier": "05143",
                            "type": "MODEL",
                            "score": actual_score - 10,  # Slight variation
                            "reasons": reasons
                        }
                    ],
                    "trades": [
                        # Credit card account
                        {
                            "accountType": {"code": "R", "description": "Revolving"},
                            "currentBalance": 5000,
                            "creditLimit": 10000,
                            "paymentHistory": {"code": "1", "description": "Current"},
                            "status": {"code": "O", "description": "Open"},
                            "paymentPattern": "111111111111111111111111"  # Last 24 months
                        },
                        # Auto loan
                        {
                            "accountType": {"code": "I", "description": "Installment"},
                            "currentBalance": 15000,
                            "originalAmount": 25000,
                            "paymentHistory": {"code": "1", "description": "Current"},
                            "status": {"code": "O", "description": "Open"},
                            "paymentPattern": "111111111111111111111111"
                        }
                    ],
                    "inquiries": [
                        {
                            "inquiryDate": "2024-01-15",
                            "customerNumber": "ABC123",
                            "industryCode": {"code": "B", "description": "Bank"}
                        }
                    ],
                    "fileAge": user_data.get("credit_history_length", 60),
                    "consumerStatements": [
                        {
                            "dateReported": "2024-01",
                            "statement": "Consumer statement regarding credit history"
                        }
                    ]
                }]
            },
            "status": "completed",
            "requestId": f"sandbox-{random.randint(100000, 999999)}"
        } 