"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { HelpCircle, CheckCircle2, AlertCircle, Loader2 } from "lucide-react"
import { Progress } from "@/components/ui/progress"
import ApprovalChart from './ui/RejectionChart'

export function LineOfCreditForm() {
  const [step, setStep] = useState(1)
  const [progress, setProgress] = useState(20)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [apiError, setApiError] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")

  const [formData, setFormData] = useState({
    full_name: "",
    age: 30,
    province: "ON",
    employment_status: "Full-time",
    months_employed: 12,
    annual_income: 50000,
    self_reported_debt: 0,
    debt_to_income_ratio: 0,
    credit_score: 700,
    credit_history_length: 60,
    missed_payments: 0,
    credit_utilization: 30,
    num_open_accounts: 2,
    num_credit_inquiries: 1,
    payment_history: "On Time",
    current_credit_limit: 5000,
    monthly_expenses: 2000,
    self_reported_expenses: 2000,
    estimated_debt: 0,
    requested_amount: 10000,
    preferred_term_months: 36,
    collateral_available: 0,
    equifax_consent: false,
    sin: "",
    date_of_birth: "",
    street_address: "",
    city: "",
    postal_code: "",
    house_number: "",
    street_name: "",
    street_type: "ST",
  })

  // Add state for field validation errors
  const [fieldErrors, setFieldErrors] = useState({})

  const handleChange = (field, value) => {
    // Create an updated form data object to make multiple changes if needed
    const updatedFormData = {
      ...formData,
      [field]: value,
    };
    
    // If employment status is being changed to student, unemployed, or retired, set months_employed to 0
    if (field === 'employment_status') {
      if (value === 'Student' || value === 'Unemployed' || value === 'Retired') {
        updatedFormData.months_employed = 0;
      }
    }
    
    // Update the form data with all changes
    setFormData(updatedFormData);

    // Clear error for the field that was just updated
    if (fieldErrors[field]) {
      const updatedErrors = { ...fieldErrors };
      delete updatedErrors[field]; // Remove the error completely instead of setting to null
      setFieldErrors(updatedErrors);
    }
    
    // If we also updated months_employed, clear its error too
    if (field === 'employment_status' && (value === 'Student' || value === 'Unemployed' || value === 'Retired')) {
      if (fieldErrors.months_employed) {
        const updatedErrors = { ...fieldErrors };
        delete updatedErrors.months_employed;
        setFieldErrors(updatedErrors);
      }
    }

    // Validate field based on constraints
    switch (field) {
      case 'full_name':
        if (!value || value.trim() === '') {
          setFieldErrors({
            ...fieldErrors,
            full_name: "Full name is required."
          });
        }
        break;
        
      case 'age':
        if (value < 19) {
          setFieldErrors({
            ...fieldErrors,
            age: "Age must be at least 19 years old to apply for a line of credit."
          });
        } else if (value > 100) {
          setFieldErrors({
            ...fieldErrors,
            age: "Age must be 100 years or less to apply for a line of credit."
          });
        }
        break;
        
      case 'self_reported_expenses':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            self_reported_expenses: "Monthly expenses cannot be negative."
          });
        } else if (value > 10000) {
          setFieldErrors({
            ...fieldErrors,
            self_reported_expenses: "Monthly expenses cannot exceed $10,000."
          });
        }
        break;
        
      case 'self_reported_debt':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            self_reported_debt: "Monthly debt payments cannot be negative."
          });
        } else if (value > 10000) {
          setFieldErrors({
            ...fieldErrors,
            self_reported_debt: "Monthly debt payments cannot exceed $10,000."
          });
        }
        break;
        
      case 'estimated_debt':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            estimated_debt: "Estimated debt cannot be negative."
          });
        } else if (value > 10000) {
          setFieldErrors({
            ...fieldErrors,
            estimated_debt: "Estimated debt cannot exceed $10,000."
          });
        }
        break;
        
      case 'months_employed':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            months_employed: "Months employed cannot be negative."
          });
        } else if (value > 600) {
          setFieldErrors({
            ...fieldErrors,
            months_employed: "Months employed cannot exceed 600."
          });
        }
        break;
        
      case 'annual_income':
        if (value < 20000) {
          setFieldErrors({
            ...fieldErrors,
            annual_income: "Annual income must be at least $20,000."
          });
        } else if (value > 200000) {
          setFieldErrors({
            ...fieldErrors,
            annual_income: "Annual income cannot exceed $200,000."
          });
        }
        break;
        
      case 'credit_score':
        if (value < 300) {
          setFieldErrors({
            ...fieldErrors,
            credit_score: "Credit score must be at least 300."
          });
        } else if (value > 900) {
          setFieldErrors({
            ...fieldErrors,
            credit_score: "Credit score cannot exceed 900."
          });
        }
        break;
        
      case 'credit_utilization':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            credit_utilization: "Credit utilization cannot be negative."
          });
        } else if (value > 100) {
          setFieldErrors({
            ...fieldErrors,
            credit_utilization: "Credit utilization cannot exceed 100%."
          });
        }
        break;
        
      case 'num_open_accounts':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            num_open_accounts: "Number of open accounts cannot be negative."
          });
        } else if (value > 20) {
          setFieldErrors({
            ...fieldErrors,
            num_open_accounts: "Number of open accounts cannot exceed 20."
          });
        }
        break;
        
      case 'num_credit_inquiries':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            num_credit_inquiries: "Number of credit inquiries cannot be negative."
          });
        } else if (value > 10) {
          setFieldErrors({
            ...fieldErrors,
            num_credit_inquiries: "Number of credit inquiries cannot exceed 10."
          });
        }
        break;
        
      case 'current_credit_limit':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            current_credit_limit: "Current credit limit cannot be negative."
          });
        } else if (value > 50000) {
          setFieldErrors({
            ...fieldErrors,
            current_credit_limit: "Current credit limit cannot exceed $50,000."
          });
        }
        break;
        
      case 'requested_amount':
        if (value < 1000) {
          setFieldErrors({
            ...fieldErrors,
            requested_amount: "Requested amount must be at least $1,000."
          });
        } else if (value > 50000) {
          setFieldErrors({
            ...fieldErrors,
            requested_amount: "Requested amount cannot exceed $50,000."
          });
        }
        break;
        
      case 'monthly_expenses':
        if (value < 0) {
          setFieldErrors({
            ...fieldErrors,
            monthly_expenses: "Monthly expenses cannot be negative."
          });
        } else if (value > 10000) {
          setFieldErrors({
            ...fieldErrors,
            monthly_expenses: "Monthly expenses cannot exceed $10,000."
          });
        }
        break;
    }
  }

  const validateStep = (currentStep) => {
    const errors = {};
    
    // Step 1 - Personal Information
    if (currentStep === 1) {
      // Full name validation
      if (!formData.full_name || formData.full_name.trim() === '') {
        errors.full_name = "Full name is required.";
      }
      
      // Age validation (19-100)
      if (formData.age < 19) {
        errors.age = "Age must be at least 19 years old to apply for a line of credit.";
      } else if (formData.age > 100) {
        errors.age = "Age must be 100 years or less to apply for a line of credit.";
      }
    }
    
    // Step 2 - Financial Information
    else if (currentStep === 2) {
      // Self-reported expenses validation (0-10,000)
      if (formData.self_reported_expenses < 0) {
        errors.self_reported_expenses = "Monthly expenses cannot be negative.";
      } else if (formData.self_reported_expenses > 10000) {
        errors.self_reported_expenses = "Monthly expenses cannot exceed $10,000.";
      }
      
      // Self-reported debt validation (0-10,000)
      if (formData.self_reported_debt < 0) {
        errors.self_reported_debt = "Monthly debt payments cannot be negative.";
      } else if (formData.self_reported_debt > 10000) {
        errors.self_reported_debt = "Monthly debt payments cannot exceed $10,000.";
      }
      
      // Estimated debt validation (0-10,000)
      if (formData.estimated_debt < 0) {
        errors.estimated_debt = "Estimated debt cannot be negative.";
      } else if (formData.estimated_debt > 10000) {
        errors.estimated_debt = "Estimated debt cannot exceed $10,000.";
      }
    }
    
    // Step 3 - Employment & Income
    else if (currentStep === 3) {
      // Months employed validation (0-600) - skip if employment status is Student, Unemployed, or Retired
      if (!['Student', 'Unemployed', 'Retired'].includes(formData.employment_status)) {
        if (formData.months_employed < 0) {
          errors.months_employed = "Months employed cannot be negative.";
        } else if (formData.months_employed > 600) {
          errors.months_employed = "Months employed cannot exceed 600.";
        }
      }
      
      // Annual income validation (20,000-200,000)
      if (formData.annual_income < 20000) {
        errors.annual_income = "Annual income must be at least $20,000.";
      } else if (formData.annual_income > 200000) {
        errors.annual_income = "Annual income cannot exceed $200,000.";
      }
    }
    
    // Step 4 - Credit Information
    else if (currentStep === 4) {
      // Credit score validation (300-900)
      if (formData.credit_score < 300) {
        errors.credit_score = "Credit score must be at least 300.";
      } else if (formData.credit_score > 900) {
        errors.credit_score = "Credit score cannot exceed 900.";
      }
      
      // Credit utilization validation (0-100%)
      if (formData.credit_utilization < 0) {
        errors.credit_utilization = "Credit utilization cannot be negative.";
      } else if (formData.credit_utilization > 100) {
        errors.credit_utilization = "Credit utilization cannot exceed 100%.";
      }
      
      // Open accounts validation (0-20)
      if (formData.num_open_accounts < 0) {
        errors.num_open_accounts = "Number of open accounts cannot be negative.";
      } else if (formData.num_open_accounts > 20) {
        errors.num_open_accounts = "Number of open accounts cannot exceed 20.";
      }
      
      // Credit inquiries validation (0-10)
      if (formData.num_credit_inquiries < 0) {
        errors.num_credit_inquiries = "Number of credit inquiries cannot be negative.";
      } else if (formData.num_credit_inquiries > 10) {
        errors.num_credit_inquiries = "Number of credit inquiries cannot exceed 10.";
      }
      
      // Current credit limit validation
      if (formData.current_credit_limit < 0) {
        errors.current_credit_limit = "Current credit limit cannot be negative.";
      } else if (formData.current_credit_limit > 50000) {
        errors.current_credit_limit = "Current credit limit cannot exceed $50,000.";
      }
    }
    
    // Step 5 - Line of Credit Request
    else if (currentStep === 5) {
      // Requested amount validation (1,000-50,000)
      if (formData.requested_amount < 1000) {
        errors.requested_amount = "Requested amount must be at least $1,000.";
      } else if (formData.requested_amount > 50000) {
        errors.requested_amount = "Requested amount cannot exceed $50,000.";
      }
      
      // Monthly expenses validation (0-10,000)
      if (formData.monthly_expenses < 0) {
        errors.monthly_expenses = "Monthly expenses cannot be negative.";
      } else if (formData.monthly_expenses > 10000) {
        errors.monthly_expenses = "Monthly expenses cannot exceed $10,000.";
      }
    }
    
    // Step 6 - Credit Check Authorization
    else if (currentStep === 6) {
      // Credit check authorization is not mandatory - removed validation
    }
    
    return errors;
  };

  const nextStep = () => {
    // Validate current step before proceeding
    const errors = validateStep(step);
    
    // If there are errors, update state and stop
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return;
    }
    
    // If validation passed, proceed to next step
    const newStep = step + 1;
    setStep(newStep);
    setProgress(newStep * 20);
  }

  const prevStep = () => {
    const newStep = step - 1
    setStep(newStep)
    setProgress(newStep * 20)
  }

  const handleSubmit = async () => {
    setLoading(true)
    setApiError(false)
    setErrorMessage("")
    
    // Validate all fields before submission
    const errors = {
      ...validateStep(1),
      ...validateStep(2),
      ...validateStep(3),
      ...validateStep(4),
      ...validateStep(5),
      ...validateStep(6)
    };
    
    // If there are any errors, don't submit
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setApiError(true);
      
      // Create a user-friendly error message from the validation errors
      const errorMessages = Object.values(errors);
      setErrorMessage(errorMessages.join("\n"));
      setLoading(false);
      return;
    }
    
    try {
      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        setApiError(true)
        if (response.status === 422) {
          const errorData = await response.json()
          console.error("Validation error:", errorData)
          
          // Extract detailed error message
          if (errorData.detail && Array.isArray(errorData.detail)) {
            const errorMessages = errorData.detail.map(error => {
              // Handle specific field errors
              if (error.loc && error.loc.length > 1) {
                const fieldName = error.loc[1];
                
                // Age-specific error message
                if (fieldName === 'age') {
                  if (error.msg.includes('greater than or equal to')) {
                    return "Age must be at least 19 years old to apply for a line of credit.";
                  }
                  if (error.msg.includes('less than or equal to')) {
                    return "Age must be 100 years or less to apply for a line of credit.";
                  }
                  return `Age error: ${error.msg}`;
                }
                
                // Format field name for display (e.g., convert "full_name" to "Full Name")
                const formattedFieldName = fieldName
                  .split('_')
                  .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(' ');
                
                return `${formattedFieldName}: ${error.msg}`;
              }
              return error.msg;
            });
            
            setErrorMessage(errorMessages.join("\n"));
        } else {
            setErrorMessage("Validation error: Please check your input data.");
          }
        } else {
          setErrorMessage(`Server responded with status: ${response.status}`);
        }
        return; // Don't proceed with submission
      }

      const data = await response.json();
      setResult(data);
      setStep(7);
      setProgress(100);
    } catch (error) {
      console.error("Error submitting form:", error);
      setApiError(true);
      if (!errorMessage) {
        setErrorMessage("Network error: Unable to connect to the server. Please check your internet connection and try again.");
      }
      setResult(null);
      setStep(7);
      setProgress(100);
    } finally {
      setLoading(false);
    }
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat("en-CA", {
      style: "currency",
      currency: "CAD",
      maximumFractionDigits: 0,
    }).format(value);
  }

  const formatPercent = (value) => {
    return `${value}%`
  }

  const resetForm = () => {
    setStep(1)
    setProgress(20)
    setResult(null)
    setApiError(false)
    setErrorMessage("")
    setFieldErrors({})
    setFormData({
      full_name: "",
      age: 30,
      province: "ON",
      employment_status: "Full-time",
      months_employed: 12,
      annual_income: 50000,
      self_reported_debt: 0,
      debt_to_income_ratio: 0,
      credit_score: 700,
      credit_history_length: 60,
      missed_payments: 0,
      credit_utilization: 30,
      num_open_accounts: 2,
      num_credit_inquiries: 1,
      payment_history: "On Time",
      current_credit_limit: 5000,
      monthly_expenses: 2000,
      self_reported_expenses: 2000,
      estimated_debt: 0,
      requested_amount: 10000,
      preferred_term_months: 36,
      collateral_available: 0,
      equifax_consent: false,
      sin: "",
      date_of_birth: "",
      street_address: "",
      city: "",
      postal_code: "",
      house_number: "",
      street_name: "",
      street_type: "ST",
    })
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="text-[#3d8b37]">Line of Credit Application</CardTitle>
        <CardDescription>Complete the form below to check your eligibility for a TD Line of Credit</CardDescription>
        <Progress value={progress} className="h-2 mt-2" />
      </CardHeader>
      <CardContent>
        {apiError && (
          <div className="mb-4 p-4 border border-red-200 rounded-lg bg-red-50">
            <div className="flex items-start">
              <AlertCircle className="h-5 w-5 text-red-600 mr-2 mt-0.5" />
              <div>
                <h4 className="text-red-800 font-medium mb-1">Error</h4>
                {errorMessage.split('\n').map((message, i) => (
                  <p key={i} className="text-red-700 text-sm">{message}</p>
                ))}
              </div>
            </div>
          </div>
        )}
        
        {step === 1 && (
          <div className="space-y-6">
            <div className="bg-[#f0f7ef] p-4 rounded-lg border border-[#3d8b37] mb-6">
              <h3 className="font-medium text-[#3d8b37] flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5" />
                Documents You'll Need
              </h3>
              <ul className="mt-2 space-y-2 text-sm">
                <li>• Government-issued ID (passport, driver's license)</li>
                <li>• Proof of income (pay stubs, tax returns)</li>
                <li>• Information about your existing debts and expenses</li>
                <li>• Your Social Insurance Number (for credit check)</li>
              </ul>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="full_name">
                  Full Name <span className="text-red-500">*</span>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-full-name">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="w-80">Enter your full legal name as it appears on your government ID.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  id="full_name"
                  placeholder="Enter your full name"
                  value={formData.full_name}
                  onChange={(e) => handleChange("full_name", e.target.value)}
                  className={fieldErrors.full_name ? "border-red-500" : ""} />
                {fieldErrors.full_name ? (
                  <p className="text-xs text-red-500">{fieldErrors.full_name}</p>
                ) : (
                  <p className="text-xs text-gray-500">As it appears on your government ID</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="age">
                  Your Age
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>You must be at least 19 years old to apply for a line of credit.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  id="age"
                  type="number"
                  min={19}
                  max={100}
                  placeholder="Enter your age (19-100)"
                  value={formData.age === null ? '' : formData.age}
                  onChange={(e) => handleChange("age", e.target.value === '' ? null : Number.parseInt(e.target.value))}
                  className={fieldErrors.age ? "border-red-500" : ""} />
                {fieldErrors.age ? (
                  <p className="text-xs text-red-500">{fieldErrors.age}</p>
                ) : (
                  <p className="text-xs text-gray-500">Applicants must be between 19-100 years old</p>
                )}
              </div>

              <div className="space-y-2">
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    <Label htmlFor="province">Province</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild data-testid="tooltip-trigger">
                          <HelpCircle className="h-4 w-4 text-gray-400" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Select your province of residence.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <Select value={formData.province} onValueChange={(value) => handleChange("province", value)}>
                    <SelectTrigger id="province" className="w-full">
                      <SelectValue placeholder="Select your province" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ON">Ontario</SelectItem>
                      <SelectItem value="BC">British Columbia</SelectItem>
                      <SelectItem value="AB">Alberta</SelectItem>
                      <SelectItem value="QC">Quebec</SelectItem>
                      <SelectItem value="MB">Manitoba</SelectItem>
                      <SelectItem value="SK">Saskatchewan</SelectItem>
                      <SelectItem value="NS">Nova Scotia</SelectItem>
                      <SelectItem value="NB">New Brunswick</SelectItem>
                      <SelectItem value="NL">Newfoundland and Labrador</SelectItem>
                      <SelectItem value="PE">Prince Edward Island</SelectItem>
                      <SelectItem value="YT">Yukon</SelectItem>
                      <SelectItem value="NT">Northwest Territories</SelectItem>
                      <SelectItem value="NU">Nunavut</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

              </div>
            </div>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-6">
            <h3 className="text-lg font-medium">Financial Information</h3>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="self_reported_expenses">
                  Monthly Expenses (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter your average monthly expenses excluding debt payments.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={0}
                    max={10000}
                    step={100}
                    value={[formData.self_reported_expenses]}
                    onValueChange={(value) => handleChange("self_reported_expenses", value[0])}
                    className={fieldErrors.self_reported_expenses ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className={`font-medium ${fieldErrors.self_reported_expenses ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.self_reported_expenses)}
                    </span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
                  {fieldErrors.self_reported_expenses && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.self_reported_expenses}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="self_reported_debt">
                  Monthly Debt Payments (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter your total monthly debt payments (rent, loans, etc.).</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={0}
                    max={10000}
                    step={100}
                    value={[formData.self_reported_debt]}
                    onValueChange={(value) => handleChange("self_reported_debt", value[0])}
                    className={fieldErrors.self_reported_debt ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className={`font-medium ${fieldErrors.self_reported_debt ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.self_reported_debt)}
                    </span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
                  {fieldErrors.self_reported_debt && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.self_reported_debt}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="estimated_debt">
                  Estimated Monthly Debt from Other Banks (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter your estimated monthly debt payments to other financial institutions.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={0}
                    max={10000}
                    step={100}
                    value={[formData.estimated_debt]}
                    onValueChange={(value) => handleChange("estimated_debt", value[0])}
                    className={fieldErrors.estimated_debt ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className={`font-medium ${fieldErrors.estimated_debt ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.estimated_debt)}
                    </span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
                  {fieldErrors.estimated_debt && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.estimated_debt}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 3 && (
          <div className="space-y-6">
            <h3 className="text-lg font-medium">Employment & Income Information</h3>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="employment_status">
                  Employment Status
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Select your current employment status.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Select
                  value={formData.employment_status}
                  onValueChange={(value) => handleChange("employment_status", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your employment status" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Full-time">Full-time</SelectItem>
                    <SelectItem value="Part-time">Part-time</SelectItem>
                    <SelectItem value="Unemployed">Unemployed</SelectItem>
                    <SelectItem value="Self-employed">Self-employed</SelectItem>
                    <SelectItem value="Retired">Retired</SelectItem>
                    <SelectItem value="Student">Student</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="months_employed">
                  Months at Current Job
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter the number of months you've been at your current job. Enter 0 if unemployed.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  id="months_employed"
                  type="number"
                  min={0}
                  max={600}
                  placeholder="Enter months employed (0-600)"
                  value={formData.months_employed === null ? '' : formData.months_employed}
                  onChange={(e) => handleChange("months_employed", e.target.value === '' ? null : Number.parseInt(e.target.value))}
                  className={`${fieldErrors.months_employed ? "border-red-500" : ""} ${['Student', 'Unemployed', 'Retired'].includes(formData.employment_status) ? "bg-gray-100" : ""}`}
                  disabled={['Student', 'Unemployed', 'Retired'].includes(formData.employment_status)} />
                {fieldErrors.months_employed ? (
                  <p className="text-xs text-red-500">{fieldErrors.months_employed}</p>
                ) : (
                  <p className="text-xs text-gray-500">
                    {['Student', 'Unemployed', 'Retired'].includes(formData.employment_status) 
                      ? "Automatically set to 0 based on employment status" 
                      : "Must be between 0-600 months"}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="annual_income">
                  Annual Income (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter your total yearly income including side income.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={20000}
                    max={200000}
                    step={1000}
                    value={[formData.annual_income]}
                    onValueChange={(value) => handleChange("annual_income", value[0])}
                    className={fieldErrors.annual_income ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$20,000</span>
                    <span className={`font-medium ${fieldErrors.annual_income ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.annual_income)}
                    </span>
                    <span className="text-sm text-gray-500">$200,000</span>
                  </div>
                  {fieldErrors.annual_income && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.annual_income}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-6">
            <h3 className="text-lg font-medium">Credit Information</h3>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="credit_score">
                  Credit Score
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          Enter your credit score (Equifax/TransUnion standard). You can find this in your credit
                          report.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={300}
                    max={900}
                    step={10}
                    value={[formData.credit_score]}
                    onValueChange={(value) => handleChange("credit_score", value[0])}
                    className={fieldErrors.credit_score ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">300</span>
                    <span className={`font-medium ${fieldErrors.credit_score ? "text-red-500" : ""}`}>
                      {formData.credit_score}
                    </span>
                    <span className="text-sm text-gray-500">900</span>
                  </div>
                  {fieldErrors.credit_score && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.credit_score}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="credit_utilization">
                  Credit Utilization (%)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>
                          Enter your current credit utilization percentage (how much of your available credit you're
                          using).
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={0}
                    max={100}
                    step={1}
                    value={[formData.credit_utilization]}
                    onValueChange={(value) => handleChange("credit_utilization", value[0])}
                    className={fieldErrors.credit_utilization ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">0%</span>
                    <span className={`font-medium ${fieldErrors.credit_utilization ? "text-red-500" : ""}`}>
                      {formatPercent(formData.credit_utilization)}
                    </span>
                    <span className="text-sm text-gray-500">100%</span>
                  </div>
                  {fieldErrors.credit_utilization && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.credit_utilization}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="payment_history">
                  Payment History
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Select your typical payment behavior.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Select
                  value={formData.payment_history}
                  onValueChange={(value) => handleChange("payment_history", value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select your payment history" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="On Time">Always on time</SelectItem>
                    <SelectItem value="Late <30">Occasionally late (less than 30 days)</SelectItem>
                    <SelectItem value="Last 30-60">Sometimes late (30-60 days)</SelectItem>
                    <SelectItem value="Late>60">Often late (more than 60 days)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="num_open_accounts">
                    Open Credit Accounts
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild data-testid="tooltip-trigger">
                          <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Number of active credit accounts you currently have.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </Label>
                  <Input
                    id="num_open_accounts"
                    type="number"
                    min={0}
                    max={20}
                    placeholder="Enter number (0-20)"
                    value={formData.num_open_accounts === null ? '' : formData.num_open_accounts}
                    onChange={(e) => handleChange("num_open_accounts", e.target.value === '' ? null : Number.parseInt(e.target.value))}
                    className={fieldErrors.num_open_accounts ? "border-red-500" : ""} />
                  {fieldErrors.num_open_accounts && (
                    <p className="text-xs text-red-500">{fieldErrors.num_open_accounts}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="num_credit_inquiries">
                    Recent Credit Inquiries
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild data-testid="tooltip-trigger">
                          <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Number of hard inquiries on your credit report in the last 12 months.</p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </Label>
                  <Input
                    id="num_credit_inquiries"
                    type="number"
                    min={0}
                    max={10}
                    placeholder="Enter number (0-10)"
                    value={formData.num_credit_inquiries === null ? '' : formData.num_credit_inquiries}
                    onChange={(e) => handleChange("num_credit_inquiries", e.target.value === '' ? null : Number.parseInt(e.target.value))}
                    className={fieldErrors.num_credit_inquiries ? "border-red-500" : ""} />
                  {fieldErrors.num_credit_inquiries && (
                    <p className="text-xs text-red-500">{fieldErrors.num_credit_inquiries}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="current_credit_limit">
                  Current Total Credit Limit (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter the sum of all your current credit limits across all accounts.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={0}
                    max={50000}
                    step={1000}
                    value={[formData.current_credit_limit]}
                    onValueChange={(value) => handleChange("current_credit_limit", value[0])}
                    className={fieldErrors.current_credit_limit ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className={`font-medium ${fieldErrors.current_credit_limit ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.current_credit_limit)}
                    </span>
                    <span className="text-sm text-gray-500">$50,000</span>
                  </div>
                  {fieldErrors.current_credit_limit && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.current_credit_limit}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 5 && (
          <div className="space-y-6">
            <h3 className="text-lg font-medium">Line of Credit Request</h3>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="requested_amount">
                  Requested Credit Limit (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter the credit limit you would like to request.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={1000}
                    max={50000}
                    step={1000}
                    value={[formData.requested_amount]}
                    onValueChange={(value) => handleChange("requested_amount", value[0])}
                    className={fieldErrors.requested_amount ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$1,000</span>
                    <span className={`font-medium ${fieldErrors.requested_amount ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.requested_amount)}
                    </span>
                    <span className="text-sm text-gray-500">$50,000</span>
                  </div>
                  {fieldErrors.requested_amount && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.requested_amount}</p>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="monthly_expenses">
                  Expected Monthly Credit Usage (CAD)
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild data-testid="tooltip-trigger">
                        <HelpCircle className="h-4 w-4 inline-block ml-1 text-gray-400" />
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>Enter how much you expect to spend on this line of credit each month.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <div className="space-y-3">
                  <Slider
                    min={0}
                    max={10000}
                    step={100}
                    value={[formData.monthly_expenses]}
                    onValueChange={(value) => handleChange("monthly_expenses", value[0])}
                    className={fieldErrors.monthly_expenses ? "opacity-80" : ""} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className={`font-medium ${fieldErrors.monthly_expenses ? "text-red-500" : ""}`}>
                      {formatCurrency(formData.monthly_expenses)}
                    </span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
                  {fieldErrors.monthly_expenses && (
                    <p className="text-xs text-red-500 mt-1">{fieldErrors.monthly_expenses}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {step === 6 && (
          <div className="space-y-6">
            <div className="bg-[#f0f7ef] p-4 rounded-lg border border-[#3d8b37] mb-6">
              <h3 className="font-medium text-[#3d8b37] flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5" />
                Credit Check Authorization
              </h3>
              <p className="mt-2 text-sm">
                To provide you with the most accurate pre-qualification results, we can check your credit score through Equifax's Pre-Approval of One service. This will not impact your credit score.
              </p>
            </div>

            <div className="space-y-4">
              <div className="flex items-start space-x-3">
                <input
                  type="checkbox"
                  id="equifax_consent"
                  className="mt-1"
                  checked={formData.equifax_consent}
                  onChange={(e) => handleChange("equifax_consent", e.target.checked)}
                />
                <Label htmlFor="equifax_consent" className="font-medium">
                  I authorize TD Bank to check my credit through Equifax's Pre-Approval of One service
                </Label>
              </div>

              {formData.equifax_consent && (
                <>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="sin">Social Insurance Number (SIN)</Label>
                      <Input
                        id="sin"
                        placeholder="Enter your SIN"
                        value={formData.sin}
                        onChange={(e) => handleChange("sin", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="date_of_birth">Date of Birth</Label>
                      <Input
                        id="date_of_birth"
                        type="date"
                        value={formData.date_of_birth}
                        onChange={(e) => handleChange("date_of_birth", e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="house_number">House Number</Label>
                      <Input
                        id="house_number"
                        placeholder="123"
                        value={formData.house_number}
                        onChange={(e) => handleChange("house_number", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2 col-span-2">
                      <Label htmlFor="street_name">Street Name</Label>
                      <Input
                        id="street_name"
                        placeholder="Main"
                        value={formData.street_name}
                        onChange={(e) => handleChange("street_name", e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="street_type">Street Type</Label>
                      <Select
                        value={formData.street_type}
                        onValueChange={(value) => handleChange("street_type", value)}
                      >
                        <SelectTrigger id="street_type">
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="ST">Street (ST)</SelectItem>
                          <SelectItem value="AV">Avenue (AV)</SelectItem>
                          <SelectItem value="RD">Road (RD)</SelectItem>
                          <SelectItem value="DR">Drive (DR)</SelectItem>
                          <SelectItem value="CR">Crescent (CR)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="city">City</Label>
                      <Input
                        id="city"
                        placeholder="Toronto"
                        value={formData.city}
                        onChange={(e) => handleChange("city", e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="postal_code">Postal Code</Label>
                      <Input
                        id="postal_code"
                        placeholder="A1B 2C3"
                        value={formData.postal_code}
                        onChange={(e) => handleChange("postal_code", e.target.value)}
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {step === 7 && (
          <div className="space-y-6">
            {apiError ? (
              <div className="p-6 rounded-lg border bg-yellow-50 border-yellow-200">
                <h3 className="text-xl font-bold mb-2 text-yellow-700">
                  Server Error
                </h3>
                <p className="text-gray-700 mb-4">
                  {errorMessage || "We're experiencing technical difficulties connecting to our server. Please try again later or contact customer support if the problem persists."}
                </p>
                <div className="mt-6">
                  <div className="space-y-4">
                    <Button
                      className="w-full bg-[#3d8b37] hover:bg-[#2c6428]"
                      onClick={() => {
                        setApiError(false)
                        setErrorMessage("")
                        setStep(6)
                        setProgress(80)
                      }}
                    >
                      Try Again
                    </Button>
                  </div>
                </div>
              </div>
            ) : result && (
              <>
                <div className={`p-6 rounded-lg border ${result.loan_approved ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"}`}>
                  <h3 className={`text-xl font-bold mb-2 ${result.loan_approved ? "text-green-700" : "text-red-700"}`}>
                    {result.loan_approved ? "Congratulations! You Pre-Qualify" : "We're Sorry"}
                  </h3>
                  <p className="text-gray-700 mb-4">
                    {result.loan_approved
                      ? `Based on the information provided, you pre-qualify for a line of credit up to ${formatCurrency(result.approved_amount)}.`
                      : "Based on the information provided, we are unable to pre-qualify you for a line of credit at this time."}
                  </p>

                  {result.loan_approved && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm text-gray-500">Pre-Qualified Amount</p>
                          <p className="text-xl font-bold text-[#3d8b37]">{formatCurrency(result.approved_amount)}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-500">Estimated Interest Rate</p>
                          <p className="text-xl font-bold text-[#3d8b37]">{formatPercent(result.interest_rate)}</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {formData.equifax_consent && result.credit_report && (
                  <div className="mt-6 p-6 rounded-lg border border-blue-200 bg-blue-50">
                    <h3 className="text-xl font-bold mb-4 text-blue-700">Your Credit Report Summary</h3>
                    <p className="text-gray-700 mb-4">
                      This information was obtained from Equifax's Pre-Approval of One service and did not impact your credit score.
                    </p>

                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <p className="text-sm text-gray-500">Credit Score</p>
                        <p className="text-xl font-bold">{result.credit_report.credit_score}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Credit History Length</p>
                        <p className="text-xl font-bold">{result.credit_report.credit_history_length} months</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Missed Payments</p>
                        <p className="text-xl font-bold">{result.credit_report.missed_payments}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Credit Utilization</p>
                        <p className="text-xl font-bold">{formatPercent(result.credit_report.credit_utilization)}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Open Accounts</p>
                        <p className="text-xl font-bold">{result.credit_report.open_accounts}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Recent Credit Inquiries</p>
                        <p className="text-xl font-bold">{result.credit_report.credit_inquiries}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Payment History</p>
                        <p className="text-xl font-bold">{result.credit_report.payment_history}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-500">Total Credit Limit</p>
                        <p className="text-xl font-bold">{formatCurrency(result.credit_report.total_credit_limit)}</p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="mt-6">
                  {result.loan_approved ? (
                    <div className="space-y-4">
                      <p className="text-gray-700">
                        To proceed with your application, please visit your nearest TD branch or call us at 1-800-555-1234.
                      </p>
                      <Button className="w-full bg-[#3d8b37] hover:bg-[#2c6428]">
                        Schedule an Appointment
                      </Button>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <p className="text-gray-700">
                        We recommend improving your credit score and reducing your debt before applying again. Our financial advisors can help you create a plan.
                      </p>
                      <Button className="w-full bg-[#3d8b37] hover:bg-[#2c6428]">
                        Speak with a Financial Advisor
                      </Button>
                    </div>
                  )}
                </div>

                {!result.loan_approved && result.explanation && (
                  <div className="mt-8">
                    <ApprovalChart 
                      featureImportance={result.explanation.technical_details.feature_importance}
                      baseValue={result.explanation.technical_details.base_value}
                      approvalProbability={result.approval_probability}
                      approvalThreshold={result.approval_threshold}
                    />
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                      <h4 className="text-lg font-semibold mb-2 text-gray-800">Understanding Your Results</h4>
                      <p className="text-gray-600 mb-2">
                        The chart above shows the top 5 factors that influenced your loan decision. 
                        Each factor's impact is shown as a percentage of the total decision.
                      </p>
                      <p className="text-gray-600">
                        Green markers indicate the threshold values you need to meet or exceed for a better chance of approval.
                        Focus on improving these factors to increase your chances of approval.
                      </p>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </CardContent>
      <CardFooter className="flex justify-between">
        {step > 1 && step < 7 && (
          <Button variant="outline" onClick={prevStep}>
            Back
          </Button>
        )}
        {step < 6 && (
          <Button 
            className={`ml-auto ${Object.keys(fieldErrors).length > 0 ? 'bg-red-500 hover:bg-red-600' : 'bg-[#3d8b37] hover:bg-[#2c6428]'}`} 
            onClick={nextStep}
          >
            {Object.keys(fieldErrors).length > 0 ? 'Fix Errors' : 'Next'}
          </Button>
        )}
        {step === 6 && (
          <Button
            className="ml-auto bg-[#3d8b37] hover:bg-[#2c6428]"
            onClick={handleSubmit}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing
              </>
            ) : (
              "Submit Application"
            )}
          </Button>
        )}
        {step === 7 && (
          <Button
            variant="outline"
            onClick={resetForm}
          >
            Start New Application
          </Button>
        )}
      </CardFooter>
    </Card>
  );
}

