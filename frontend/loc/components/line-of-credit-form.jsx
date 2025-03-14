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

  const handleChange = (field, value) => {
    setFormData({
      ...formData,
      [field]: value,
    })
  }

  const nextStep = () => {
    const newStep = step + 1
    setStep(newStep)
    setProgress(newStep * 20)
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
          setErrorMessage("Validation error: Please check your input data.")
          console.error("Validation error:", errorData)
        } else {
          setErrorMessage(`Server responded with status: ${response.status}`)
        }
        throw new Error(`Server responded with status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      setStep(7)
      setProgress(100)
    } catch (error) {
      console.error("Error submitting form:", error)
      setApiError(true)
      if (!errorMessage) {
        setErrorMessage("Network error: Unable to connect to the server. Please check your internet connection and try again.")
      }
      setResult(null)
      setStep(7)
      setProgress(100)
    } finally {
      setLoading(false)
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
                  Full Name
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
                  onChange={(e) => handleChange("full_name", e.target.value)} />
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
                        <p>You must be at least 18 years old to apply for a line of credit.</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </Label>
                <Input
                  id="age"
                  type="number"
                  min={18}
                  max={100}
                  placeholder="Enter your age"
                  value={formData.age === null ? '' : formData.age}
                  onChange={(e) => handleChange("age", e.target.value === '' ? null : Number.parseInt(e.target.value))} />
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
                    onValueChange={(value) => handleChange("self_reported_expenses", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className="font-medium">{formatCurrency(formData.self_reported_expenses)}</span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
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
                    onValueChange={(value) => handleChange("self_reported_debt", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className="font-medium">{formatCurrency(formData.self_reported_debt)}</span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
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
                    onValueChange={(value) => handleChange("estimated_debt", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className="font-medium">{formatCurrency(formData.estimated_debt)}</span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
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
                  placeholder="Enter months employed"
                  value={formData.months_employed === null ? '' : formData.months_employed}
                  onChange={(e) => handleChange("months_employed", e.target.value === '' ? null : Number.parseInt(e.target.value))} />
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
                    onValueChange={(value) => handleChange("annual_income", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$20,000</span>
                    <span className="font-medium">{formatCurrency(formData.annual_income)}</span>
                    <span className="text-sm text-gray-500">$200,000</span>
                  </div>
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
                    onValueChange={(value) => handleChange("credit_score", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">300</span>
                    <span className="font-medium">{formData.credit_score}</span>
                    <span className="text-sm text-gray-500">900</span>
                  </div>
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
                    onValueChange={(value) => handleChange("credit_utilization", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">0%</span>
                    <span className="font-medium">{formatPercent(formData.credit_utilization)}</span>
                    <span className="text-sm text-gray-500">100%</span>
                  </div>
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
                    placeholder="Enter number"
                    value={formData.num_open_accounts === null ? '' : formData.num_open_accounts}
                    onChange={(e) => handleChange("num_open_accounts", e.target.value === '' ? null : Number.parseInt(e.target.value))} />
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
                    placeholder="Enter number"
                    value={formData.num_credit_inquiries === null ? '' : formData.num_credit_inquiries}
                    onChange={(e) => handleChange("num_credit_inquiries", e.target.value === '' ? null : Number.parseInt(e.target.value))} />
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
                    onValueChange={(value) => handleChange("current_credit_limit", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className="font-medium">{formatCurrency(formData.current_credit_limit)}</span>
                    <span className="text-sm text-gray-500">$50,000</span>
                  </div>
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
                    onValueChange={(value) => handleChange("requested_amount", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$1,000</span>
                    <span className="font-medium">{formatCurrency(formData.requested_amount)}</span>
                    <span className="text-sm text-gray-500">$50,000</span>
                  </div>
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
                    onValueChange={(value) => handleChange("monthly_expenses", value[0])} />
                  <div className="flex justify-between">
                    <span className="text-sm text-gray-500">$0</span>
                    <span className="font-medium">{formatCurrency(formData.monthly_expenses)}</span>
                    <span className="text-sm text-gray-500">$10,000</span>
                  </div>
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
          <Button className="ml-auto bg-[#3d8b37] hover:bg-[#2c6428]" onClick={nextStep}>
            Next
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

