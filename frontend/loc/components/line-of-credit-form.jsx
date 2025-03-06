"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { HelpCircle, CheckCircle2, AlertCircle, Loader2, Info, ArrowRight } from "lucide-react"
import { Progress } from "@/components/ui/progress"

export function LineOfCreditForm() {
  const [step, setStep] = useState(1)
  const [progress, setProgress] = useState(20)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)

  const [formData, setFormData] = useState({
    full_name: "",
    self_reported_expenses: 0,
    credit_score: 700,
    annual_income: 50000,
    self_reported_debt: 0,
    requested_amount: 10000,
    age: 30,
    province: "ON",
    employment_status: "Full-time",
    months_employed: 12,
    credit_utilization: 30,
    num_open_accounts: 2,
    num_credit_inquiries: 1,
    payment_history: "On Time",
    current_credit_limit: 5000,
    monthly_expenses: 2000,
    estimated_debt: 0,
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
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      const data = await response.json()
      setResult(data)
      setStep(6)
      setProgress(100)
    } catch (error) {
      console.error("Error submitting form:", error)
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
                  value={formData.age}
                  onChange={(e) => handleChange("age", Number.parseInt(e.target.value))} />
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
                  value={formData.months_employed}
                  onChange={(e) => handleChange("months_employed", Number.parseInt(e.target.value))} />
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

        {step === 3 && (
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
                    value={formData.num_open_accounts}
                    onChange={(e) => handleChange("num_open_accounts", Number.parseInt(e.target.value))} />
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
                    value={formData.num_credit_inquiries}
                    onChange={(e) => handleChange("num_credit_inquiries", Number.parseInt(e.target.value))} />
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

              <div className="bg-[#f0f7ef] p-4 rounded-lg border border-[#3d8b37] mt-4">
                <h3 className="font-medium text-[#3d8b37] flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5" />
                  Review Your Application
                </h3>
                <p className="mt-2 text-sm">
                  Please review all the information you've provided before submitting. By clicking "Submit Application",
                  you authorize TD Bank to perform a credit check and process your application.
                </p>
              </div>
            </div>
          </div>
        )}

        {step === 6 && result && (
          <div className="space-y-6">
            <div
              className={`p-4 rounded-lg border ${result.approved === 1 ? "bg-[#f0f7ef] border-[#3d8b37]" : "bg-gray-100 border-gray-300"}`}>
              <h3
                className={`font-medium flex items-center gap-2 ${result.approved === 1 ? "text-[#3d8b37]" : "text-gray-700"}`}>
                {result.approved === 1 ? (
                  <>
                    <CheckCircle2 className="h-5 w-5" />
                    Congratulations! Your application is pre-approved
                  </>
                ) : (
                  <>
                    <AlertCircle className="h-5 w-5" />
                    We're unable to pre-approve your application at this time
                  </>
                )}
              </h3>
            </div>

            {result.approved === 1 && (
              <div className="grid md:grid-cols-2 gap-6">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Credit Limit</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold text-[#3d8b37]">{formatCurrency(result.approved_amount)}</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">Interest Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold text-[#3d8b37]">{result.interest_rate.toFixed(2)}%</p>
                  </CardContent>
                </Card>
              </div>
            )}

            <div className="space-y-4">
              {result.approved === 1 ? (
                <>
                  <p>
                    Based on the information you've provided, we're pleased to offer you a TD Line of Credit. This is a
                    pre-approval and is subject to verification of the information provided.
                  </p>
                  <div className="flex flex-col space-y-4">
                    <Button className="bg-[#3d8b37] hover:bg-[#2d6a27]">
                      Schedule an Appointment to Complete Your Application
                    </Button>
                    <Button variant="outline">Save Your Pre-Approval for Later</Button>
                  </div>
                </>
              ) : (
                <>
                  <p>
                    Based on the information you've provided, we're unable to pre-approve your application at this time.
                    This doesn't mean you won't qualify for a line of credit.
                  </p>
                  <div className="flex flex-col space-y-4">
                    <Button className="bg-[#3d8b37] hover:bg-[#2d6a27]">Speak with a TD Financial Advisor</Button>
                    <Button variant="outline">Learn How to Improve Your Application</Button>
                  </div>
                </>
              )}
            </div>
          </div>
        )}
      </CardContent>
      {step < 6 && (
        <CardFooter className="flex justify-between">
          {step > 1 ? (
            <Button variant="outline" onClick={prevStep}>
              Back
            </Button>
          ) : (
            <div></div>
          )}

          {step < 5 ? (
            <Button className="bg-[#3d8b37] hover:bg-[#2d6a27]" onClick={nextStep}>
              Continue
            </Button>
          ) : (
            <Button
              className="bg-[#3d8b37] hover:bg-[#2d6a27]"
              onClick={handleSubmit}
              disabled={loading}>
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
        </CardFooter>
      )}
      
      {step === 5 && (
        <div className="mt-6 mx-auto max-w-md">
          <div className="bg-blue-50 border-l-4 border-blue-500 rounded-md p-4 flex items-start shadow-md animate-pulse">
            <Info className="text-blue-500 mr-3 mt-0.5 h-5 w-5 flex-shrink-0" />
            <div>
              <p className="font-medium text-blue-700">Important</p>
              <p className="text-blue-600 mt-1">
                To get accurate results, you must login and verify your identity so we can securely fetch your financial data.
              </p>
              <div className="mt-2 flex items-center text-blue-700 font-medium">
                <span>Verify now</span>
                <ArrowRight className="ml-1 h-4 w-4" />
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}

