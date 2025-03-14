import numpy as np
import pandas as pd

# -------------------------------
# 1) CONFIG
# -------------------------------
ROWS = 5000  
np.random.seed(42)

# -------------------------------
# 2) Generate Applicant Dataset
# -------------------------------
def generate_applicant_dataset(num_rows=ROWS):
    """
    Creates an Applicant Input Dataset with:
      - applicant_id,
      - annual_income (lognormal ~ mean 60k, std ~30k, clipped 20k–200k),
      - self_reported_debt (~10–30% of monthly income),
      - self_reported_expenses (0–10k),
      - requested_amount (1k–50k),
      - age (19–100),
      - province (ON, BC),
      - employment_status (Full-time, Part-time, Unemployed),
      - months_employed (0–600)
    Returns a DataFrame of shape (num_rows, 9).
    """
    applicant_ids = [f"A{i:05d}" for i in range(1, num_rows + 1)]

    # annual_income: lognormal ~ mean=60k, std=30k
    raw_incomes = np.random.lognormal(mean=11, sigma=0.5, size=num_rows)
    annual_income = np.clip(raw_incomes, 20000, 200000)

    # monthly_income
    monthly_income = annual_income / 12.0

    # MODIFIED: self_reported_debt: 5–20% of monthly income (reduced from 10-30%)
    debt_factor = np.random.uniform(0.075, 0.25, size=num_rows)
    self_reported_debt = debt_factor * monthly_income

    # self_reported_expenses: uniform(0–10,000)
    self_reported_expenses = np.random.uniform(0, 10_000, num_rows)

    # requested_amount: uniform(1,000–50,000)
    requested_amount = np.random.uniform(1000, 30000, num_rows)

    # age: 19–100
    age = np.random.randint(19, 101, num_rows)

    # province: 70% "ON", 30% "BC"
    province = np.random.choice(["ON", "BC"], size=num_rows, p=[0.7, 0.3])

    # employment_status: Full-time(60%), Part-time(30%), Unemployed(10%)
    emp_choices = ["Full-time", "Part-time", "Unemployed"]
    emp_probs   = [0.6, 0.3, 0.1]
    employment_status = np.random.choice(emp_choices, size=num_rows, p=emp_probs)

    # months_employed: 0–600
    months_employed = np.random.randint(0, 601, num_rows)

    df_app = pd.DataFrame({
        "applicant_id": applicant_ids,
        "annual_income": annual_income,
        "self_reported_debt": self_reported_debt,
        "self_reported_expenses": self_reported_expenses,
        "requested_amount": requested_amount,
        "age": age,
        "province": province,
        "employment_status": employment_status,
        "months_employed": months_employed
    })

    return df_app

# -------------------------------
# 3) Generate Third-Party Dataset
# -------------------------------
def generate_credit_dataset(num_rows=ROWS):
    """
    Creates a Third-Party Credit Dataset with:
      - applicant_id,
      - credit_score: Normal(680,50) clipped to [300,900],
      - limit_factor: uniform(0.5–2.0) => total_credit_limit post-merge,
      - credit_utilization: Beta(2,4) => mean ~33% => scaled to 0–100,
      - num_open_accounts (0–20),
      - num_credit_inquiries (0–10),
      - payment_history (On Time, Late<30, Late 30-60, Late>60)
    Returns a DataFrame of shape (num_rows, 7).
    """
    applicant_ids = [f"A{i:05d}" for i in range(1, num_rows + 1)]

    # credit_score from Normal(680,100), clipped to [300,900]
    raw_scores = np.random.normal(loc=700, scale=40, size=num_rows)
    credit_score = np.clip(raw_scores, 300, 900).astype(int)

    # limit_factor => 50–200% of annual_income
    limit_factor = np.random.uniform(0.5, 2.0, num_rows)

    # credit_utilization => Beta(2,4) => mean ~0.33 => scale to 0..100
    beta_vals = np.random.beta(a=2, b=4, size=num_rows)
    credit_utilization = beta_vals * 100.0

    num_open_accounts = np.random.randint(0, 21, num_rows)
    num_credit_inquiries = np.random.randint(0, 11, num_rows)

    pmt_choices = ["On Time", "Late<30", "Late 30-60", "Late>60"]
    pmt_probs   = [0.70,     0.15,        0.10,        0.05]
    payment_history = np.random.choice(pmt_choices, size=num_rows, p=pmt_probs)

    df_credit = pd.DataFrame({
        "applicant_id": applicant_ids,
        "credit_score": credit_score,
        "limit_factor": limit_factor,
        "credit_utilization": credit_utilization,
        "num_open_accounts": num_open_accounts,
        "num_credit_inquiries": num_credit_inquiries,
        "payment_history": payment_history
    })

    return df_credit

# -------------------------------
# 4) Merge Data + total_credit_limit
# -------------------------------
def create_merged_dataset(df_applicant, df_credit):
    """
    Merges the two datasets on applicant_id. Then total_credit_limit = annual_income*limit_factor.
    Adds minimal placeholders for monthly_expenses. (We'll finalize approval, amount, and rate in later steps.)
    """
    df_merged = pd.merge(df_applicant, df_credit, on="applicant_id", how="inner")

    df_merged["total_credit_limit"] = df_merged["limit_factor"] * df_merged["annual_income"]
    df_merged.drop(columns=["limit_factor"], inplace=True)

    df_merged["monthly_expenses"] = np.random.uniform(0, 10000, len(df_merged))

    return df_merged

# -------------------------------
# 5) Derived Column: estimated_debt
# -------------------------------
def add_estimated_debt(df):
    """
    Adds 'estimated_debt' = total_credit_limit * (credit_utilization/100) * 0.02
    Modified from 0.03 to 0.02 to reduce DTI values overall
    """
    df["estimated_debt"] = (
        df["total_credit_limit"] * (df["credit_utilization"] / 100.0) * 0.025  # Reduced from 0.03
    )
    return df

# -------------------------------
# 6) Final Approval + Amount
# -------------------------------
def finalize_approval(df):
    """
    For each row, compute:
      - total_monthly_debt = self_reported_debt + estimated_debt
      - DTI = (total_monthly_debt + requested_amount*0.03)/(annual_income/12)*100

    Approve if:
      - credit_score >= 660
      - DTI <= 40
      - payment_history="On Time"
    Deny if:
      - credit_score < 500
      - DTI > 50
      - credit_utilization > 80
    Otherwise, random.

    We store results in df["approved"].
    """
    # compute total monthly debt
    monthly_debt = df["self_reported_debt"] + df["estimated_debt"]

    # compute DTI
    monthly_income = df["annual_income"] / 12.0
    dti = (monthly_debt + (df["requested_amount"] * 0.015)) / monthly_income * 100.0  # Reduced from 0.03 to 0.015
    df["DTI"] = dti

    def approve_row(row):
        score = row["credit_score"]
        dti_val = row["DTI"]
        util = row["credit_utilization"]
        pmt = row["payment_history"].lower()

        # Approve if ...
        if score >= 660 and dti_val <= 40.0 and pmt in ["on time", "on-time", "on_time"]:
            return 1

        # Deny if ...
        if score < 500 or dti_val > 50.0 or util > 80.0:
            return 0

        # Otherwise random
        return np.random.choice([0,1])

    df["approved"] = df.apply(approve_row, axis=1)
    return df

def finalize_approved_amount_and_interest(df):
    """
    If approved=1 => compute theoretical credit limit using the rules:
      - Base limit by credit_score:
          >=660 => 50% annual_income
          500-659 => 25%
          <500 => 10%
      - DTI adjustments:
          if DTI>40 => half
          elif DTI>30 => -25%
      - if credit_utilization>50 => -20%
      - if employment_status='Full-time' & months_employed>=12 => +10%
      - if payment_history='Late>60' => -50%
      - apply credit_score caps
         >=750 => 25k
         660–749 => 15k
         500–659 => 10k
         <500 => 5k
      => final_limit => min(base, cap)
      => approved_amount = min(requested_amount, final_limit)
    else => 0

    Then compute interest_rate with:
      base random in [3..6.75], 
      + or - adjustments:
        - credit_score>=750 => -1%
        - credit_score 500..659 => +2%, <500 => +4%
        - if DTI>30 => +1%
        - if payment_history='Late>60' => +2%
      clamp [3,15]
    """
    def compute_theoretical_limit(row):
        score = row["credit_score"]
        inc = row["annual_income"]
        dti = row["DTI"]
        util = row["credit_utilization"]
        emp = row["employment_status"].lower()
        months_emp = row["months_employed"]
        pmt = row["payment_history"].lower()

        # Base limit by score
        if score >= 660:
            base = 0.50 * inc
        elif score >= 500:
            base = 0.25 * inc
        else:
            base = 0.10 * inc

        # DTI adjustments
        if dti > 40.0:
            base *= 0.5
        elif dti > 30.0:
            base *= 0.75

        # credit_utilization>50 => -20%
        if util > 50.0:
            base *= 0.8

        # employment bonus
        if emp in ["full-time", "full_time"] and months_emp >= 12:
            base *= 1.10

        # payment penalty
        if pmt in ["late>60", "late >60", "late 60+", "60+"]:
            base *= 0.5

        # credit_score caps
        if score >= 750:
            cap = 25000
        elif score >= 660:
            cap = 15000
        elif score >= 500:
            cap = 10000
        else:
            cap = 5000

        final_limit = min(base, cap)
        return final_limit

    def compute_interest_rate(row):
        # base 3..6.75 random
        base_rate = np.random.uniform(3.0, 6.75)

        score = row["credit_score"]
        dti   = row["DTI"]
        pmt   = row["payment_history"].lower()

        # Score-based
        if score >= 750:
            base_rate -= 1.0
        elif score < 500:
            base_rate += 4.0
        elif score < 660:  # 500..659
            base_rate += 2.0
        # else 660..749 => no change

        # DTI>30 => +1%
        if dti > 30.0:
            base_rate += 1.0

        # Payment history Late>60 => +2%
        if pmt in ["late>60", "late >60", "late 60+", "60+"]:
            base_rate += 2.0

        # clamp to [3,15]
        if base_rate < 3.0:
            base_rate = 3.0
        if base_rate > 15.0:
            base_rate = 15.0

        return base_rate

    # For each row, if not approved => amount=0
    # if approved => min(requested, theoretical_limit)
    # also compute interest_rate
    def finalize_row(row):
        approved = row["approved"]
        if approved == 0:
            return pd.Series({"approved_amount": 0.0, "interest_rate": 0.0})
        else:
            # compute theoretical
            limit = compute_theoretical_limit(row)
            amt = min(row["requested_amount"], limit)
            rate = compute_interest_rate(row)
            return pd.Series({"approved_amount": amt, "interest_rate": rate})

    # apply row by row
    results = df.apply(finalize_row, axis=1)
    df["approved_amount"] = results["approved_amount"]
    df["interest_rate"]   = results["interest_rate"]
    return df

def introduce_missingness(df, missing_frac=0.02, columns=None):
    """
    Introduce NaNs in 'missing_frac' fraction of rows
    for each column in 'columns'. 
    If columns=None, we apply it to all columns.
    """
    if columns is None:
        columns = df.columns

    n_rows = len(df)
    for col in columns:
        n_missing = int(np.floor(missing_frac * n_rows))
        # Randomly pick n_missing row indices for this column
        idx_to_nan = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[idx_to_nan, col] = np.nan
    return df

def introduce_noise(df, noise_frac=0.05, columns=None):
    """
    Injects noise in a refined manner:
      - For annual_income, noise values are chosen from either [10000,20000) or (200000,210000]
      - For credit_score, noise values are chosen from either [200,300) or (900,1000]
      - For months_employed, noise values are chosen from either [-5,-1] or (600,650]
    """
    if columns is None:
        columns = []

    n_rows = len(df)
    n_noisy = int(np.floor(noise_frac * n_rows))

    for col in columns:
        idx_to_noise = np.random.choice(df.index, size=n_noisy, replace=False)
        if col == "annual_income":
            noise_values = np.random.choice(["low", "high"], size=n_noisy)
            new_vals = []
            for nv in noise_values:
                if nv == "low":
                    new_vals.append(np.random.uniform(20000, 22000))
                else:
                    new_vals.append(np.random.uniform(190000, 200000))
            df.loc[idx_to_noise, col] = new_vals

        elif col == "credit_score":
            noise_values = np.random.choice(["low", "high"], size=n_noisy)
            new_vals = []
            for nv in noise_values:
                if nv == "low":
                    new_vals.append(np.random.randint(200, 300))
                else:
                    new_vals.append(np.random.randint(901, 1000))
            df.loc[idx_to_noise, col] = new_vals

        elif col == "months_employed":
            noise_values = np.random.choice(["low", "high"], size=n_noisy)
            new_vals = []
            for nv in noise_values:
                if nv == "low":
                    new_vals.append(np.random.randint(-5, 0))
                else:
                    new_vals.append(np.random.randint(601, 651))
            df.loc[idx_to_noise, col] = new_vals

    return df

def diagnose_dti_components(df, sample_size=20):
    """
    For each row, computes:
      - monthly_income = annual_income / 12
      - self_debt_ratio = (self_reported_debt / monthly_income) * 100
      - estimated_debt_ratio = (estimated_debt / monthly_income) * 100
      - requested_component_ratio = (requested_amount * 0.03 / monthly_income) * 100
      - computed_DTI = sum of the above three ratios
    Prints a sample of rows and aggregated summary statistics.
    """
    valid_df = df.dropna(subset=["annual_income", "self_reported_debt", "estimated_debt", "requested_amount"])
    valid_df = valid_df[valid_df["annual_income"] > 0]

    monthly_income = valid_df["annual_income"] / 12.0
    self_debt_ratio = (valid_df["self_reported_debt"] / monthly_income) * 100
    estimated_debt_ratio = (valid_df["estimated_debt"] / monthly_income) * 100
    requested_component_ratio = (valid_df["requested_amount"] * 0.03 / monthly_income) * 100
    computed_DTI = self_debt_ratio + estimated_debt_ratio + requested_component_ratio

    diag_df = pd.DataFrame({
        "monthly_income": monthly_income,
        "self_debt_ratio": self_debt_ratio,
        "estimated_debt_ratio": estimated_debt_ratio,
        "requested_component_ratio": requested_component_ratio,
        "computed_DTI": computed_DTI
    })

    print("\n[DIAGNOSTIC] Sample DTI Component Breakdown (first {} rows):".format(sample_size))
    print(diag_df.head(sample_size))
    
    print("\n[DIAGNOSTIC] Aggregated Summary Statistics for DTI Components:")
    print(diag_df.describe())

# -------------------------------
# 7) Main
# -------------------------------
def main():
    # 1) Generate applicant + credit
    df_applicant = generate_applicant_dataset()
    df_credit = generate_credit_dataset()

    # 2) Merge -> total_credit_limit
    df_merged = create_merged_dataset(df_applicant, df_credit)

    # 3) Add estimated_debt
    df_merged = add_estimated_debt(df_merged)

    # 4) Compute approval
    df_merged = finalize_approval(df_merged)

    # 5) Compute approved_amount + interest
    df_merged = finalize_approved_amount_and_interest(df_merged)

    # # 6) Introduce missingness (1–2%)
    # df_merged = introduce_missingness(
    #     df_merged,
    #     missing_frac=0.02,  # ~2%
    #     columns=df_merged.columns  # or a subset
    # )

    # # 7) Introduce noise (5–10%) in some numeric columns
    # df_merged = introduce_noise(
    #     df_merged,
    #     noise_frac=0.05,   # ~8%
    #     columns=["annual_income", "credit_score", "months_employed"]
    #     # or whichever columns you want
    # )

    diagnose_dti_components(df_merged)

    # 6) Save
    df_applicant.to_csv("datasets/data/applicant_dataset.csv", index=False)
    df_credit.to_csv("datasets/data/third_party_dataset.csv", index=False)
    df_merged.to_csv("datasets/data/synthetic_loan_applications.csv", index=False)

    print("[INFO] Applicant dataset shape:", df_applicant.shape)
    print("[INFO] Third-party dataset shape:", df_credit.shape)
    print("[INFO] Merged dataset shape:", df_merged.shape)
    print(df_merged.head(10))

if __name__ == "__main__":
    main()
