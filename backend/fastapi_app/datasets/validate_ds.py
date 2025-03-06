from collections import Counter
import sys
import pandas as pd
import numpy as np

def read_dataset(csv_path):
    """
    Attempts to read the CSV into a pandas DataFrame.
    Returns the DataFrame if successful, or None if there's an error.
    """
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return None

def check_shape(df, min_rows=4000, max_rows=5000):
    """
    Checks the shape of the DataFrame (row count) to ensure it's within [min_rows, max_rows].
    Prints warnings if it's outside that range.
    """
    n_rows, n_cols = df.shape
    print(f"[INFO] Loaded dataset with {n_rows} rows and {n_cols} columns.")

    if n_rows < min_rows or n_rows > max_rows:
        print(f"[WARNING] Row count {n_rows} is outside the {min_rows}–{max_rows} range.")
    else:
        print(f"[OK] Row count {n_rows} is within expected range ({min_rows}–{max_rows}).")

def check_required_columns(df, required_columns):
    """
    Checks if all required columns are present and reports any extras.
    """
    missing_cols = set(required_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(required_columns)

    if missing_cols:
        print(f"[ERROR] Missing required columns: {missing_cols}")
    else:
        print("[OK] All required columns are present.")

    if extra_cols:
        print(f"[INFO] Extra columns found (not in spec): {extra_cols}")

def check_data_types(df, required_columns, expected_dtypes):
    """
    Checks if each column has the expected pandas dtype 'kind'.
    Because of missing values, an 'int' column in the spec may show as float in pandas.
    We do a relaxed check by comparing just the 'kind' (e.g., 'f' for float, 'O' for object).
    """
    for col in required_columns:
        if col not in df.columns:
            # Already flagged as missing; skip
            continue

        actual_kind = df[col].dtype.kind  # e.g. 'f' for float, 'i' for int, 'O' for object
        expected_kind = expected_dtypes.get(col, None)
        if expected_kind is None:
            # Not in our mapping, skip
            continue

        if actual_kind != expected_kind:
            print(
                f"[WARNING] Column '{col}' dtype is '{df[col].dtype}' (kind='{actual_kind}'), "
                f"expected kind='{expected_kind}'."
            )

def check_range(df, column, valid_min, valid_max, allow_missing=True):
    """
    Helper to determine how many values in `column` fall within [valid_min, valid_max].
    Prints the fraction of out-of-range values. 
    """
    if column not in df.columns:
        return

    series = df[column]
    if allow_missing:
        series = series.dropna()

    total_count = len(series)
    if total_count == 0:
        print(f"    {column}: No non-missing values to check.")
        return

    in_range_mask = (series >= valid_min) & (series <= valid_max)
    in_range_count = in_range_mask.sum()
    out_of_range_count = total_count - in_range_count
    fraction_out = out_of_range_count / total_count

    print(
        f"    {column}: {in_range_count}/{total_count} in range "
        f"({valid_min}, {valid_max}), out-of-range fraction={fraction_out:.2%}"
    )

def check_numeric_ranges(df):
    """
    Invokes check_range for each numeric column we want to validate.
    Because we injected noise (5-10%), we expect some out-of-range values.
    """
    print("\n[INFO] Checking fraction of values within normal ranges (allowing for noise):")
    check_range(df, "self_reported_expenses", 0, 10000)
    check_range(df, "credit_score", 300, 900)
    check_range(df, "annual_income", 20000, 200000)
    check_range(df, "self_reported_debt", 0, 10000)
    check_range(df, "requested_amount", 1000, 50000)
    check_range(df, "age", 19, 100)
    check_range(df, "months_employed", 0, 600)
    check_range(df, "credit_utilization", 0, 100)
    check_range(df, "num_open_accounts", 0, 20)
    check_range(df, "num_credit_inquiries", 0, 10)
    check_range(df, "total_credit_limit", 0, 50000)
    check_range(df, "monthly_expenses", 0, 10000)
    check_range(df, "approved_amount", 0, 50000)
    check_range(df, "estimated_debt", 0, 10000)
    check_range(df, "interest_rate", 3.0, 15.0)

def check_categorical_values(df):
    """
    Checks that certain columns only contain specific valid categories (aside from NaNs).
    """
    print("\n[INFO] Checking categorical columns (province, employment_status, payment_history, approved)...")

    # Province check
    if "province" in df.columns:
        invalid_prov = df[~df["province"].isin(["ON", "BC"]) & df["province"].notna()]
        if not invalid_prov.empty:
            print(f"[WARNING] {len(invalid_prov)} rows have invalid 'province' values (not 'ON' or 'BC').")

    # Employment status check
    if "employment_status" in df.columns:
        invalid_emp = df[
            ~df["employment_status"].isin(["Full-time", "Part-time", "Unemployed"]) &
            df["employment_status"].notna()
        ]
        if not invalid_emp.empty:
            print(f"[WARNING] {len(invalid_emp)} rows have invalid 'employment_status'.")

    # Payment history check
    if "payment_history" in df.columns:
        valid_ph = ["On Time", "Late<30", "Late 30-60", "Late>60"]
        invalid_ph = df[~df["payment_history"].isin(valid_ph) & df["payment_history"].notna()]
        if not invalid_ph.empty:
            print(f"[WARNING] {len(invalid_ph)} rows have invalid 'payment_history' values.")

    # Approved check
    if "approved" in df.columns:
        invalid_approved = df[(df["approved"].notna()) & (~df["approved"].isin([0,1]))]
        if not invalid_approved.empty:
            print(f"[WARNING] {len(invalid_approved)} rows have 'approved' not in [0,1].")

def check_approved_distribution(df):
    """
    Checks the fraction of approved == 1 and denied == 0.
    Displays actual approval and denial percentages and warns if approval is outside 60–70%.
    """
    if "approved" not in df.columns:
        print("[ERROR] 'approved' column is missing from the dataset.")
        return

    valid_approved = df["approved"].dropna()
    total_count = len(valid_approved)

    if total_count == 0:
        print("[WARNING] 'approved' column has only NaNs.")
        return

    count_approved = (valid_approved == 1).sum()
    count_denied = (valid_approved == 0).sum()

    frac_approved = count_approved / total_count
    frac_denied = count_denied / total_count

    print(f"[INFO] Approval rate: {frac_approved:.2%} ({count_approved}/{total_count})")
    print(f"[INFO] Denial rate: {frac_denied:.2%} ({count_denied}/{total_count})")

    if not (0.60 <= frac_approved <= 0.70):
        print("[WARNING] Approval rate is outside the expected 60–70% range.")

    if not (0.30 <= frac_denied <= 0.40):
        print("[WARNING] Denial rate is outside the expected 30–40% range.")

def check_missing_data(df):
    """
    Prints out the percentage of missing data per column and warns if >5%.
    """
    n_rows = len(df)
    missing_counts = df.isna().sum()
    fraction_missing = (missing_counts / n_rows) * 100

    print("\n[INFO] Missing data fraction by column (%):")
    print(fraction_missing)

    high_missing = fraction_missing[fraction_missing > 5]
    if not high_missing.empty:
        print("[WARNING] Some columns exceed 5% missing values:")
        print(high_missing)

def check_credit_score_distribution(df, expected_mean=680, expected_std=100, tolerance=50):
    """
    Checks that the credit_score's sample mean & std are somewhat close to (680, 100).
    'tolerance' here is a rough boundary for how many points away from the expected
    we allow before printing a warning.
    """
    if "credit_score" not in df.columns:
        return

    data = df["credit_score"].dropna()
    if len(data) == 0:
        print("[WARNING] 'credit_score' is all NaN.")
        return

    sample_mean = data.mean()
    sample_std = data.std()

    # Check how far from expected
    mean_diff = abs(sample_mean - expected_mean)
    std_diff = abs(sample_std - expected_std)

    print(f"[INFO] credit_score distribution: mean={sample_mean:.2f}, std={sample_std:.2f}")

    # If these differences are big, raise a warning
    if mean_diff > tolerance:
        print(f"[WARNING] credit_score mean is off by >{tolerance} from {expected_mean}.")

    if std_diff > tolerance:
        print(f"[WARNING] credit_score std is off by >{tolerance} from {expected_std}.")

def check_annual_income_distribution(df, expected_mean=60000, expected_std=30000, z_tolerance=2.0):
    """
    Checks that annual_income's sample mean & std are roughly ~60k mean, ~30k std.
    We do a simple z-score check: if the difference from expected mean/std 
    is more than 'z_tolerance' standard deviations, we warn.
    """
    if "annual_income" not in df.columns:
        print("[WARNING] 'annual_income' column is missing.")
        return

    data = df["annual_income"].dropna()
    if len(data) == 0:
        print("[WARNING] 'annual_income' is all NaN.")
        return

    sample_mean = data.mean()
    sample_std = data.std()

    # Compute how many stdevs the sample is from the "expected" mean
    # we do an approximate approach: z_mean = |(sample_mean - expected_mean)| / expected_std
    # this presumes the expected_std is the correct scale.
    if expected_std > 0:
        z_mean = abs(sample_mean - expected_mean) / expected_std
        z_std = abs(sample_std - expected_std) / expected_std
    else:
        z_mean, z_std = 0, 0  # avoid division by zero

    print(f"[INFO] annual_income distribution: mean={sample_mean:.2f}, std={sample_std:.2f}")

    if z_mean > z_tolerance:
        print(f"[WARNING] annual_income mean differs from {expected_mean} by more than {z_tolerance} std devs.")

    if z_std > z_tolerance:
        print(f"[WARNING] annual_income std differs from {expected_std} by more than {z_tolerance} std devs.")

def check_monthly_debt_ratio(df):
    """
    Checks that self_reported_debt is roughly 10–30% of the applicant's monthly income.
    i.e. ratio = (self_reported_debt) / (annual_income/12) in [0.1, 0.3].
    Prints how many rows are in range vs. out-of-range.
    """
    # First, ensure the columns exist
    if "annual_income" not in df.columns or "self_reported_debt" not in df.columns:
        print("[WARNING] Missing annual_income or self_reported_debt column.")
        return

    # Drop rows with missing data
    valid_df = df.dropna(subset=["annual_income", "self_reported_debt"])
    if len(valid_df) == 0:
        print("[WARNING] No non-missing rows to check for monthly debt ratio.")
        return

    monthly_income = valid_df["annual_income"] / 12.0
    ratio = valid_df["self_reported_debt"] / monthly_income

    # We expect ratio ~0.1–0.3
    in_range_mask = ratio.between(0.1, 0.3)
    in_range_count = in_range_mask.sum()
    total_count = len(ratio)
    out_of_range_count = total_count - in_range_count
    fraction_out = out_of_range_count / total_count if total_count else 0

    print(f"[INFO] Monthly debt ratio: {in_range_count}/{total_count} in [10%, 30%], "
          f"out-of-range fraction={fraction_out:.2%}")

def validate_approved_amount(row):
    """
    Validate approved_amount for a single applicant row using the provided rules.
    
    Args:
        row (dict or pandas.Series): Must contain these keys at minimum:
            - approved: 0 or 1
            - approved_amount: float
            - annual_income: float
            - credit_score: float
            - credit_utilization: float (0-100)
            - payment_history: str
            - employment_status: str
            - months_employed: float or int
            - self_reported_debt: float
            - estimated_debt: float
            - requested_amount: float
    
    Returns:
        dict with:
            - valid: bool (True if the row's approved_amount follows the rules)
            - explanation: str describing any rule violations or success message
    """
    # ------------------------------
    # 1) Basic field extraction
    # ------------------------------
    approved = row.get("approved", None)
    app_amount = row.get("approved_amount", None)

    income = row.get("annual_income", 0.0)
    score = row.get("credit_score", 0.0)
    util  = row.get("credit_utilization", 0.0)  # 0-100
    pmt_history = str(row.get("payment_history", "")).lower()
    emp_status  = str(row.get("employment_status", "")).lower()
    months_emp  = row.get("months_employed", 0)
    sr_debt     = row.get("self_reported_debt", 0.0)
    est_debt    = row.get("estimated_debt", 0.0)
    req_amount  = row.get("requested_amount", 0.0)

    # Convert utilization to fraction [0..1]
    util_fraction = util / 100.0

    # ------------------------------
    # 2) Compute DTI
    # ------------------------------
    # total monthly debt = self_reported_debt + estimated_debt
    # DTI = (total_monthly_debt + (requested_amount * 0.03)) / (annual_income / 12)
    total_monthly_debt = sr_debt + est_debt
    if income <= 0:
        # If income is 0 or negative, any DTI is effectively infinite
        dti = 9999
    else:
        monthly_income = income / 12.0
        dti = (total_monthly_debt + (req_amount * 0.03)) / monthly_income * 100.0  # store as percentage if desired

    # ------------------------------
    # 3) Check if row *should* be approved or denied, per the rules
    # ------------------------------
    # Approve if Credit Score ≥ 660, DTI ≤ 40%, Payment History="On-time"
    # Deny if Score <500 OR DTI>50% OR util>80%
    should_approve = (score >= 660) and (dti <= 40.0) and (pmt_history.lower() in ["on time", "on-time", "on_time"])
    should_deny    = (score < 500) or (dti > 50.0) or (util > 80.0)

    # If row "should_approve" but the row says approved=0, or vice versa, that's suspicious. 
    # But let's not fail the entire limit check automatically; we'll keep it as a note.
    # Because the question specifically wants to validate "approved_amount" – not the entire 
    # yes/no logic. Still, it's relevant if the row is approved=0 but "should be" 1, or vice versa.

    # ------------------------------
    # 4) Compute "theoretical" max limit based on rules
    # ------------------------------
    # A) Base Limit by credit_score
    if score >= 660:
        base_limit = 0.50 * income
    elif score >= 500:
        base_limit = 0.25 * income
    else:
        base_limit = 0.10 * income

    # B) DTI Adjustments
    #   - ≤30% => no reduction
    #   - 30% < DTI ≤ 40% => -25%
    #   - DTI > 40% => -50% or deny
    if dti > 40.0:
        # They might be denied. But if they're not auto-denied by the other rules, it's -50%:
        base_limit *= 0.50
    elif dti > 30.0:  # so 30 < DTI <= 40
        base_limit *= 0.75
    # else if ≤30%, no reduction

    # C) credit_score caps
    #   ≥750 => $25k max; 660–749 => $15k; 500–659 => $10k; <500 => $5k
    if score >= 750:
        score_cap = 25000
    elif score >= 660:
        score_cap = 15000
    elif score >= 500:
        score_cap = 10000
    else:
        score_cap = 5000

    # D) Reduce base limit by 20% if credit_utilization > 50%
    if util > 50.0:
        base_limit *= 0.80

    # E) Employment Bonus: "Full-time" AND months_employed >= 12 => +10%
    if emp_status in ["full-time", "full_time"] and months_emp >= 12:
        base_limit *= 1.10

    # F) Payment Penalty: "Late >60": -50%
    if pmt_history in ["late>60", "late >60", "late 60+", "60+"]:
        base_limit *= 0.50

    # Finally, the "theoretical" limit is the min of base_limit vs. score_cap
    theoretical_limit = min(base_limit, score_cap)

    # If the row "should be" denied, let's treat the theoretical limit as 0
    # (The rules mention "DTI>40% => reduce by 50% or deny", or "score<500 => deny", etc.)
    # But we'll only do that if it's definitely a deny condition from above.
    if should_deny:
        theoretical_limit = 0.0

    # ------------------------------
    # 5) Compare row's actual approved_amount to theoretical
    # ------------------------------
    # If approved=1, we expect 0 < approved_amount <= theoretical_limit (within some tolerance).
    # If approved=0, we expect approved_amount == 0, or maybe near zero for noise.
    
    # We'll define a small epsilon for float comparisons:
    EPSILON = 1e-2  # $0.01

    # We'll gather results:
    errors = []

    # (a) Approved=0 => expect 0
    if approved == 0:
        if abs(app_amount) > EPSILON:
            errors.append(f"Row is denied (approved=0) but has nonzero approved_amount={app_amount:.2f}")
    else:  # approved=1
        if app_amount < -EPSILON:  # negative
            errors.append(f"Row is approved but has negative approved_amount={app_amount:.2f}")
        if app_amount > theoretical_limit + EPSILON:
            errors.append(
                f"Row has approved_amount={app_amount:.2f}, exceeds theoretical limit={theoretical_limit:.2f}"
            )

    # (b) Also note if there's a mismatch in whether we "should" approve vs. the row's actual.
    if should_approve and approved == 0:
        errors.append("Rules suggest approval, but row is denied.")
    if should_deny and approved == 1:
        errors.append("Rules suggest denial, but row is approved.")

    if not errors:
        return {
            "valid": True,
            "explanation": (
                f"Approved amount ${app_amount:.2f} is consistent with the rules. Theoretical limit=${theoretical_limit:.2f}."
            ),
        }
    else:
        return {
            "valid": False,
            "explanation": "; ".join(errors),
        }
    
def check_approved_amounts(df):
    """
    Iterates each row of df, runs validate_approved_amount, collects statistics,
    and prints a summary.
    """
    required = [
        "approved", "approved_amount", "annual_income", "credit_score",
        "credit_utilization", "payment_history", "employment_status",
        "months_employed", "self_reported_debt", "estimated_debt",
        "requested_amount"
    ]

    # Ensure columns exist
    missing = set(required) - set(df.columns)
    if missing:
        print(f"[ERROR] Missing columns for 'check_approved_amounts': {missing}")
        return

    # We'll track how many pass/fail
    total_checked = 0
    failures = 0

    # We'll store up to 5 error messages for demonstration
    sample_errors = []

    # For large DataFrames, consider df.itertuples() or df.iterrows()
    for idx, row in df.iterrows():
        result = validate_approved_amount(row)
        if not result["valid"]:
            failures += 1
            if len(sample_errors) < 5:
                sample_errors.append((idx, result["explanation"]))
        total_checked += 1

    print(f"\n[INFO] Completed check_approved_amounts on {total_checked} rows.")
    if failures == 0:
        print("[OK] No rule violations found for approved_amount.")
    else:
        frac_fail = failures / total_checked
        print(
            f"[WARNING] {failures} rows failed approved_amount validation "
            f"({frac_fail:.2%} of {total_checked})."
        )
        # Print a few sample error messages
        for i, msg in sample_errors:
            print(f"   Row {i}: {msg}")

def check_global_noise_and_missing(
    df,
    missing_range=(0.01, 0.02),
    noise_range=(0.05, 0.10),
    valid_ranges=None
):
    """
    Strictly validates that the dataset overall:
       - Has a missing fraction in [missing_range[0], missing_range[1]]
       - Has an out-of-range (noise) fraction in [noise_range[0], noise_range[1]]

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        missing_range (tuple): (min, max) fraction of missing data allowed. Default (0.01, 0.02) => 1–2%.
        noise_range (tuple): (min, max) fraction of out-of-range values allowed. Default (0.05, 0.10) => 5–10%.
        valid_ranges (dict): dictionary of {col_name: (vmin, vmax)} specifying the valid domain for each column.
                             We'll only check these columns for out-of-range "noise."

    Prints:
        - Overall missing fraction + pass/fail
        - Overall noise fraction + pass/fail
        - A breakdown (by column) of how many out-of-range values if desired
    """

    print("\n[INFO] === Global Noise & Missingness Validation ===")

    # ---------------------------------------
    # 1) Check Overall Missing Fraction
    # ---------------------------------------
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        print("[WARNING] DataFrame is empty (no cells). Can't validate missing/noise.")
        return

    total_cells = n_rows * n_cols
    total_missing = df.isna().sum().sum()  # total number of NaN cells
    missing_fraction = total_missing / total_cells

    min_miss, max_miss = missing_range
    missing_ok = (min_miss <= missing_fraction <= max_miss)

    print(f"[INFO] Overall missing fraction: {missing_fraction:.2%} (expected {min_miss:.2%}–{max_miss:.2%})")
    if missing_ok:
        print("[OK] Overall missing fraction is within the expected range.")
    else:
        print("[WARNING] Overall missing fraction is NOT in the [1%, 2%] guideline.")

    # ---------------------------------------
    # 2) Check Overall Noise Fraction
    # ---------------------------------------
    if not valid_ranges:
        print("[WARNING] No valid_ranges provided; skipping noise check.")
        return

    total_outliers = 0
    total_nonmissing = 0

    # We'll also keep a small breakdown for reference
    breakdown_info = {}

    for col, (vmin, vmax) in valid_ranges.items():
        if col not in df.columns:
            breakdown_info[col] = "[Missing column in DataFrame]"
            continue

        series = df[col].dropna()
        nonmissing_count = len(series)
        if nonmissing_count == 0:
            breakdown_info[col] = "[All NaN]"
            continue

        out_of_range_mask = (series < vmin) | (series > vmax)
        num_out = out_of_range_mask.sum()

        total_outliers += num_out
        total_nonmissing += nonmissing_count

        frac_out = num_out / nonmissing_count
        breakdown_info[col] = (
            f"Out-of-range: {num_out}/{nonmissing_count} => {frac_out:.2%} "
            f"(valid range [{vmin}, {vmax}])"
        )

    if total_nonmissing == 0:
        # Means everything was missing in the columns we track
        print("[WARNING] No non-missing data in the 'valid_ranges' columns => can't assess noise fraction.")
        return

    noise_fraction = total_outliers / total_nonmissing
    min_noise, max_noise = noise_range
    noise_ok = (min_noise <= noise_fraction <= max_noise)

    print(f"\n[INFO] Overall noise fraction: {noise_fraction:.2%} (expected {min_noise:.2%}–{max_noise:.2%})")
    if noise_ok:
        print("[OK] Overall noise fraction is within the expected range.")
    else:
        print("[WARNING] Overall noise fraction is NOT in the 5–10% guideline.")

    # Print a breakdown by column if you want
    print("\n[INFO] Noise breakdown by column:")
    for col, info in breakdown_info.items():
        print(f"   {col}: {info}")

    print("[INFO] === End Noise & Missingness Validation ===\n")

def check_total_credit_limit_correlation(df):
    """
    Validates that total_credit_limit is roughly 50–200% of annual_income
    for each row, i.e. ratio = total_credit_limit / annual_income is in [0.5, 2.0].

    Prints the fraction of rows that violate this condition, 
    plus a few sample violations for inspection.
    """
    required_cols = ["total_credit_limit", "annual_income"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing required columns for 'check_total_credit_limit_correlation': {missing}")
        return

    # Drop rows with missing total_credit_limit or annual_income
    valid_df = df.dropna(subset=required_cols)
    n_total = len(valid_df)
    if n_total == 0:
        print("[WARNING] No non-missing rows to check total_credit_limit correlation.")
        return

    # Compute ratio
    ratio = valid_df["total_credit_limit"] / valid_df["annual_income"]

    # We'll define the valid range as [0.5, 2.0]
    lower_bound, upper_bound = 0.5, 2.0
    in_range_mask = ratio.between(lower_bound, upper_bound)
    in_range_count = in_range_mask.sum()
    out_of_range_count = n_total - in_range_count
    frac_out = out_of_range_count / n_total

    print(f"\n[INFO] Checking total_credit_limit ~ 50–200% of annual_income:")
    print(f"   {in_range_count}/{n_total} rows in range, out-of-range fraction={frac_out:.2%}")

    # If you'd like to print sample violations
    # (e.g., up to 5 examples) for debugging:
    if out_of_range_count > 0:
        sample_violations = valid_df[~in_range_mask].head(5)
        print("[WARNING] Sample rows outside [0.5x, 2.0x] range:")
        for idx, row in sample_violations.iterrows():
            print(f"   Index={idx}, annual_income={row['annual_income']}, "
                  f"total_credit_limit={row['total_credit_limit']}, "
                  f"ratio={row['total_credit_limit']/row['annual_income']:.2f}")
            
def check_credit_utilization_distribution(df, col='credit_utilization', expected_mean=30.0, mean_tolerance=5.0):
    """
    Validates that credit_utilization (0–100%) is roughly ~30% on average.
    Also optionally does a K-S test comparing it to Beta(2,4) if scipy is available.

    Args:
        df (pd.DataFrame): The dataset containing credit_utilization.
        col (str): The column name. Default 'credit_utilization'.
        expected_mean (float): The approximate mean utilization (in %). Default 30%.
        mean_tolerance (float): If the sample mean differs from expected_mean by more 
                                than this, we warn. Default 5.0 => +/-5%.

    Prints:
        - The sample mean and how it compares to expected_mean.
        - A K-S test result (if scipy is installed), telling how well the data 
          matches Beta(2,4). This is purely optional and a rough indicator.
    """
    if col not in df.columns:
        print(f"[WARNING] '{col}' column not found.")
        return

    series = df[col].dropna()
    count = len(series)
    if count == 0:
        print(f"[WARNING] '{col}' has no non-missing values.")
        return

    # 1) Check the sample mean
    sample_mean = series.mean()
    diff = abs(sample_mean - expected_mean)
    print(f"[INFO] credit_utilization: sample mean={sample_mean:.2f}%, expected ~{expected_mean}%")
    if diff > mean_tolerance:
        print(f"[WARNING] credit_utilization mean differs from {expected_mean}% by more than {mean_tolerance}%.")
    else:
        print("[OK] credit_utilization mean is within tolerance.")

    # 2) (Optional) K-S test vs Beta(2,4) distribution
    #    Beta(2,4) has a mean of ~2/(2+4)=0.33 => 33%.
    #    But your data is 0–100%, so we must scale by /100 for the test.
    try:
        from scipy.stats import beta, kstest

        # scale to 0..1
        scaled = series / 100.0

        # K-S test comparing "scaled" data to Beta(2,4)
        # If p-value is high => we fail to reject that it *could* be Beta(2,4).
        # If p-value is low => the data likely doesn't follow Beta(2,4).
        alpha, bparam = 2, 4
        def cdf_beta(x): 
            return beta.cdf(x, alpha, bparam)

        ks_stat, p_value = kstest(scaled, cdf_beta)
        print(f"[INFO] K-S test vs Beta(2,4): ks_stat={ks_stat:.4f}, p-value={p_value:.4f}")
        if p_value < 0.05:
            print("[WARNING] 'credit_utilization' fails Beta(2,4) K-S test at 5% significance level.")
        else:
            print("[OK] 'credit_utilization' is not rejected as Beta(2,4) by K-S test.")
    except ImportError:
        print("[INFO] scipy not installed or unavailable; skipping K-S test.")

def check_estimated_debt_formula(df, tolerance_ratio=0.1):
    """
    Validates that estimated_debt is consistent with:
        estimated_debt = total_credit_limit * (credit_utilization / 100) * 0.03
    
    Args:
        df (pd.DataFrame): Must have columns:
            - 'total_credit_limit'
            - 'credit_utilization'
            - 'estimated_debt'
        tolerance_ratio (float): Allowed relative error, e.g. 0.1 => 10% mismatch allowed.

    Prints:
        - The fraction of rows that match the formula within 'tolerance_ratio'.
        - Some examples of rows that fail the check if any.
    """
    required = ["total_credit_limit", "credit_utilization", "estimated_debt"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns for check_estimated_debt_formula: {missing_cols}")
        return

    # Drop rows missing any of the required columns
    valid_df = df.dropna(subset=required)
    n_total = len(valid_df)
    if n_total == 0:
        print("[WARNING] No non-missing rows to validate estimated_debt formula.")
        return

    # Recompute the formula
    recomputed = (
        valid_df["total_credit_limit"] 
        * (valid_df["credit_utilization"] / 100.0) 
        * 0.03
    )
    actual = valid_df["estimated_debt"]

    # We'll define relative error = |recomputed - actual| / (|actual| + small_epsilon)
    small_epsilon = 1e-9
    rel_error = (recomputed - actual).abs() / (actual.abs() + small_epsilon)

    # "In tolerance" if relative error < tolerance_ratio
    in_tolerance_mask = (rel_error <= tolerance_ratio)
    in_tolerance_count = in_tolerance_mask.sum()
    out_of_tolerance_count = n_total - in_tolerance_count
    frac_out = out_of_tolerance_count / n_total

    print(f"\n[INFO] Checking estimated_debt formula among {n_total} valid rows.")
    print(f"[INFO] Tolerance ratio: {tolerance_ratio:.0%} (e.g. 0.1 => 10% mismatch allowed)")
    print(f"     {in_tolerance_count}/{n_total} rows match the formula, out-of-tolerance fraction={frac_out:.2%}")

    # Print some sample violations
    if out_of_tolerance_count > 0:
        sample_violations = valid_df[~in_tolerance_mask].head(5)
        print("[WARNING] Sample rows failing estimated_debt check:")
        for idx, row in sample_violations.iterrows():
            row_recomputed = (row["total_credit_limit"] 
                              * (row["credit_utilization"] / 100.0) 
                              * 0.03)
            row_error = abs(row_recomputed - row["estimated_debt"])
            row_rel_error = row_error / (abs(row["estimated_debt"]) + small_epsilon)
            print(
                f"   Index={idx}, total_credit_limit={row['total_credit_limit']}, "
                f"credit_utilization={row['credit_utilization']}, "
                f"estimated_debt(actual)={row['estimated_debt']:.2f}, "
                f"recomputed={row_recomputed:.2f}, "
                f"rel_error={row_rel_error:.2%}"
            )

def diagnose_approval_decisions(df):
    """
    For each row where approved == 0, determine the specific reason(s) for denial
    based on the following rules:
      - Deny if credit_score < 500.
      - Deny if DTI > 50%.
      - Deny if credit_utilization > 80%.
      - If none of these conditions hold but the row is still denied,
        then it falls into the ambiguous (grey) area where rules suggest approval.
    
    Prints a detailed per-row diagnosis and a summary count for each denial reason.
    """
    denial_counter = Counter()
    details = []

    # Iterate over rows that are denied
    for idx, row in df.iterrows():
        if row["approved"] == 0:
            reasons = []
            # Check conditions
            if row["credit_score"] < 500:
                reasons.append("Credit Score < 500")
            if row["DTI"] > 50.0:
                reasons.append("DTI > 50%")
            if row["credit_utilization"] > 80.0:
                reasons.append("Credit Utilization > 80%")
            
            # If none of the explicit denial conditions are met,
            # but the row is still denied, then it must have been randomly chosen in the grey area.
            if not reasons:
                # If the row meets the approval conditions (credit_score>=660, DTI<=40%, and payment_history = "On Time")
                # but is still denied, report that.
                pmt = str(row.get("payment_history", "")).lower()
                if row["credit_score"] >= 660 and row["DTI"] <= 40.0 and pmt in ["on time", "on-time", "on_time"]:
                    reasons.append("Rules suggest approval, but row is denied (grey area random choice)")
                else:
                    reasons.append("Other unspecified denial reason")
            
            # Update counter and details
            for r in reasons:
                denial_counter[r] += 1
            details.append((idx, reasons))

    # Print detailed diagnosis for each denied row
    print("\n[DIAGNOSTIC] Denial Reasons Per Row:")
    for idx, reason_list in details:
        print(f"   Row {idx}: {', '.join(reason_list)}")

    # Print summary counts
    print("\n[DIAGNOSTIC] Summary of Denial Reasons:")
    for reason, count in denial_counter.items():
        print(f"   {reason}: {count} occurrences")

# To use this function, call it after validation. For example, at the end of your main():
# diagnose_approval_decisions(df_merged)


def validate_dataset(csv_path):
    """
    Orchestrates the validation steps for the synthetic dataset.
    """
    # 1) Read the dataset
    df = read_dataset(csv_path)
    if df is None:
        return

    # 2) Check shape
    check_shape(df)

    # 3) Check required columns
    required_columns = [
        "applicant_id",
        "self_reported_expenses",
        "credit_score",
        "annual_income",
        "self_reported_debt",
        "requested_amount",
        "age",
        "province",
        "employment_status",
        "months_employed",
        "credit_utilization",
        "num_open_accounts",
        "num_credit_inquiries",
        "payment_history",
        "total_credit_limit",
        "monthly_expenses",
        "approved",
        "approved_amount",
        "estimated_debt",
        "interest_rate"
    ]
    check_required_columns(df, required_columns)

    # 4) Check data types
    # 'O' means object/string, 'f' means float, 'i' means int
    expected_dtypes = {
        "applicant_id": "O",
        "self_reported_expenses": "f",
        "credit_score": "f",
        "annual_income": "f",
        "self_reported_debt": "f",
        "requested_amount": "f",
        "age": "f",
        "province": "O",
        "employment_status": "O",
        "months_employed": "f",
        "credit_utilization": "f",
        "num_open_accounts": "f",
        "num_credit_inquiries": "f",
        "payment_history": "O",
        "total_credit_limit": "f",
        "monthly_expenses": "f",
        "approved": "f",
        "approved_amount": "f",
        "estimated_debt": "f",
        "interest_rate": "f"
    }
    check_data_types(df, required_columns, expected_dtypes)

    # 5) Check numeric ranges
    check_numeric_ranges(df)

    # 6) Check categorical values
    check_categorical_values(df)

    # 7) Check distribution of approved
    check_approved_distribution(df)

    # 8) Check missing data
    check_missing_data(df)

     # 9) Check credit_score distribution
    check_credit_score_distribution(df)

    # 10) Check annual_income distribution (NEW)
    check_annual_income_distribution(df)

    # 10) Check monthly_debt ratio (NEW)
    check_monthly_debt_ratio(df)

    # Finally, run the new check:
    check_approved_amounts(df)

    # Suppose we define the valid ranges for numeric columns:
    col_ranges = {
        "credit_score": (300, 900),
        "annual_income": (20000, 200000),
        "self_reported_debt": (0, 10000),
        "requested_amount": (1000, 50000),
        "months_employed": (0, 600),
        "credit_utilization": (0, 100),
        "num_open_accounts": (0, 20),
        "num_credit_inquiries": (0, 10),
        "total_credit_limit": (0, 50000),
        "monthly_expenses": (0, 10000),
        "approved_amount": (0, 50000),
        "estimated_debt": (0, 10000),
        "interest_rate": (3.0, 15.0),
        "self_reported_expenses": (0, 10000)

    }

    check_global_noise_and_missing(
        df,
        missing_range=(0.01, 0.02),  # i.e. 1–2%
        noise_range=(0.05, 0.10),    # i.e. 5–10%
        valid_ranges=col_ranges
    )

    check_total_credit_limit_correlation(df)

    # Validate credit_utilization distribution
    check_credit_utilization_distribution(
        df, 
        col='credit_utilization', 
        expected_mean=30.0,   # 30%
        mean_tolerance=5.0    # +/- 5% is okay
    )

    # Now check the estimated_debt formula
    check_estimated_debt_formula(
        df,
        tolerance_ratio=0.1  # 10% mismatch allowed
    )

    diagnose_approval_decisions(df)


    # 9) Done
    print("\n[INFO] Validation complete. Review any warnings above for issues.\n")

if __name__ == "__main__":
    # Name of your log file
    log_file = "validation_output.log"

    # Open the file for writing, redirect sys.stdout to it
    with open(log_file, "w") as f:
        old_stdout = sys.stdout
        try:
            # Redirect all print statements to f
            sys.stdout = f
            
            # Now call your validation function so all prints go to the file
            validate_dataset("synthetic_loan_applications.csv")

            print("\n[INFO] Validation completed; all output in this log file.")
        finally:
            # Restore original stdout so further prints (if any) go back to terminal
            sys.stdout = old_stdout
