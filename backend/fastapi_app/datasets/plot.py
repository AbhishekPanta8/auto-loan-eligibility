#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_dti_diagnostics(df):
    """
    Generates histograms for:
      - annual_income
      - monthly_income
      - DTI
    Saves them to PNG files.
    """
    # Ensure monthly_income exists; if not, compute it.
    if "monthly_income" not in df.columns:
        df["monthly_income"] = df["annual_income"] / 12.0

    # Plot Annual Income histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df["annual_income"].dropna(), bins=50, color='green', alpha=0.7)
    plt.title("Annual Income Distribution")
    plt.xlabel("Annual Income")
    plt.ylabel("Frequency")
    plt.savefig("annual_income_histogram.png")
    plt.close()

    # Plot Monthly Income histogram
    plt.figure(figsize=(8, 6))
    plt.hist(df["monthly_income"].dropna(), bins=50, color='orange', alpha=0.7)
    plt.title("Monthly Income Distribution")
    plt.xlabel("Monthly Income")
    plt.ylabel("Frequency")
    plt.savefig("monthly_income_histogram.png")
    plt.close()

    # Plot DTI histogram
    if "DTI" in df.columns:
        plt.figure(figsize=(8, 6))
        plt.hist(df["DTI"].dropna(), bins=50, color='blue', alpha=0.7)
        plt.title("DTI Distribution")
        plt.xlabel("DTI (%)")
        plt.ylabel("Frequency")
        plt.savefig("DTI_histogram.png")
        plt.close()
    else:
        print("[WARNING] DTI column not found; cannot plot DTI histogram.")

    print("[INFO] Histograms saved: 'annual_income_histogram.png', 'monthly_income_histogram.png', 'DTI_histogram.png'.")

def additional_dti_analysis(df):
    """
    Adds deeper diagnostics to understand high DTI:
      1) Scatter plot of DTI vs monthly_income
      2) Show sample of rows with DTI>50
      3) Compare monthly_income and DTI for approved=0 vs. approved=1 via boxplots
      4) Check for extremely low or high annual_income values that might be from noise
    Saves additional plots and prints sample data for inspection.
    """

    # Ensure monthly_income exists
    if "monthly_income" not in df.columns:
        df["monthly_income"] = df["annual_income"] / 12.0

    # 1) Scatter plot: DTI vs monthly_income
    if "DTI" in df.columns:
        plt.figure(figsize=(8,6))
        plt.scatter(df["monthly_income"], df["DTI"], alpha=0.5, color="purple")
        plt.title("DTI vs. Monthly Income")
        plt.xlabel("Monthly Income")
        plt.ylabel("DTI (%)")
        plt.savefig("DTI_vs_monthly_income.png")
        plt.close()
        print("[INFO] Scatter plot saved: 'DTI_vs_monthly_income.png'")
    else:
        print("[WARNING] 'DTI' column not found; skipping DTI vs monthly_income scatter plot.")

    # 2) Show sample of rows with DTI>50
    if "DTI" in df.columns:
        high_dti_df = df[df["DTI"] > 50].copy()
        print("\n[INFO] Sample rows with DTI>50 (top 20):")
        cols_to_show = ["annual_income", "monthly_income", "requested_amount",
                        "self_reported_debt", "estimated_debt", "DTI"]
        print(high_dti_df[cols_to_show].head(20))
    else:
        print("[WARNING] 'DTI' column not found; cannot filter DTI>50.")

    # 3) Compare monthly_income/DTI for approved=0 vs. approved=1
    if "approved" in df.columns:
        # Boxplot: monthly_income by approval
        plt.figure(figsize=(8,6))
        sns.boxplot(x="approved", y="monthly_income", data=df, palette="Set2")
        plt.title("Monthly Income by Approval Status")
        plt.savefig("monthly_income_approved_box.png")
        plt.close()
        print("[INFO] Boxplot saved: 'monthly_income_approved_box.png'")

        # Boxplot: DTI by approval
        if "DTI" in df.columns:
            plt.figure(figsize=(8,6))
            sns.boxplot(x="approved", y="DTI", data=df, palette="Set2")
            plt.title("DTI by Approval Status")
            plt.savefig("DTI_approved_box.png")
            plt.close()
            print("[INFO] Boxplot saved: 'DTI_approved_box.png'")
    else:
        print("[WARNING] 'approved' column not found; cannot do boxplots by approval status.")

    # 4) Check for extremely low or high annual_income values (possible noise)
    noise_suspects = df[(df["annual_income"] < 20000) | (df["annual_income"] > 200000)]
    if len(noise_suspects) > 0:
        print("\n[INFO] Suspected noise-based incomes (<20k or >200k). Showing top 20:")
        print(noise_suspects[["annual_income", "monthly_income", "DTI", "approved"]].head(20))
    else:
        print("\n[INFO] No rows found with annual_income <20k or >200k. Possibly no extreme noise values found.")


def main():
    try:
        df = pd.read_csv("synthetic_loan_applications.csv")
    except Exception as e:
        print(f"[ERROR] Could not read 'synthetic_loan_applications.csv': {e}")
        return

    # 1) Basic histograms
    plot_dti_diagnostics(df)

    # 2) Additional DTI analysis (scatter plot, high DTI rows, boxplots, noise checks)
    additional_dti_analysis(df)

    print("\n[INFO] Additional DTI diagnostics complete. Check the printed info and PNG plots.")

if __name__ == "__main__":
    main()
