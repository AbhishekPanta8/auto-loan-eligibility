# Credit Score API Integration for TD Bank

This project integrates credit score checking functionality into the **TD Bank** application. The solution evaluates two key API providers: **Flinks Credit Builder API** and **Inverite (Upstart)**, as well as the existing credit score checking technology already in place at TD Bank, which uses **TransUnion's CreditView® Dashboard**.

## Existing Technology

### **TD Bank - Credit Score Check**
TD Bank customers can check their credit score via the [TD Bank Credit Score page](https://www.td.com/ca/en/personal-banking/products/borrowing/check-your-credit-score), which uses TransUnion’s **CreditView® Dashboard**. The TD app provides customers with their credit score without affecting their score (through a soft inquiry), and it refreshes on a weekly basis. This functionality is intended for educational purposes and is not used for direct financial decisions.

### **TransUnion CreditView®**
TD Bank leverages **TransUnion's CreditView®** to provide customers with their credit score, which is updated weekly. This dashboard helps users better understand their financial standing but is not directly used for loan approvals. The **soft inquiry** ensures that checking the score does not negatively impact the customer's credit score.

For more information on TransUnion's CreditView® service, visit: [TransUnion CreditView](https://www.transunion.ca/product/creditview)

---

## API Options for Credit Score Integration

The goal of this project is to integrate credit score checking functionality into TD Bank's existing system for line of credit applications. We are evaluating the following API providers:

### **1. Flinks Credit Builder API**
- **Primary Focus**: Provides access to financial data aggregation, including credit score from **TransUnion** and **Equifax**.
- **Key Features**:
  - Offers financial insights, credit scores, and transaction data.
  - Integrates directly with major Canadian credit bureaus.
  - Developer-friendly, with a **self-service sandbox** for testing and immediate access to the API.
  - Ideal for credit risk assessments and standard line of credit applications.
  - **Compliant** with Canadian financial regulations.
- **Pricing**: Paid API with clear pricing models, and sandbox access for development.
- **Documentation**: Public API documentation available [here](https://docs.flinks.com/).
  
**Why Flinks**: Flinks is the **best option** for fast integration, clear documentation, and full compliance with Canadian financial norms. It aligns well with TD Bank’s existing use of TransUnion and Equifax for credit score verification.

---

### **2. Inverite (Upstart)**
- **Primary Focus**: Specializes in credit verification using **alternative data sources** (e.g., rent payments, payroll), in addition to traditional credit bureau data.
- **Key Features**:
  - Offers both **credit score** and **alternative credit data**, which can be valuable for non-prime or subprime customers.
  - Provides APIs for credit score verification and financial verification.
  - **Compliance** with Canadian regulations but is more suited for alternative lending scenarios.
- **Pricing**: Paid API, pricing structure is less transparent and requires engagement with the provider.
- **Documentation**: Requires onboarding before gaining access to API documentation.

**Why Inverite**: Inverite is suitable if you are looking to integrate **alternative credit data** into your line of credit application process. It is more focused on non-traditional credit assessments and might be beneficial for subprime customers.

---

## Comparison: Flinks vs Inverite

| Factor                    | **Flinks**                                   | **Inverite (Upstart)**                             |
|---------------------------|----------------------------------------------|---------------------------------------------------|
| **Speed of Implementation**| ✅ Fast setup, sandbox access               | ❌ Requires onboarding steps                     |
| **Data Coverage**          | ✅ Bureau-based (TransUnion, Equifax)       | ✅ Alternative data (rent, payroll, etc.)         |
| **Use Case Fit**           | ✅ Ideal for prime credit applications      | ✅ Better for alternative/subprime lending        |
| **Regulatory Compliance**  | ✅ Fully compliant with Canadian norms      | ✅ Compliant, but more suited for alternative lending |
| **Cost & API Access**      | ❌ Paid API with clear pricing models       | ❌ Paid API, less transparent pricing structure   |
| **Market Adoption**        | ✅ Widely adopted in Canada                 | ⚠️ Less clear after Upstart acquisition          |

---

## Recommendation

### **Flinks**
- **Best fit for the TD Bank project**, as it provides quick, reliable access to credit scores and aligns with TD's current use of **TransUnion and Equifax**. It offers a fast implementation process, clear API documentation, and complies with Canadian regulations, making it ideal for integrating into TD's line of credit application process.

- **Pros**:
  - Fast integration with sandbox access.
  - Fully compatible with existing financial infrastructure at TD (TransUnion and Equifax).
  - Established and trusted in the Canadian market.
  
- **Cons**:
  - Paid API access.

### **Inverite (Upstart)**
- **Consider Inverite** if you're focusing on using **alternative credit data** for assessing non-prime or subprime applicants. It offers valuable features for more diverse credit scoring models but may not align as well with TD’s existing infrastructure for standard line of credit applications.

- **Pros**:
  - Strong in alternative lending and credit verification.
  
- **Cons**:
  - Slower implementation process.
  - Less adoption in traditional financial institutions like TD.

---



# Flinks
## Frontend
https://docs.flinks.com/docs/choose-a-front-end-solution
Set Up a Custom API Integration - Only to show TD bank for login, not recommended as we will limit our users   
Using Flinks Connect or Using Flinks Express
## Flinks Connect
https://docs.flinks.com/docs/set-up-flinks-connect
https://docs.flinks.com/docs/connect-new-accounts

## Conclusion

- **Flinks**: Review the [API documentation](https://docs.flinks.com/) and start the integration process. Flinks is the preferred choice due to the speed of implementation and alignment with TD's current systems.
- **Inverite (Upstart)**: If alternative credit models are desired, contact Inverite for onboarding and access to their API documentation.
