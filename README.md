# 🛡️ DigiLife Insurance — AI-Powered Analytics Dashboard

A data-driven decision-making platform for a digital-first life insurance startup in India.
Built on a synthetic consumer survey dataset of 1,500 respondents across 5 life insurance products.

---

## 📦 Products Covered

| # | Product | Target Segment |
|---|---------|---------------|
| 1 | **Term Life Insurance** | Salaried urban, 25–45 yrs |
| 2 | **Credit Life / Loan Protection** | Loan holders via NBFC/bank partners |
| 3 | **Whole Life Insurance** | HNI, Business Owners, 40+ age group |
| 4 | **Child Education & Future Plan** | Married parents, 28–42 yrs |
| 5 | **Group Term Life** | SME owners, corporates |

---

## 🗂️ Repository Structure

```
├── app.py                        # Main Streamlit dashboard
├── generate_survey_data.py       # Synthetic dataset generator
├── digital_insurance_survey.csv  # Pre-generated dataset (1,500 rows × 44 cols)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/digilife-insurance-analytics.git
cd digilife-insurance-analytics
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## 📊 Dashboard Tabs

### 📊 Overview
- KPI metrics: purchase rate, avg premium, churn rate, avg LTV
- Product mix donut chart
- Product interest by income band (line chart)
- Churn rate by distribution channel
- Age × Income LTV heatmap
- Full correlation matrix of key variables

### 🎯 Classification
Three tasks selectable via radio button:
- **Buy Propensity** — Will this customer purchase a policy?
- **Product Recommendation** — Which product is best suited for them?
- **Churn Prediction** — Will this customer leave within 12 months?

Models available: Random Forest, Logistic Regression, Gradient Boosting  
Outputs: Accuracy, AUC-ROC, Confusion Matrix, Feature Importance chart, Business Insight

### 🔵 Clustering (Customer Segmentation)
- K-Means clustering with configurable K (2–8)
- Elbow chart + Silhouette score validation
- Cluster profile table with churn %, top product, income, age
- Scatter plot: Income vs WTP coloured by cluster
- **Personalised strategy per segment** — discount type, channel, bundle recommendation, premium mode

### 🔗 Association Rule Mining
- Apriori algorithm on product interest and customer profile attributes
- Configurable: Min Support, Min Confidence, Min Lift
- Two analysis modes: Product Cross-Sell Rules | Customer Profile → Product Rules
- Confidence vs Lift scatter plot
- Actionable bundling recommendations

### 📈 Regression
Three targets:
- **Customer LTV (₹)** — Predict lifetime value for acquisition budget allocation
- **Annual Premium Pricing (₹)** — Data-driven premium quote engine
- **Satisfaction Score (1–5)** — Early warning system for at-risk customers

Models: Gradient Boosting, Linear Regression, Random Forest  
Outputs: R², MAE, MAPE, Actual vs Predicted scatter, Residual distribution

---

## 📁 Dataset — Column Reference

| Column Group | Columns |
|---|---|
| **Identifiers** | Customer_ID, Survey_Date |
| **Demographics** | Age, Gender, City_Tier, State, Occupation, Education, Marital_Status, Dependents |
| **Financial** | Annual_Income_INR, Existing_Loans, Loan_Amount_INR, Has_Savings, Monthly_Savings_INR, Existing_Insurance |
| **Risk & Awareness** | Risk_Appetite, Financial_Literacy (1–5), Awareness_Term/CreditLife/WholeLife/ChildPlan/GroupTerm |
| **Health & Lifestyle** | BMI, Smoker, Pre_Existing_Cond, Exercise_Frequency |
| **Digital Behaviour** | Digital_Savvy_Score (1–5), Preferred_Channel, Social_Media_Use, Online_Purchase_Hist |
| **Willingness to Pay** | WTP_Monthly_INR, Price_Sensitivity |
| **Product Interest** | Interest_TermLife/CreditLife/WholeLife/ChildPlan/GroupTerm (1–5 scale) |
| **Outcomes** | Product_Purchased, Annual_Premium_INR, Churned, Satisfaction_Score, Referral_Likelihood, Customer_LTV_INR |

---

## 🔧 Regenerate the Dataset

To generate a fresh synthetic dataset (different random seed etc.):

```bash
python generate_survey_data.py
```

This will overwrite `digital_insurance_survey.csv` with a new 1,500-row dataset.

---

## 🧠 ML Algorithms Used

| Algorithm | Library | Use Case |
|---|---|---|
| Random Forest Classifier | scikit-learn | Buy propensity, product recommendation, churn |
| Logistic Regression | scikit-learn | Buy propensity, churn (interpretable baseline) |
| Gradient Boosting Classifier | scikit-learn | Churn prediction (high accuracy) |
| K-Means Clustering | scikit-learn | Customer segmentation |
| Apriori + Association Rules | mlxtend | Product bundling, cross-sell |
| Gradient Boosting Regressor | scikit-learn | LTV, premium pricing, satisfaction |
| Linear Regression | scikit-learn | Premium pricing (interpretable baseline) |

---

## 📋 Key Business Objectives

1. **Target the right customer** — Classification models identify high-propensity buyers by segment
2. **Personalised discounts & bundles** — Clustering + Association Rules drive tailored offers
3. **Reduce churn** — Churn prediction flags at-risk customers for proactive retention
4. **Dynamic premium pricing** — Regression model powers income-tiered quote engine
5. **Maximise LTV** — LTV prediction informs acquisition budget allocation by segment

---

## 👤 Author

**Abhay Nath Jha**  
MD & CEO, Star Union Dai-ichi Life Insurance Co. Ltd.  
DBA Program — Emerging Technologies (Gen AI), Golden Gate University, USA  
📧 tutuabhay@gmail.com

---

## ⚠️ Disclaimer

This project uses entirely **synthetic data** generated for academic and research purposes.
No real customer data has been used. All distributions are calibrated to publicly available
Indian insurance market benchmarks (IRDAI Annual Reports, Swiss Re Sigma).

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
