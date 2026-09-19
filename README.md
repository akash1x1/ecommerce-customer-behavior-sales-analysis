# E-Commerce Customer Behavior and Sales Analysis using Python

## About
This is an Exploratory Data Analysis (EDA) project on an e-commerce
transactions dataset. It uses Python, Pandas, NumPy, Matplotlib and
Seaborn to understand customer behavior and sales patterns. No
machine learning is used - this is a pure EDA project.

## Dataset
`ecommerce_customer_behavior_eda.csv`
- 17,049 rows (transactions), 18 columns
- 5,000 unique customers
- Date range: January 2023 to March 2024
- No missing values, no duplicate rows

## Files
- `eda.py` - the full analysis (single file)
- `ecommerce_customer_behavior_eda.csv` - the dataset
- `README.md` - this file

## How to Run
1. Install the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn
   ```
2. Run the script:
   ```
   python eda.py
   ```
3. Each chart will open in its own window. Close a chart window to
   move on to the next one. All printed output (statistics, insights)
   will appear in the terminal.

## What the Analysis Covers
1. Data loading and basic understanding
2. Data quality checks (missing values, duplicates)
3. Descriptive statistics (mean, median, std, quartiles)
4. Univariate analysis (age, sales amount, categories, payment
   methods, devices, ratings)
5. Bivariate analysis (age vs spending, quantity vs spending)
6. Customer behavior (returning vs new customers, top customers,
   transaction-level vs customer-level spending)
7. Product and sales analysis (total sales vs average transaction
   value per category)
8. Payment and device analysis
9. Date/time analysis (monthly sales trend)
10. IQR-based outlier analysis on transaction amount
11. Skewness analysis
12. Correlation analysis
13. Key insights and conclusion

## Key Findings (short version)
- Electronics generates the highest total sales despite not having
  the most transactions - its average order value is much higher.
- About 88% of transactions come from returning customers.
- Transaction amount is right-skewed, with ~11% of transactions
  flagged as high-value outliers (likely genuine, not data errors).
- Transaction amount is strongly related to price and quantity, but
  not to customer age, session duration, or rating.
- December had the highest monthly sales.
