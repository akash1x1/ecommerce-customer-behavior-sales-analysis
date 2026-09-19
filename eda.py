# ecommerce customer behavior and sales analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load dataset
df = pd.read_csv("ecommerce_customer_behavior_eda.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nShape of dataset (rows, columns):", df.shape)

print("\nInfo about the dataset:")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist())

# numerical and categorical columns
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

print("\nNumerical columns:", numerical_cols)
print("Categorical columns:", categorical_cols)

print("\nUnique customers:", df["Customer_ID"].nunique())
print("Unique orders:", df["Order_ID"].nunique())
print("Unique product categories:", df["Product_Category"].nunique())
print("Unique cities:", df["City"].nunique())
print("Unique payment methods:", df["Payment_Method"].nunique())
print("Unique device types:", df["Device_Type"].nunique())

# convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"])
print("\nDate range: from", df["Date"].min(), "to", df["Date"].max())

# data quality check
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nTotal duplicate rows:", df.duplicated().sum())
print("Duplicate Order_IDs:", df["Order_ID"].duplicated().sum())

print("\nAny negative or zero Unit_Price?", (df["Unit_Price"] <= 0).sum())
print("Any negative or zero Quantity?", (df["Quantity"] <= 0).sum())
print("Any negative or zero Total_Amount?", (df["Total_Amount"] <= 0).sum())

print("\nCategorical value checks:")
print("Gender values:", df["Gender"].unique())
print("Payment_Method values:", df["Payment_Method"].unique())
print("Device_Type values:", df["Device_Type"].unique())
print("Customer_Rating values:", sorted(df["Customer_Rating"].unique()))

# no missing values or duplicates found, so no cleaning needed
# just add year/month columns for later date analysis
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()

# descriptive statistics
print("\nDescribe() on numerical columns:")
print(df.describe())

print("\nTotal_Amount -> Mean:", round(df["Total_Amount"].mean(), 2),
      " Median:", df["Total_Amount"].median())
print("Age -> Mean:", round(df["Age"].mean(), 2), " Median:", df["Age"].median())

# age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Number of Transactions")
plt.show()

# total amount distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Total_Amount"], bins=40, kde=True)
plt.title("Distribution of Total Transaction Amount")
plt.xlabel("Total Amount")
plt.ylabel("Number of Transactions")
plt.show()

# product category counts
plt.figure(figsize=(9, 5))
sns.countplot(y="Product_Category", data=df,
              order=df["Product_Category"].value_counts().index)
plt.title("Number of Transactions per Product Category")
plt.xlabel("Number of Transactions")
plt.ylabel("Product Category")
plt.show()

# gender distribution
plt.figure(figsize=(6, 5))
sns.countplot(x="Gender", data=df)
plt.title("Number of Transactions by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Transactions")
plt.show()

# payment method counts
plt.figure(figsize=(8, 5))
sns.countplot(y="Payment_Method", data=df,
              order=df["Payment_Method"].value_counts().index)
plt.title("Number of Transactions by Payment Method")
plt.xlabel("Number of Transactions")
plt.ylabel("Payment Method")
plt.show()

# device type counts
plt.figure(figsize=(6, 5))
sns.countplot(x="Device_Type", data=df,
              order=df["Device_Type"].value_counts().index)
plt.title("Number of Transactions by Device Type")
plt.xlabel("Device Type")
plt.ylabel("Number of Transactions")
plt.show()

# customer rating counts
plt.figure(figsize=(6, 5))
sns.countplot(x="Customer_Rating", data=df, order=sorted(df["Customer_Rating"].unique()))
plt.title("Distribution of Customer Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Transactions")
plt.show()

# boxplot of total amount (also used later for outlier check)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Total_Amount"])
plt.title("Boxplot of Total Transaction Amount")
plt.xlabel("Total Amount")
plt.show()

# age vs total amount
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Age", y="Total_Amount", data=df, alpha=0.4)
plt.title("Age vs Total Transaction Amount")
plt.xlabel("Age")
plt.ylabel("Total Amount")
plt.show()

# quantity vs total amount
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Quantity", y="Total_Amount", data=df, alpha=0.4)
plt.title("Quantity vs Total Transaction Amount")
plt.xlabel("Quantity")
plt.ylabel("Total Amount")
plt.show()

# customer behavior: returning vs new customers
print("\nReturning vs New customers (% of transactions):")
print(df["Is_Returning_Customer"].value_counts(normalize=True) * 100)

print("\nAverage Total_Amount per transaction (returning vs new):")
print(df.groupby("Is_Returning_Customer")["Total_Amount"].mean())

# customer-level spending (groups all orders of the same customer together)
customer_spending = df.groupby("Customer_ID")["Total_Amount"].sum()
print("\nCustomer-level total spending - summary statistics:")
print(customer_spending.describe())

print("\nTop 5 customers by total spending:")
print(customer_spending.sort_values(ascending=False).head())

orders_per_customer = df.groupby("Customer_ID")["Order_ID"].count()
print("\nAverage number of orders per customer:", round(orders_per_customer.mean(), 2))

# spending by age group
df["Age_Group"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 100],
                          labels=["18-25", "26-35", "36-45", "46-55", "56+"])
print("\nAverage Total_Amount by Age Group:")
print(df.groupby("Age_Group")["Total_Amount"].mean())

# product and sales analysis
sales_by_category = df.groupby("Product_Category")["Total_Amount"].sum().sort_values(ascending=False)
print("\nTotal sales by Product Category:")
print(sales_by_category)

avg_value_by_category = df.groupby("Product_Category")["Total_Amount"].mean().sort_values(ascending=False)
print("\nAverage transaction value by Product Category:")
print(avg_value_by_category)

qty_by_category = df.groupby("Product_Category")["Quantity"].sum().sort_values(ascending=False)
print("\nTotal quantity sold by Product Category:")
print(qty_by_category)

# bar chart of total sales by category
plt.figure(figsize=(9, 5))
sns.barplot(x=sales_by_category.values, y=sales_by_category.index)
plt.title("Total Sales by Product Category")
plt.xlabel("Total Sales")
plt.ylabel("Product Category")
plt.show()

# payment and device analysis
print("\nMost common payment method:")
print(df["Payment_Method"].value_counts())

print("\nTotal sales by Payment Method:")
print(df.groupby("Payment_Method")["Total_Amount"].sum().sort_values(ascending=False))

print("\nMost common device type:")
print(df["Device_Type"].value_counts())

print("\nAverage transaction value by Device Type:")
print(df.groupby("Device_Type")["Total_Amount"].mean().sort_values(ascending=False))

# date/time analysis
monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Total_Amount"].sum()
monthly_orders = df.groupby(df["Date"].dt.to_period("M"))["Order_ID"].count()

print("\nMonthly sales:")
print(monthly_sales)
print("\nHighest sales month:", monthly_sales.idxmax(), "with", round(monthly_sales.max(), 2))

# line chart of monthly sales
plt.figure(figsize=(12, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Total Sales Over Time")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# iqr and outlier analysis on total amount
Q1 = df["Total_Amount"].quantile(0.25)
Q3 = df["Total_Amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nQ1:", Q1, " Q3:", Q3, " IQR:", IQR)
print("Lower bound:", lower_bound, " Upper bound:", upper_bound)

outliers = df[(df["Total_Amount"] < lower_bound) | (df["Total_Amount"] > upper_bound)]
print("Number of potential outliers:", len(outliers))
print("Percentage of total data:", round(len(outliers) / len(df) * 100, 2), "%")

# skewness analysis
print("\nSkewness of Total_Amount:", round(df["Total_Amount"].skew(), 2))
print("Skewness of Unit_Price:", round(df["Unit_Price"].skew(), 2))
print("Skewness of Age:", round(df["Age"].skew(), 2))
print("Skewness of Session_Duration_Minutes:", round(df["Session_Duration_Minutes"].skew(), 2))

# correlation analysis
corr_cols = ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
             "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"]
correlation_matrix = df[corr_cols].corr()

print("\nCorrelation with Total_Amount:")
print(correlation_matrix["Total_Amount"].sort_values(ascending=False))

# heatmap of correlation
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Numerical Columns")
plt.tight_layout()
plt.show()

# key insights
print("\nKEY INSIGHTS")
print("""
1. Electronics generated the highest total sales overall, even though
   it did not have the highest number of transactions - its average
   transaction value is much higher than every other category.
2. Sports had the highest transaction COUNT, showing that transaction
   count and total sales tell different stories and should not be
   confused.
3. Credit Card is the most commonly used payment method.
4. Mobile is the most used device for transactions, though Desktop
   has a very slightly higher average transaction value.
5. About 88% of transactions came from returning customers.
6. Interestingly, returning customers spend a very similar (slightly
   lower) amount per transaction on average compared to new customers.
7. Total_Amount is strongly right-skewed - most transactions are of
   low/medium value, with a smaller number of high-value transactions
   pulling the average upward.
8. Around 11% of transactions were flagged as statistical outliers on
   the high end, most likely genuine high-value purchases rather than
   data errors.
9. Total_Amount is strongly driven by Unit_Price and, to a lesser
   extent, Quantity and Discount_Amount - but shows almost no
   relationship with Age, Session_Duration_Minutes, Pages_Viewed or
   Customer_Rating.
10. December had the highest monthly sales, which could suggest a
    seasonal/holiday shopping effect, though month-to-month sales are
    fairly stable overall.
""")

# conclusion
print("CONCLUSION")
print("""
This project analyzed an e-commerce transactions dataset of 17,049
orders from 5,000 customers to understand customer behavior and sales
patterns. The data was found to be clean, with no missing values or
duplicates. Electronics stands out as the biggest revenue driver due
to high transaction values, while Sports sees the most frequent
purchases. Most customers are returning customers, though this alone
does not translate into meaningfully higher per-order spending.
Total_Amount is right-skewed with genuine high-value outliers, and is
mainly explained by pricing and quantity rather than customer
demographics or on-site browsing behavior. These findings could help
an e-commerce business decide where to focus marketing (e.g. high
sales categories), understand which channels/devices drive value, and
recognize that spending patterns are more closely tied to what is
being bought than who is buying it.
""")
