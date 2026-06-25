# =====================================
# UNEMPLOYMENT ANALYSIS IN INDIA
# =====================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------
# Load Cleaned Dataset
# -------------------------------------

df = pd.read_csv("Unemployment_in_India_Cleaned_dataset.csv")

print("Dataset Loaded Successfully")

# -------------------------------------
# Basic Information
# -------------------------------------

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------------
# Data Preprocessing
# -------------------------------------

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# -------------------------------------
# Statistical Summary
# -------------------------------------

print("\nStatistical Summary")
print(df.describe())

# -------------------------------------
# Average Unemployment by State
# -------------------------------------

state_unemployment = df.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values(ascending=False)

print("\nAverage Unemployment Rate by State:")
print(state_unemployment)

# -------------------------------------
# Visualization 1
# Top 10 States
# -------------------------------------

plt.figure(figsize=(12,6))

state_unemployment.head(10).plot(
    kind='bar'
)

plt.title("Top 10 States with Highest Unemployment Rate")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("State")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------------------
# Visualization 2
# Monthly Trend
# -------------------------------------

monthly_trend = df.groupby('Date')[
    'Estimated Unemployment Rate (%)'
].mean()

plt.figure(figsize=(12,6))

plt.plot(
    monthly_trend.index,
    monthly_trend.values,
    marker='o'
)

plt.title("Monthly Unemployment Trend")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate")

plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------------
# Visualization 3
# Urban vs Rural
# -------------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x='Area',
    y='Estimated Unemployment Rate (%)'
)

plt.title("Urban vs Rural Unemployment Rate")

plt.tight_layout()
plt.show()

# -------------------------------------
# Covid-19 Analysis
# -------------------------------------

covid = df[df['Year'] == 2020]

plt.figure(figsize=(12,6))

sns.lineplot(
    data=covid,
    x='Date',
    y='Estimated Unemployment Rate (%)',
    marker='o'
)

plt.title("Covid-19 Impact on Unemployment (2020)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# -------------------------------------
# Heatmap
# -------------------------------------

heatmap_data = df.pivot_table(
    values='Estimated Unemployment Rate (%)',
    index='Region',
    columns='Year',
    aggfunc='mean'
)

plt.figure(figsize=(10,8))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap='YlOrRd'
)

plt.title("State-wise Unemployment Heatmap")

plt.tight_layout()
plt.show()

# -------------------------------------
# Correlation Matrix
# -------------------------------------

plt.figure(figsize=(8,5))

numeric_columns = [
    'Estimated Unemployment Rate (%)',
    'Estimated Employed',
    'Estimated Labour Participation Rate (%)'
]

sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# -------------------------------------
# Important Insights
# -------------------------------------

highest_state = state_unemployment.idxmax()
highest_rate = state_unemployment.max()

lowest_state = state_unemployment.idxmin()
lowest_rate = state_unemployment.min()

print("\n============================")
print("PROJECT INSIGHTS")
print("============================")

print(f"Highest unemployment state: {highest_state}")
print(f"Average unemployment rate: {highest_rate:.2f}%")

print(f"\nLowest unemployment state: {lowest_state}")
print(f"Average unemployment rate: {lowest_rate:.2f}%")

print("\nCovid-19 period shows a sharp rise in unemployment.")
print("Urban and Rural regions show different unemployment patterns.")
print("Employment level and labour participation influence unemployment.")
