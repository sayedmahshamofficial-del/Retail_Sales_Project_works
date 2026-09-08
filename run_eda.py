import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["font.size"] = 10

IMG = "/home/claude/retail_project/images"
DATA = "/home/claude/retail_project/data/retail_sales_dataset.csv"

df = pd.read_csv(DATA, parse_dates=["Date"])

before_rows = len(df)
df = df.drop_duplicates()
after_dedup = len(df)

zero_price = (df["Unit_Price"] == 0).sum()
df["Unit_Price"] = df["Unit_Price"].replace(0, np.nan)
df["Unit_Price"] = df.groupby("Category")["Unit_Price"].transform(lambda s: s.fillna(s.median()))
df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]

missing_age = df["Avg_Customer_Age"].isna().sum()
df["Avg_Customer_Age"] = df.groupby("Category")["Avg_Customer_Age"].transform(lambda s: s.fillna(s.median()))

df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.day_name()
df["Weekday_Num"] = df["Date"].dt.dayofweek
df["Is_Weekend"] = (df["Weekday_Num"] >= 5).astype(int)
df["Promotion_Label"] = df["Promotion"].map({0: "No Promotion", 1: "Promotion"})

df.to_csv("/home/claude/retail_project/data/retail_sales_cleaned.csv", index=False)

daily_revenue = df.groupby("Date")["Revenue"].sum().reset_index()
category_revenue = df.groupby("Category")["Revenue"].sum().sort_values(ascending=False)
weekday_revenue = df.groupby("Weekday")["Revenue"].mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
promo_effect = df.groupby("Promotion_Label")["Revenue"].mean()
monthly_revenue = df.groupby(df["Date"].dt.to_period("M"))["Revenue"].sum()

corr_cols = ["Units_Sold", "Unit_Price", "Revenue", "Customer_Count", "Avg_Customer_Age", "Promotion"]
corr = df[corr_cols].corr()

with open("/home/claude/retail_project/eda_findings.txt", "w") as f:
    f.write("RETAIL SALES EDA FINDINGS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Raw rows: {before_rows} | After removing duplicates: {after_dedup} "
            f"({before_rows - after_dedup} duplicates removed)\n")
    f.write(f"Rows with implausible zero Unit_Price: {zero_price}\n")
    f.write(f"Rows with missing Avg_Customer_Age: {missing_age}\n\n")
    f.write("Total revenue by category:\n")
    f.write(category_revenue.to_string() + "\n\n")
    f.write("Average daily revenue by weekday:\n")
    f.write(weekday_revenue.to_string() + "\n\n")
    f.write("Average daily revenue: promotion vs no promotion:\n")
    f.write(promo_effect.to_string() + "\n\n")
    f.write("Correlation matrix:\n")
    f.write(corr.to_string() + "\n")

palette_cat = dict(zip(categories := category_revenue.index.tolist(),
                        sns.color_palette("viridis", len(category_revenue))))

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(daily_revenue["Date"], daily_revenue["Revenue"], color="#4C72B0", linewidth=1)
ax.plot(daily_revenue["Date"], daily_revenue["Revenue"].rolling(30).mean(), color="#DD8452", linewidth=2.2)
ax.set_title("Daily Total Revenue with 30-Day Rolling Average")
ax.set_ylabel("Revenue")
ax.legend(["Daily Revenue", "30-Day Rolling Avg"])
plt.tight_layout()
plt.savefig(f"{IMG}/01_daily_revenue_trend.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=category_revenue.values, y=category_revenue.index, hue=category_revenue.index,
            palette="viridis", legend=False, ax=ax)
ax.set_title("Total Revenue by Category")
ax.set_xlabel("Total Revenue")
plt.tight_layout()
plt.savefig(f"{IMG}/02_revenue_by_category.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=weekday_revenue.index, y=weekday_revenue.values, hue=weekday_revenue.index,
            palette="crest", legend=False, ax=ax)
ax.set_title("Average Daily Revenue by Weekday")
ax.set_ylabel("Average Revenue")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{IMG}/03_revenue_by_weekday.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df, x="Promotion_Label", y="Revenue", hue="Promotion_Label",
            palette={"No Promotion": "#4C72B0", "Promotion": "#DD8452"}, legend=False, ax=ax)
ax.set_title("Revenue Distribution: Promotion vs No Promotion")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(f"{IMG}/04_promotion_effect.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(f"{IMG}/05_correlation_heatmap.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
monthly_revenue.plot(kind="bar", ax=ax, color="#55A868")
ax.set_title("Total Revenue by Month")
ax.set_ylabel("Revenue")
ax.set_xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{IMG}/06_monthly_revenue.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.scatterplot(data=df.sample(600, random_state=1), x="Customer_Count", y="Revenue",
                 hue="Category", palette="viridis", alpha=0.7, ax=ax)
ax.set_title("Customer Count vs Revenue by Category")
plt.tight_layout()
plt.savefig(f"{IMG}/07_customers_vs_revenue.png")
plt.close()

pivot = df.pivot_table(index="Category", columns="Weekday", values="Revenue", aggfunc="mean")
pivot = pivot[["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]]
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
ax.set_title("Average Revenue: Category vs Weekday")
plt.tight_layout()
plt.savefig(f"{IMG}/08_category_weekday_heatmap.png")
plt.close()

print("EDA complete, charts saved.")
print(category_revenue)
print(promo_effect)
