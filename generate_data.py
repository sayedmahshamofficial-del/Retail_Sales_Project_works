import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

start = pd.Timestamp("2024-01-01")
days = 730
dates = pd.date_range(start, periods=days, freq="D")

categories = ["Electronics", "Clothing", "Groceries", "Furniture", "Toys"]
base_price = {"Electronics": 250, "Clothing": 45, "Groceries": 12, "Furniture": 380, "Toys": 30}
base_units = {"Electronics": 18, "Clothing": 40, "Groceries": 120, "Furniture": 6, "Toys": 25}
cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad"]

rows = []
for i, d in enumerate(dates):
    dow = d.dayofweek
    month = d.month
    weekend_boost = 1.25 if dow >= 5 else 1.0
    season_boost = 1.4 if month in (11, 12) else (0.85 if month in (1, 7) else 1.0)
    trend = 1 + 0.0004 * i
    promo = rng.random() < 0.12
    promo_boost = 1.35 if promo else 1.0

    for cat in categories:
        noise = rng.normal(1, 0.12)
        units = base_units[cat] * weekend_boost * season_boost * trend * promo_boost * noise
        units = max(0, round(units + rng.normal(0, 3)))
        price = base_price[cat] * rng.normal(1, 0.05)
        price = round(max(price, 1), 2)
        revenue = round(units * price, 2)
        customers = max(1, round(units / rng.uniform(1.1, 2.5)))
        avg_age = round(rng.normal(35, 10), 1)
        city = rng.choice(cities)

        rows.append({
            "Date": d,
            "Category": cat,
            "City": city,
            "Units_Sold": units,
            "Unit_Price": price,
            "Revenue": revenue,
            "Customer_Count": customers,
            "Avg_Customer_Age": avg_age,
            "Promotion": int(promo),
        })

df = pd.DataFrame(rows)

zero_idx = rng.choice(df.index, size=25, replace=False)
df.loc[zero_idx, "Unit_Price"] = 0

nan_idx = rng.choice(df.index, size=30, replace=False)
df.loc[nan_idx, "Avg_Customer_Age"] = np.nan

dup_idx = rng.choice(df.index, size=15, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

df.to_csv("/home/claude/retail_project/data/retail_sales_dataset.csv", index=False)
print(df.shape)
print(df.head())
