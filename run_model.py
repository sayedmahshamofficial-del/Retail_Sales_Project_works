import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.dpi"] = 130

IMG = "/home/claude/retail_project/images"
df = pd.read_csv("/home/claude/retail_project/data/retail_sales_cleaned.csv", parse_dates=["Date"])

daily = df.groupby(["Date", "Category"], as_index=False).agg(
    Revenue=("Revenue", "sum"),
    Units_Sold=("Units_Sold", "sum"),
    Customer_Count=("Customer_Count", "sum"),
    Promotion=("Promotion", "max"),
)
daily["Month"] = daily["Date"].dt.month
daily["Weekday_Num"] = daily["Date"].dt.dayofweek
daily["Is_Weekend"] = (daily["Weekday_Num"] >= 5).astype(int)
daily["Day_Index"] = (daily["Date"] - daily["Date"].min()).dt.days

daily = daily.sort_values(["Category", "Date"])
daily["Revenue_Lag1"] = daily.groupby("Category")["Revenue"].shift(1)
daily["Revenue_Lag7"] = daily.groupby("Category")["Revenue"].shift(7)
daily = daily.dropna().reset_index(drop=True)

cat_dummies = pd.get_dummies(daily["Category"], prefix="Cat")
features = pd.concat([
    daily[["Month", "Weekday_Num", "Is_Weekend", "Promotion", "Day_Index",
           "Revenue_Lag1", "Revenue_Lag7"]],
    cat_dummies,
], axis=1)
target = daily["Revenue"]

split_date = daily["Date"].quantile(0.85, interpolation="nearest")
train_mask = daily["Date"] <= split_date
X_train, X_test = features[train_mask], features[~train_mask]
y_train, y_test = target[train_mask], target[~train_mask]

model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

with open("/home/claude/retail_project/model_results.txt", "w") as f:
    f.write("SALES PREDICTION MODEL RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: RandomForestRegressor(n_estimators=300, max_depth=10)\n")
    f.write(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")
    f.write(f"Split date: train up to {split_date.date()}, test after\n\n")
    f.write(f"MAE: {mae:,.2f}\n")
    f.write(f"RMSE: {rmse:,.2f}\n")
    f.write(f"MAPE: {mape:,.2f}%\n")
    f.write(f"R2 Score: {r2:.4f}\n")

print(f"MAE={mae:.2f} RMSE={rmse:.2f} MAPE={mape:.2f}% R2={r2:.4f}")

importances = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette="viridis", legend=False, ax=ax)
ax.set_title("Feature Importance for Revenue Prediction")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{IMG}/09_feature_importance.png")
plt.close()

results_df = daily[~train_mask][["Date", "Category"]].copy()
results_df["Actual"] = y_test.values
results_df["Predicted"] = preds

fig, ax = plt.subplots(figsize=(11, 5))
agg_actual = results_df.groupby("Date")["Actual"].sum()
agg_pred = results_df.groupby("Date")["Predicted"].sum()
ax.plot(agg_actual.index, agg_actual.values, label="Actual", color="#4C72B0", linewidth=2)
ax.plot(agg_pred.index, agg_pred.values, label="Predicted", color="#DD8452", linewidth=2, linestyle="--")
ax.set_title("Actual vs Predicted Daily Revenue (Test Period)")
ax.set_ylabel("Total Revenue Across Categories")
ax.legend()
plt.tight_layout()
plt.savefig(f"{IMG}/10_actual_vs_predicted.png")
plt.close()

fig, ax = plt.subplots(figsize=(6.5, 6))
sns.scatterplot(x=y_test, y=preds, alpha=0.6, ax=ax, color="#4C72B0")
lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Actual Revenue")
ax.set_ylabel("Predicted Revenue")
ax.set_title(f"Predicted vs Actual (R2 = {r2:.3f})")
plt.tight_layout()
plt.savefig(f"{IMG}/11_prediction_scatter.png")
plt.close()

results_df.to_csv("/home/claude/retail_project/data/prediction_results.csv", index=False)
print("Model complete, charts and results saved.")
