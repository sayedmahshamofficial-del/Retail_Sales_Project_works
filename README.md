# Real-World Data Project — Retail Sales Analysis & Forecasting

## What's in this folder

- **Retail_Sales_Report.docx** — the final structured report (objective, cleaning,
  statistics, all charts, model results, findings, conclusion). Main file to submit.
- **data/retail_sales_dataset.csv** — raw dataset (3,665 records: 2 years of daily
  sales across 5 categories and 5 cities).
- **data/retail_sales_cleaned.csv** — cleaned dataset after removing duplicates and
  imputing missing values (3,650 records).
- **data/prediction_results.csv** — actual vs predicted revenue on the test set.
- **notebook/Retail_Sales_Analysis.ipynb** — full Jupyter notebook with all code,
  narrative markdown, and pre-rendered outputs (EDA + prediction model).
- **images/** — every chart as a standalone PNG (11 charts total: trends, category
  breakdown, weekday patterns, promotion effect, correlation, feature importance,
  actual vs predicted).
- **generate_data.py** — script that generated the dataset.
- **run_eda.py** — cleaning, statistics, and EDA chart generation.
- **run_model.py** — RandomForestRegressor training, evaluation, and prediction charts.
- **eda_findings.txt** — raw statistical summary from the EDA step.
- **model_results.txt** — model metrics (MAE, RMSE, MAPE, R²).

## How to run it yourself

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py   # creates data/retail_sales_dataset.csv
python run_eda.py         # cleans data, saves EDA charts to images/
python run_model.py       # trains model, saves prediction charts to images/
```

Or open `notebook/Retail_Sales_Analysis.ipynb` directly — it already contains
executed outputs end to end.

## Key takeaway

Electronics and Furniture drive the most revenue, weekends and promotions both
lift sales significantly, and a lag-based Random Forest model forecasts daily
revenue with R² ≈ 0.82, though it underpredicts the sharpest holiday-season
peaks. Full reasoning and charts are in `Retail_Sales_Report.docx`.
