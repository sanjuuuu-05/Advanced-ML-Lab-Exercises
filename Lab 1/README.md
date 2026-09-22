# Interactive Car Price Prediction - AML Lab Exercise 1

This Streamlit application implements the finalized AML Lab Exercise 1:
**Car Price Prediction Using Regression**.

## Main features

- Interactive dataset exploration
- Adjustable EDA visualizations
- Selectable feature vs price plots
- Adjustable histogram bins
- Adjustable scatter point size and opacity
- Selectable correlation heatmap attributes
- Interactive train/test split
- Adjustable random state
- Automatic model retraining
- Multiple Linear Regression
- MAE, MSE, RMSE and R² Score
- Actual vs Predicted visualization
- Residual/error visualization
- Interactive new-car specification controls
- Instant predicted car price
- Current car configuration preview

## Run

Put `CarPrice_Assignment.csv` in the project folder (or upload it in the app):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Expected dataset:
- 205 records
- 26 attributes
- target: `price`
