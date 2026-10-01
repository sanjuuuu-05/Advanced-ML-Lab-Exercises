# Lab Exercise 3 — Demonstrate Feature Selection

## Dataset (Kaggle)
Use the Wine Quality dataset by Yasser H.:
https://www.kaggle.com/datasets/yasserh/wine-quality-dataset

Download `WineQT.csv` and place it in this folder for the Jupyter Notebook.
The Streamlit app also lets you upload the CSV from the sidebar.

## Dataset use
- Regression target: `quality` (numeric wine quality score)
- Classification target: generated from `quality` using a threshold (default: quality >= 6 is "Good"; below 6 is "Low")
- `Id`, if present, is removed because it is an identifier, not a meaningful measurement.
- The exact row/column count is displayed after loading the file.

## Setup
```bash
pip install -r requirements.txt
```

## Run Jupyter
Open `Lab3_Feature_Selection.ipynb` and run cells from top to bottom.

## Run Streamlit
From this folder:
```bash
streamlit run app.py
```

## What is covered
- Missing values, duplicate check, descriptive statistics and EDA
- Low Variance Filter
- High Correlation Filter
- Factor Analysis (feature reduction into latent factors)
- Backward Feature Elimination
- Forward Feature Selection
- Linear Regression and Logistic Regression on original and selected feature sets
- Manual batch Gradient Descent for linear and logistic regression, with learning-rate/iteration comparison
- K-fold cross-validation (mean and standard deviation)
- GridSearchCV for Ridge and Logistic Regression
- Metric tables and visualizations

Note: Filter and sequential selection methods are fit on training data only to reduce leakage. Factor Analysis is fit on training data only and then applied to test data.
