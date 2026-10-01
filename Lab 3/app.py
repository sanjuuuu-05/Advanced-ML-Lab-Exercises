import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SequentialFeatureSelector
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE / THEME
# ============================================================
st.set_page_config(
    page_title="Wine Quality | Feature Selection Lab",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.18);
        padding: 12px;
        border-radius: 12px;
    }
    .insight {
        padding: 14px 16px;
        border-left: 4px solid #4F8BF9;
        background: rgba(79,139,249,0.08);
        border-radius: 8px;
        margin: 8px 0;
    }
    .small-note {
        color: #777;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🍷 Wine Quality — Interactive Feature Selection Dashboard")
st.caption(
    "MAI511-2 Advanced Machine Learning | Exploratory Analysis • Feature Selection • "
    "Regression • Classification • Gradient Descent • Hyperparameter Tuning"
)


# ============================================================
# HELPERS
# ============================================================
def insight_box(text):
    st.markdown(f'<div class="insight">💡 <b>What this graph says:</b> {text}</div>',
                unsafe_allow_html=True)


def pct(x):
    return f"{x * 100:.1f}%"


def safe_corr(a, b):
    tmp = pd.concat([a, b], axis=1).dropna()
    if len(tmp) < 2:
        return np.nan
    return tmp.iloc[:, 0].corr(tmp.iloc[:, 1])


def plotly_config():
    return {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Dashboard Controls")

uploaded = st.sidebar.file_uploader("Upload WineQT.csv", type=["csv"])
local_path = st.sidebar.text_input("Or local CSV path", value="WineQT.csv")

st.sidebar.divider()
threshold = st.sidebar.slider(
    "Good quality threshold",
    min_value=4, max_value=8, value=6,
    help="Classification label becomes 1 when quality ≥ this value."
)
corr_threshold = st.sidebar.slider(
    "High-correlation cutoff",
    min_value=0.70, max_value=0.99, value=0.90, step=0.01,
    help="Features above this absolute correlation are candidates for removal."
)
variance_threshold = st.sidebar.number_input(
    "Low variance threshold",
    min_value=0.0, value=0.0, step=0.001, format="%.3f"
)
test_size = st.sidebar.slider("Test-set fraction", 0.15, 0.40, 0.20, 0.05)
cv_folds = st.sidebar.selectbox("Cross-validation folds", [3, 5, 10], index=1)

st.sidebar.divider()
st.sidebar.info(
    "Tip: change the quality threshold or correlation cutoff and watch how the "
    "class balance, selected features, and model metrics change."
)


# ============================================================
# DATA LOAD
# ============================================================
@st.cache_data
def read_csv(uploaded_file, path):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(path)


try:
    df = read_csv(uploaded, local_path)
except Exception:
    st.error(
        "Upload `WineQT.csv` using the sidebar, or place it beside `app.py`."
    )
    st.stop()

if "quality" not in df.columns:
    st.error("Expected a `quality` column in the uploaded dataset.")
    st.stop()

if "Id" in df.columns:
    df = df.drop(columns=["Id"])

df = df.replace([np.inf, -np.inf], np.nan)

# ============================================================
# DATASET OVERVIEW
# ============================================================
st.header("1. Dataset Overview")

missing_cells = int(df.isna().sum().sum())
duplicates = int(df.duplicated().sum())
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Input features", f"{df.shape[1] - 1}")
c3.metric("Missing cells", f"{missing_cells:,}")
c4.metric("Duplicate rows", f"{duplicates:,}")
c5.metric("Quality mean", f"{df['quality'].mean():.2f}")

with st.expander("🔎 Preview, data types and descriptive statistics"):
    tab_a, tab_b, tab_c = st.tabs(["Data preview", "Statistics", "Data quality"])

    with tab_a:
        st.dataframe(df.head(15), use_container_width=True, height=350)

    with tab_b:
        st.dataframe(df.describe().T.round(3), use_container_width=True)

    with tab_c:
        quality_table = pd.DataFrame({
            "Data type": df.dtypes.astype(str),
            "Missing": df.isna().sum(),
            "Missing %": (df.isna().mean() * 100).round(2),
            "Unique": df.nunique(),
        })
        st.dataframe(quality_table, use_container_width=True)


# ============================================================
# EDA
# ============================================================
st.header("2. Interactive Exploratory Analysis")

eda1, eda2, eda3 = st.tabs([
    "📊 Quality distribution",
    "🔥 Correlations",
    "🔬 Feature vs quality"
])

with eda1:
    fig = px.histogram(
        df, x="quality", color="quality",
        title="Distribution of Wine Quality Scores",
        text_auto=True,
        marginal="box",
    )
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    q_counts = df["quality"].value_counts().sort_index()
    most_common_quality = q_counts.idxmax()
    insight_box(
        f"Quality <b>{most_common_quality}</b> is the most frequent score with "
        f"<b>{q_counts.max():,}</b> observations. The distribution is useful for "
        "checking whether the target is balanced or concentrated around a few scores."
    )

with eda2:
    corr = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    corr_quality = corr["quality"].drop("quality").sort_values()
    strongest = corr_quality.abs().sort_values(ascending=False).index[0]
    strongest_value = corr_quality[strongest]
    insight_box(
        f"<b>{strongest}</b> has the strongest linear relationship with quality in "
        f"absolute value (<b>{strongest_value:.2f}</b>). Correlation measures linear "
        "association; it does not by itself prove causation."
    )

with eda3:
    feature_choice = st.selectbox(
        "Choose a feature to compare with quality",
        [c for c in numeric_cols if c != "quality"],
        index=0,
    )
    chart_type = st.radio(
        "Visualization",
        ["Scatter", "Box plot"],
        horizontal=True,
    )

    if chart_type == "Scatter":
        fig = px.scatter(
            df, x=feature_choice, y="quality",
            opacity=0.65,
            title=f"{feature_choice} vs Wine Quality",
            hover_data=df.columns.tolist(),
        )
    else:
        fig = px.box(
            df, x="quality", y=feature_choice,
            points="outliers",
            title=f"{feature_choice} distribution by quality score",
        )

    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    r = safe_corr(df[feature_choice], df["quality"])
    if pd.notna(r):
        strength = (
            "very weak" if abs(r) < .2 else
            "weak" if abs(r) < .4 else
            "moderate" if abs(r) < .7 else
            "strong"
        )
        direction = "positive" if r >= 0 else "negative"
        insight_box(
            f"The selected feature has a <b>{strength} {direction}</b> linear "
            f"relationship with quality (correlation = <b>{r:.2f}</b>). "
            "The scatter/box plot lets you see whether that relationship is consistent "
            "or driven by a small number of observations."
        )


# ============================================================
# PREPARE DATA
# ============================================================
st.header("3. Prepare Regression & Classification Targets")

feature_cols = [c for c in df.columns if c != "quality"]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y_reg = pd.to_numeric(df["quality"], errors="coerce")

valid = y_reg.notna()
X, y_reg = X.loc[valid], y_reg.loc[valid]
y_cls = (y_reg >= threshold).astype(int)

X_train, X_test, yreg_train, yreg_test, ycls_train, ycls_test = train_test_split(
    X, y_reg, y_cls,
    test_size=test_size,
    random_state=42,
    stratify=y_cls
)

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)
X_test_imp = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_imp),
    columns=X.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_imp),
    columns=X.columns,
    index=X_test.index
)

class_counts = y_cls.value_counts().sort_index()
positive_rate = y_cls.mean()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Training rows", f"{len(X_train):,}")
m2.metric("Test rows", f"{len(X_test):,}")
m3.metric("Good-quality class", f"{class_counts.get(1, 0):,}")
m4.metric("Good-quality rate", pct(positive_rate))

st.caption(
    f"Regression predicts the original quality score. Classification predicts "
    f"whether quality ≥ {threshold}. The classification label is derived from quality."
)

class_df = pd.DataFrame({
    "Class": ["Below threshold", "Good quality"],
    "Count": [class_counts.get(0, 0), class_counts.get(1, 0)]
})
fig = px.bar(
    class_df, x="Class", y="Count", text="Count",
    title=f"Classification Class Balance (threshold = {threshold})"
)
fig.update_traces(textposition="outside")
st.plotly_chart(fig, use_container_width=True, config=plotly_config())
insight_box(
    f"At the selected threshold of <b>{threshold}</b>, "
    f"<b>{class_counts.get(1, 0):,}</b> rows are labelled good quality and "
    f"<b>{class_counts.get(0, 0):,}</b> are below the threshold. "
    "Changing the threshold changes the classification problem itself."
)


# ============================================================
# FEATURE SELECTION
# ============================================================
st.header("4. Feature Selection & Reduction")

def low_variance_features(train_raw, threshold_value):
    selector = VarianceThreshold(threshold=threshold_value)
    selector.fit(train_raw)
    return list(train_raw.columns[selector.get_support()])


def high_corr_features(train_scaled, cutoff):
    corr = train_scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > cutoff)]
    return [c for c in train_scaled.columns if c not in drop_cols]


def sequential_features(Xtr, ytr, task, direction):
    estimator = (
        LinearRegression()
        if task == "regression"
        else LogisticRegression(max_iter=3000)
    )
    n_select = max(1, min(5, Xtr.shape[1] - 1)) if Xtr.shape[1] > 2 else 1
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_select,
        direction=direction,
        scoring=(
            "neg_mean_squared_error"
            if task == "regression"
            else "accuracy"
        ),
        cv=3,
        n_jobs=-1,
    )
    selector.fit(Xtr, ytr)
    return list(Xtr.columns[selector.get_support()])


# Fit feature-selection procedures only on training data.
lv_cols = low_variance_features(X_train_imp, variance_threshold)
hc_cols = high_corr_features(X_train_scaled, corr_threshold)

fa_n = max(1, min(5, X_train_scaled.shape[1]))
fa = FactorAnalysis(n_components=fa_n, random_state=42)
fa_train = fa.fit_transform(X_train_scaled)
fa_test = fa.transform(X_test_scaled)
fa_names = [f"Factor_{i+1}" for i in range(fa_n)]

bwd_reg = sequential_features(X_train_scaled, yreg_train, "regression", "backward")
fwd_reg = sequential_features(X_train_scaled, yreg_train, "regression", "forward")
bwd_cls = sequential_features(X_train_scaled, ycls_train, "classification", "backward")
fwd_cls = sequential_features(X_train_scaled, ycls_train, "classification", "forward")

selection_info = pd.DataFrame([
    ["Original features", len(feature_cols), ", ".join(feature_cols)],
    ["Low Variance Filter", len(lv_cols), ", ".join(lv_cols)],
    ["High Correlation Filter", len(hc_cols), ", ".join(hc_cols)],
    ["Factor Analysis", fa_n, ", ".join(fa_names)],
    ["Backward Elimination (Regression)", len(bwd_reg), ", ".join(bwd_reg)],
    ["Forward Selection (Regression)", len(fwd_reg), ", ".join(fwd_reg)],
    ["Backward Elimination (Classification)", len(bwd_cls), ", ".join(bwd_cls)],
    ["Forward Selection (Classification)", len(fwd_cls), ", ".join(fwd_cls)],
], columns=["Method", "Number retained", "Retained features/components"])

sel1, sel2 = st.columns([1.25, 1])

with sel1:
    st.dataframe(selection_info, use_container_width=True, height=390)

with sel2:
    fig = px.bar(
        selection_info,
        x="Number retained",
        y="Method",
        orientation="h",
        text="Number retained",
        title="How many features/components each method retains",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=390, yaxis={"categoryorder": "array", "categoryarray": selection_info["Method"].tolist()})
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

reduction = 1 - (len(hc_cols) / len(feature_cols))
insight_box(
    f"High-correlation filtering retains <b>{len(hc_cols)}</b> of "
    f"<b>{len(feature_cols)}</b> original features. That is a reduction of "
    f"<b>{reduction * 100:.1f}%</b>. Factor Analysis instead replaces the original "
    "variables with latent components rather than simply deleting columns."
)

with st.expander("📌 See exactly which features were selected"):
    show_method = st.selectbox("Method", selection_info["Method"].tolist())
    selected_row = selection_info[selection_info["Method"] == show_method].iloc[0]
    st.write(f"**{show_method}:** {selected_row['Retained features/components']}")


# ============================================================
# MODEL FUNCTIONS
# ============================================================
def get_feature_matrices(cols, use_fa=False):
    if use_fa:
        return (
            pd.DataFrame(fa_train, index=X_train.index, columns=fa_names),
            pd.DataFrame(fa_test, index=X_test.index, columns=fa_names),
        )
    return X_train_scaled[cols], X_test_scaled[cols]


def regression_metrics(ytrue, pred):
    mse = mean_squared_error(ytrue, pred)
    return {
        "MAE": mean_absolute_error(ytrue, pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(ytrue, pred),
    }


def classification_metrics(ytrue, pred, proba=None):
    result = {
        "Accuracy": accuracy_score(ytrue, pred),
        "Precision": precision_score(ytrue, pred, zero_division=0),
        "Recall": recall_score(ytrue, pred, zero_division=0),
        "F1": f1_score(ytrue, pred, zero_division=0),
    }
    if proba is not None and len(np.unique(ytrue)) == 2:
        result["ROC-AUC"] = roc_auc_score(ytrue, proba)
    else:
        result["ROC-AUC"] = np.nan
    return result


# ============================================================
# MODEL COMPARISON
# ============================================================
st.header("5. Interactive Model Comparison")

if st.button("🚀 Run / Refresh Model Comparison", type="primary"):
    rows = []

    reg_sets = {
        "Original": feature_cols,
        "Low Variance": lv_cols,
        "High Correlation": hc_cols,
        "Backward Selection": bwd_reg,
        "Forward Selection": fwd_reg,
    }

    cls_sets = {
        "Original": feature_cols,
        "Low Variance": lv_cols,
        "High Correlation": hc_cols,
        "Backward Selection": bwd_cls,
        "Forward Selection": fwd_cls,
    }

    for name, cols in reg_sets.items():
        if not cols:
            continue
        xtr, xte = get_feature_matrices(cols)
        model = LinearRegression().fit(xtr, yreg_train)
        pred = model.predict(xte)
        met = regression_metrics(yreg_test, pred)
        cv = cross_val_score(
            LinearRegression(),
            X_train_scaled[cols],
            yreg_train,
            cv=cv_folds,
            scoring="r2",
        )
        rows.append({
            "Task": "Regression",
            "Feature set": name,
            "Features": len(cols),
            **met,
            "CV mean": cv.mean(),
            "CV std": cv.std(),
        })

    xtr_fa = pd.DataFrame(fa_train, index=X_train.index, columns=fa_names)
    xte_fa = pd.DataFrame(fa_test, index=X_test.index, columns=fa_names)

    lm_fa = LinearRegression().fit(xtr_fa, yreg_train)
    reg_fa_pred = lm_fa.predict(xte_fa)
    cv_fa_reg = cross_val_score(
        LinearRegression(), fa_train, yreg_train,
        cv=cv_folds, scoring="r2"
    )
    rows.append({
        "Task": "Regression",
        "Feature set": "Factor Analysis",
        "Features": fa_n,
        **regression_metrics(yreg_test, reg_fa_pred),
        "CV mean": cv_fa_reg.mean(),
        "CV std": cv_fa_reg.std(),
    })

    for name, cols in cls_sets.items():
        if not cols:
            continue
        xtr, xte = get_feature_matrices(cols)
        model = LogisticRegression(max_iter=3000).fit(xtr, ycls_train)
        pred = model.predict(xte)
        proba = model.predict_proba(xte)[:, 1]
        met = classification_metrics(ycls_test, pred, proba)
        cv = cross_val_score(
            LogisticRegression(max_iter=3000),
            X_train_scaled[cols],
            ycls_train,
            cv=cv_folds,
            scoring="accuracy",
        )
        rows.append({
            "Task": "Classification",
            "Feature set": name,
            "Features": len(cols),
            **met,
            "CV mean": cv.mean(),
            "CV std": cv.std(),
        })

    log_fa = LogisticRegression(max_iter=3000).fit(xtr_fa, ycls_train)
    cls_fa_pred = log_fa.predict(xte_fa)
    cls_fa_proba = log_fa.predict_proba(xte_fa)[:, 1]
    cv_fa_cls = cross_val_score(
        LogisticRegression(max_iter=3000),
        fa_train, ycls_train,
        cv=cv_folds, scoring="accuracy"
    )
    rows.append({
        "Task": "Classification",
        "Feature set": "Factor Analysis",
        "Features": fa_n,
        **classification_metrics(ycls_test, cls_fa_pred, cls_fa_proba),
        "CV mean": cv_fa_cls.mean(),
        "CV std": cv_fa_cls.std(),
    })

    result_df = pd.DataFrame(rows)
    st.session_state["result_df"] = result_df

result_df = st.session_state.get("result_df")

if result_df is None:
    st.info("Click **Run / Refresh Model Comparison** to calculate the model results.")
else:
    st.dataframe(
        result_df.style.format({
            c: "{:.3f}" for c in result_df.columns
            if c not in ["Task", "Feature set", "Features"]
        }),
        use_container_width=True,
    )

    reg_df = result_df[result_df["Task"] == "Regression"].copy()
    cls_df = result_df[result_df["Task"] == "Classification"].copy()

    vc1, vc2 = st.columns(2)

    with vc1:
        metric = st.selectbox(
            "Regression metric",
            ["RMSE", "MAE", "R2", "CV mean"],
            key="reg_metric"
        )
        fig = px.bar(
            reg_df, x="Feature set", y=metric,
            text_auto=".3f",
            title=f"Regression: {metric} by feature-selection method",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        if metric in ["RMSE", "MAE"]:
            insight_box(
                f"Lower <b>{metric}</b> means smaller prediction error on the test set. "
                "The best-looking bar for an error metric is therefore the shortest one."
            )
        else:
            insight_box(
                f"Higher <b>{metric}</b> indicates stronger regression performance. "
                "R² represents the proportion of target variance explained by the model."
            )

    with vc2:
        metric = st.selectbox(
            "Classification metric",
            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV mean"],
            key="cls_metric"
        )
        fig = px.bar(
            cls_df, x="Feature set", y=metric,
            text_auto=".3f",
            title=f"Classification: {metric} by feature-selection method",
        )
        fig.update_yaxes(range=[0, 1])
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        insight_box(
            f"This graph compares <b>{metric}</b> across feature sets. "
            "Accuracy is the overall fraction classified correctly; precision focuses "
            "on predicted good-quality wines; recall focuses on how many actual "
            "good-quality wines were found; F1 balances precision and recall."
        )

    fig = px.scatter(
        result_df,
        x="Features",
        y="CV mean",
        color="Task",
        symbol="Task",
        hover_name="Feature set",
        size="Features",
        title="Feature Count vs Cross-Validation Performance",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())
    insight_box(
        "Each point shows the trade-off between the number of retained features and "
        "average cross-validation performance. A smaller feature set can be useful when "
        "it maintains similar validation performance while reducing model complexity."
    )


# ============================================================
# RESIDUAL / CONFUSION VISUALS
# ============================================================
st.header("6. Model Diagnostics")

diag_reg, diag_cls = st.tabs(["📈 Regression diagnostics", "🎯 Classification diagnostics"])

with diag_reg:
    if result_df is None:
        st.info("Run model comparison first.")
    else:
        model_cols = {
            "Original": feature_cols,
            "Low Variance": lv_cols,
            "High Correlation": hc_cols,
            "Backward Selection": bwd_reg,
            "Forward Selection": fwd_reg,
        }
        selected_diag = st.selectbox("Regression feature set", list(model_cols.keys()))
        cols = model_cols[selected_diag]
        model = LinearRegression().fit(X_train_scaled[cols], yreg_train)
        pred = model.predict(X_test_scaled[cols])
        residuals = yreg_test.to_numpy() - pred

        residual_df = pd.DataFrame({
            "Predicted": pred,
            "Residual": residuals,
            "Actual": yreg_test.to_numpy(),
        })

        fig = px.scatter(
            residual_df,
            x="Predicted", y="Residual",
            hover_data=["Actual"],
            trendline="ols",
            title=f"Residual Plot — {selected_diag}",
        )
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        rmse = np.sqrt(mean_squared_error(yreg_test, pred))
        insight_box(
            f"Residual = actual quality − predicted quality. The residuals should ideally "
            f"be scattered around zero without a strong pattern. Here, RMSE is "
            f"<b>{rmse:.3f}</b>. A visible curve or funnel shape can indicate that the "
            "linear model is missing structure in the data."
        )

with diag_cls:
    if result_df is None:
        st.info("Run model comparison first.")
    else:
        model_cols = {
            "Original": feature_cols,
            "Low Variance": lv_cols,
            "High Correlation": hc_cols,
            "Backward Selection": bwd_cls,
            "Forward Selection": fwd_cls,
        }
        selected_diag = st.selectbox("Classification feature set", list(model_cols.keys()))
        cols = model_cols[selected_diag]

        model = LogisticRegression(max_iter=3000).fit(
            X_train_scaled[cols], ycls_train
        )
        pred = model.predict(X_test_scaled[cols])
        cm = confusion_matrix(ycls_test, pred, labels=[0, 1])

        cm_df = pd.DataFrame(
            cm,
            index=["Actual: Below threshold", "Actual: Good quality"],
            columns=["Predicted: Below threshold", "Predicted: Good quality"],
        )

        fig = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — {selected_diag}",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())

        tn, fp, fn, tp = cm.ravel()
        insight_box(
            f"The matrix contains <b>{tn}</b> true negatives, <b>{fp}</b> false positives, "
            f"<b>{fn}</b> false negatives, and <b>{tp}</b> true positives. "
            "False positives are wines predicted as good when they are below the chosen "
            "threshold; false negatives are good-quality wines that the model misses."
        )


# ============================================================
# GRADIENT DESCENT
# ============================================================
st.header("7. Gradient Descent — Interactive Learning Curve")

gd_task = st.radio(
    "Choose task",
    ["Linear Regression", "Logistic Regression"],
    horizontal=True
)
lr = st.select_slider(
    "Learning rate",
    options=[0.0001, 0.001, 0.01, 0.05, 0.1],
    value=0.01
)
iterations = st.slider(
    "Iterations", min_value=100, max_value=3000,
    value=1000, step=100
)


def run_linear_gd(Xarr, yarr, learning_rate, n_iter):
    Xarr = np.asarray(Xarr, dtype=float)
    yarr = np.asarray(yarr, dtype=float)

    mu, sd = Xarr.mean(axis=0), Xarr.std(axis=0)
    sd[sd == 0] = 1
    Xs = (Xarr - mu) / sd

    ys_std = yarr.std() if yarr.std() else 1
    ys = (yarr - yarr.mean()) / ys_std

    Xb = np.c_[np.ones(len(Xs)), Xs]
    theta = np.zeros(Xb.shape[1])
    losses = []

    for i in range(n_iter):
        err = Xb @ theta - ys
        theta -= learning_rate * (Xb.T @ err) / len(ys)
        if i % max(1, n_iter // 100) == 0:
            losses.append(float(np.mean((Xb @ theta - ys) ** 2)))

    return theta, losses


def run_logistic_gd(Xarr, yarr, learning_rate, n_iter):
    Xarr = np.asarray(Xarr, dtype=float)
    yarr = np.asarray(yarr, dtype=float)

    Xs = (Xarr - Xarr.mean(axis=0)) / np.where(
        Xarr.std(axis=0) == 0, 1, Xarr.std(axis=0)
    )
    Xb = np.c_[np.ones(len(Xs)), Xs]
    theta = np.zeros(Xb.shape[1])
    losses = []

    for i in range(n_iter):
        z = np.clip(Xb @ theta, -30, 30)
        p = 1 / (1 + np.exp(-z))
        theta -= learning_rate * (Xb.T @ (p - yarr)) / len(yarr)

        if i % max(1, n_iter // 100) == 0:
            p = np.clip(
                1 / (1 + np.exp(-np.clip(Xb @ theta, -30, 30))),
                1e-9, 1 - 1e-9
            )
            losses.append(float(
                -np.mean(yarr * np.log(p) + (1 - yarr) * np.log(1 - p))
            ))

    return theta, losses


if st.button("▶️ Run Gradient Descent"):
    gd_X = X_train_scaled[feature_cols].to_numpy()

    if gd_task == "Linear Regression":
        _, losses = run_linear_gd(
            gd_X, yreg_train.to_numpy(), lr, iterations
        )
        ylabel = "Mean Squared Error"
    else:
        _, losses = run_logistic_gd(
            gd_X, ycls_train.to_numpy(), lr, iterations
        )
        ylabel = "Log Loss"

    loss_df = pd.DataFrame({
        "Checkpoint": np.arange(1, len(losses) + 1),
        "Loss": losses,
    })

    fig = px.line(
        loss_df, x="Checkpoint", y="Loss",
        markers=False,
        title=f"{gd_task}: Training Loss",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    first_loss = losses[0]
    final_loss = losses[-1]
    change = first_loss - final_loss

    insight_box(
        f"The loss changed from <b>{first_loss:.4f}</b> to <b>{final_loss:.4f}</b>, "
        f"a reduction of <b>{change:.4f}</b>. A generally decreasing curve suggests "
        "the optimization is moving toward convergence. Oscillation or growth suggests "
        "that the learning rate may be too large."
    )


# ============================================================
# GRID SEARCH
# ============================================================
st.header("8. Hyperparameter Tuning with GridSearchCV")

if st.button("🔧 Run GridSearchCV for Ridge + Logistic Regression"):
    with st.spinner("Searching parameter combinations..."):
        ridge_grid = GridSearchCV(
            Ridge(),
            {"alpha": [0.01, 0.1, 1, 10, 100]},
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            return_train_score=True,
        )
        ridge_grid.fit(X_train_scaled, yreg_train)

        log_grid = GridSearchCV(
            LogisticRegression(max_iter=3000),
            {"C": [0.01, 0.1, 1, 10, 100]},
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=True,
        )
        log_grid.fit(X_train_scaled, ycls_train)

    ridge_pred = ridge_grid.best_estimator_.predict(X_test_scaled)
    log_pred = log_grid.best_estimator_.predict(X_test_scaled)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Ridge α", str(ridge_grid.best_params_["alpha"]))
    c2.metric(
        "Ridge test RMSE",
        f"{np.sqrt(mean_squared_error(yreg_test, ridge_pred)):.3f}"
    )
    c3.metric("Best Logistic C", str(log_grid.best_params_["C"]))
    c4.metric("Logistic test accuracy", f"{accuracy_score(ycls_test, log_pred):.3f}")

    tune1, tune2 = st.columns(2)

    with tune1:
        ridge_results = pd.DataFrame(ridge_grid.cv_results_)
        ridge_results["CV MSE"] = -ridge_results["mean_test_score"]

        fig = px.line(
            ridge_results,
            x="param_alpha",
            y="CV MSE",
            markers=True,
            log_x=True,
            title="Ridge: α vs Cross-Validation MSE",
        )
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())
        insight_box(
            "Ridge α controls the strength of L2 regularization. Larger α values impose "
            "stronger coefficient shrinkage. The plot shows how the tested values affected "
            "cross-validation error."
        )

    with tune2:
        log_results = pd.DataFrame(log_grid.cv_results_)

        fig = px.line(
            log_results,
            x="param_C",
            y="mean_test_score",
            markers=True,
            log_x=True,
            title="Logistic Regression: C vs CV Accuracy",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, config=plotly_config())
        insight_box(
            "C is the inverse of regularization strength in Logistic Regression. "
            "The graph shows how the tested C values changed cross-validation accuracy."
        )

    cm = confusion_matrix(ycls_test, log_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Below threshold", "Actual: Good quality"],
        columns=["Predicted: Below threshold", "Predicted: Good quality"],
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Tuned Logistic Regression — Test Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True, config=plotly_config())

    st.caption(
        f"Best Ridge CV MSE: {-ridge_grid.best_score_:.4f} | "
        f"Best Logistic CV accuracy: {log_grid.best_score_:.4f}"
    )


# ============================================================
# LEARNING GUIDE
# ============================================================
st.header("9. What Each Visualization Tells You")

guide = pd.DataFrame([
    ["Quality distribution", "How the target quality scores are distributed and whether some scores dominate."],
    ["Correlation heatmap", "Which numeric variables move together linearly and which variables are related to quality."],
    ["Feature vs quality", "Whether a selected feature has a visible relationship with quality and whether outliers exist."],
    ["Class balance", "How many wines fall below vs above the chosen classification threshold."],
    ["Feature-selection chart", "How aggressively each selection/reduction method reduces the input feature space."],
    ["Regression metric chart", "How feature sets affect prediction error or explained variance."],
    ["Classification metric chart", "How feature sets affect classification performance from different perspectives."],
    ["Feature count vs CV", "The trade-off between model input size and cross-validation performance."],
    ["Residual plot", "Whether regression errors look random or show systematic patterns."],
    ["Confusion matrix", "Exactly which classifications were correct and which were false positives/negatives."],
    ["Gradient-descent curve", "Whether the optimization loss is decreasing toward convergence."],
    ["GridSearch curves", "How tested hyperparameter values affect validation performance."],
], columns=["Visualization", "Meaning"])

st.dataframe(guide, use_container_width=True, hide_index=True)

st.info(
    "Educational note: classification labels are derived from the original `quality` "
    "score using the selected threshold. They are not a separate original target column. "
    "Feature-selection procedures are fitted using training data to reduce test-data leakage."
)

st.caption("Built with Streamlit + Plotly + scikit-learn. Interactive charts support hover, zoom, pan and download.")