"""
Appliances Energy Prediction — Regularized Regression Dashboard
Lab Exercise 2 | MAI511-2 Advanced Machine Learning
Linear Regression vs Lasso vs Ridge, with GridSearchCV cross-validation.
Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------------------------------------------
# PAGE CONFIG + THEME
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Appliances Energy Prediction | Regularized Regression",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=JetBrains+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

:root {
    --accent1: #6366f1;
    --accent2: #06b6d4;
    --accent3: #f472b6;
    --bg-card: rgba(255,255,255,0.04);
}

.stApp {
    background: radial-gradient(circle at 10% 0%, #1a1a2e 0%, #0f0f1a 45%, #0a0a12 100%);
}

/* Animated gradient hero header */
.hero {
    padding: 2.2rem 2rem;
    border-radius: 18px;
    margin-bottom: 1.6rem;
    background: linear-gradient(120deg, var(--accent1), var(--accent2), var(--accent3), var(--accent1));
    background-size: 300% 300%;
    animation: gradientShift 10s ease infinite;
    box-shadow: 0 10px 40px rgba(99,102,241,0.25);
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.hero h1 { color: white; margin: 0; font-weight: 700; font-size: 2.1rem; letter-spacing: -0.5px;}
.hero p { color: rgba(255,255,255,0.9); margin-top: 0.4rem; font-size: 1rem;}

/* Metric cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    backdrop-filter: blur(6px);
}
.metric-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 12px 28px rgba(99,102,241,0.25);
    border-color: var(--accent1);
}
.metric-label { color: #a1a1c2; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; }
.metric-value { color: #ffffff; font-size: 1.7rem; font-weight: 700; margin-top: 0.15rem; }
.metric-delta { font-size: 0.8rem; margin-top: 0.2rem; }
.delta-good { color: #34d399; }
.delta-bad { color: #f87171; }

/* Section fade-in */
.fade-in { animation: fadeIn 0.6s ease-in; }
@keyframes fadeIn { from {opacity:0; transform: translateY(8px);} to {opacity:1; transform: translateY(0);} }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14142b 0%, #0d0d1a 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
    border-radius: 10px 10px 0 0 !important;
}
div[data-testid="stMetricValue"] { color: #ffffff; }

/* Best-model badge */
.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    color: white;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DATA LOADING (cached — only re-runs when the file changes)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df_model = df.drop(columns=["date"])
    return df, df_model


@st.cache_resource(show_spinner="Training models — this runs once and is cached...")
def train_all_models(df_model, test_size, random_state, alpha_values):
    X = df_model.drop(columns=["Appliances"])
    y = df_model["Appliances"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Baseline
    lin = LinearRegression().fit(X_train_s, y_train)
    pred_lin = lin.predict(X_test_s)

    # Lasso + GridSearch
    lasso_grid = GridSearchCV(
        Lasso(max_iter=10000), {"alpha": alpha_values},
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    ).fit(X_train_s, y_train)
    best_lasso = lasso_grid.best_estimator_
    pred_lasso = best_lasso.predict(X_test_s)

    # Ridge + GridSearch
    ridge_grid = GridSearchCV(
        Ridge(), {"alpha": alpha_values},
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    ).fit(X_train_s, y_train)
    best_ridge = ridge_grid.best_estimator_
    pred_ridge = best_ridge.predict(X_test_s)

    def metrics(y_true, y_pred):
        return dict(
            MAE=mean_absolute_error(y_true, y_pred),
            MSE=mean_squared_error(y_true, y_pred),
            RMSE=np.sqrt(mean_squared_error(y_true, y_pred)),
            R2=r2_score(y_true, y_pred),
        )

    results = pd.DataFrame([
        {"Model": "Linear Regression", **metrics(y_test, pred_lin)},
        {"Model": "Lasso Regression", **metrics(y_test, pred_lasso)},
        {"Model": "Ridge Regression", **metrics(y_test, pred_ridge)},
    ])

    coef_table = pd.DataFrame({
        "Feature": X.columns,
        "Linear": lin.coef_,
        "Lasso": best_lasso.coef_,
        "Ridge": best_ridge.coef_,
    })

    return {
        "X": X, "y": y, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler,
        "models": {"Linear Regression": lin, "Lasso Regression": best_lasso, "Ridge Regression": best_ridge},
        "preds": {"Linear Regression": pred_lin, "Lasso Regression": pred_lasso, "Ridge Regression": pred_ridge},
        "results": results,
        "coef_table": coef_table,
        "lasso_grid": lasso_grid,
        "ridge_grid": ridge_grid,
    }


def metric_card(col, label, value, sub=None, good=True):
    delta_class = "delta-good" if good else "delta-bad"
    sub_html = f'<div class="metric-delta {delta_class}">{sub}</div>' if sub else ""
    col.markdown(
        f"""<div class="metric-card fade-in">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {sub_html}
            </div>""",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------
st.markdown(
    """<div class="hero">
            <h1>⚡ Appliances Energy Prediction</h1>
            <p>Lasso &amp; Ridge Regularized Regression · Cross-Validated Grid Search ·
               Smart Home / IoT Dataset · MAI511-2 Lab Exercise 2</p>
        </div>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# SIDEBAR — CONTROLS
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded = st.sidebar.file_uploader("Dataset (energydata_complete.csv)", type="csv")
data_path = uploaded if uploaded is not None else "energydata_complete.csv"

try:
    df_raw, df_model = load_data(data_path)
except FileNotFoundError:
    st.error(
        "Couldn't find **energydata_complete.csv**. Upload it from the sidebar, "
        "or place it in the same folder as this app."
    )
    st.stop()

st.sidebar.divider()
test_size = st.sidebar.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

st.sidebar.markdown("**Alpha values tested in grid search**")
alpha_min_exp = st.sidebar.slider("Min alpha (10^x)", -4, 0, -3)
alpha_max_exp = st.sidebar.slider("Max alpha (10^x)", 0, 3, 2)
alpha_values = np.unique(np.round(
    np.logspace(alpha_min_exp, alpha_max_exp, num=6), 6
)).tolist()

st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 Data Exploration", "🧪 Preprocessing",
     "📈 Model Training & Comparison", "🎛️ Live Alpha Tuning", "🔮 Predict a New Reading"],
)

with st.spinner("Training Linear, Lasso & Ridge models..."):
    bundle = train_all_models(df_model, test_size, int(random_state), alpha_values)

results = bundle["results"]
best_row = results.loc[results["R2"].idxmax()]

# ----------------------------------------------------------------------
# PAGE: OVERVIEW
# ----------------------------------------------------------------------
if page == "🏠 Overview":
    st.subheader("Dataset Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Rows", f"{df_raw.shape[0]:,}")
    metric_card(c2, "Original Features", f"{df_raw.shape[1] - 1}")
    metric_card(c3, "Target", "Appliances (Wh)")
    metric_card(c4, "Best Model", best_row["Model"].split()[0], sub=f"R² = {best_row['R2']:.3f}")

    st.markdown("###")
    st.subheader("Model Leaderboard")
    st.dataframe(
        results.style
        .background_gradient(subset=["R2"], cmap="Greens")
        .background_gradient(subset=["MAE", "MSE", "RMSE"], cmap="Reds_r")
        .format({"MAE": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.4f}"}),
        use_container_width=True,
    )

    fig = px.bar(
        results, x="Model", y="R2", color="Model", text_auto=".3f",
        title="R² Score by Model", color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_layout(template="plotly_dark", showlegend=False, transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Use the sidebar to switch pages, resize the test split, or widen the "
        "alpha search range — every page below recalculates live."
    )

# ----------------------------------------------------------------------
# PAGE: DATA EXPLORATION
# ----------------------------------------------------------------------
elif page == "🔍 Data Exploration":
    st.subheader("Target Distribution")
    fig = px.histogram(
        df_raw, x="Appliances", nbins=50, marginal="box",
        color_discrete_sequence=["#6366f1"],
        title="Distribution of Appliance Energy Consumption (Wh)",
    )
    fig.update_layout(template="plotly_dark", bargap=0.02)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature vs Appliances")
    numeric_cols = [c for c in df_raw.select_dtypes(include=np.number).columns if c != "Appliances"]
    default_idx = numeric_cols.index("T1") if "T1" in numeric_cols else 0
    feature = st.selectbox("Choose a feature to plot against Appliances", numeric_cols, index=default_idx)
    fig2 = px.scatter(
        df_raw, x=feature, y="Appliances", opacity=0.4, trendline="ols",
        color_discrete_sequence=["#06b6d4"], title=f"{feature} vs Appliances",
    )
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df_model.select_dtypes(include=np.number).corr()
    fig3 = px.imshow(
        corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto",
        title="Correlation Matrix — Numerical Features",
    )
    fig3.update_layout(template="plotly_dark", height=650)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Strongest Correlations with Appliances")
    top_corr = corr["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False).head(10)
    fig4 = px.bar(
        top_corr, orientation="h", color=top_corr.values, color_continuous_scale="Viridis",
        title="Top 10 Correlated Features",
    )
    fig4.update_layout(template="plotly_dark", showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE: PREPROCESSING
# ----------------------------------------------------------------------
elif page == "🧪 Preprocessing":
    st.subheader("Preprocessing Pipeline")
    steps = [
        ("1. Parse date", "`date` converted to datetime; hour, day, month, day_of_week extracted."),
        ("2. Drop raw date", "Original `date` string column removed — models need numeric input."),
        ("3. Split features / target", "`X` = all columns except `Appliances`; `y` = `Appliances`."),
        (f"4. Train/test split", f"{int((1-test_size)*100)}% train / {int(test_size*100)}% test, random_state={random_state}."),
        ("5. Standard scaling", "`StandardScaler` fit on train only, then applied to test — prevents leakage."),
    ]
    for title, desc in steps:
        st.markdown(
            f"""<div class="metric-card fade-in" style="margin-bottom:0.6rem;">
                    <b style="color:#a5b4fc;">{title}</b><br>
                    <span style="color:#d1d5db;">{desc}</span>
                </div>""",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    metric_card(c1, "Training rows", f"{bundle['X_train'].shape[0]:,}")
    metric_card(c2, "Testing rows", f"{bundle['X_test'].shape[0]:,}")

    st.subheader("Scaled Feature Preview")
    scaled_preview = pd.DataFrame(
        bundle["scaler"].transform(bundle["X_train"].head()), columns=bundle["X"].columns
    )
    st.dataframe(scaled_preview, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE: MODEL TRAINING & COMPARISON
# ----------------------------------------------------------------------
elif page == "📈 Model Training & Comparison":
    st.subheader("Evaluation Metrics")
    cols = st.columns(3)
    for i, row in results.iterrows():
        is_best = row["Model"] == best_row["Model"]
        metric_card(
            cols[i], row["Model"] + (" 🏆" if is_best else ""),
            f"R² {row['R2']:.3f}", sub=f"MAE {row['MAE']:.1f} · RMSE {row['RMSE']:.1f}",
            good=is_best,
        )

    st.markdown("###")
    tab1, tab2, tab3 = st.tabs(["Actual vs Predicted", "Coefficient Comparison", "Alpha vs CV-MSE"])

    with tab1:
        model_choice = st.selectbox("Model", results["Model"].tolist())
        pred = bundle["preds"][model_choice]
        fig = px.scatter(
            x=bundle["y_test"], y=pred, opacity=0.4,
            labels={"x": "Actual Appliances (Wh)", "y": "Predicted Appliances (Wh)"},
            title=f"{model_choice}: Actual vs Predicted", color_discrete_sequence=["#f472b6"],
        )
        lims = [bundle["y_test"].min(), bundle["y_test"].max()]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect prediction",
                                  line=dict(color="#34d399", dash="dash")))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        coef_long = bundle["coef_table"].melt(id_vars="Feature", var_name="Model", value_name="Coefficient")
        fig = px.bar(
            coef_long, x="Feature", y="Coefficient", color="Model", barmode="group",
            title="Linear vs Lasso vs Ridge Coefficients",
            color_discrete_sequence=["#6366f1", "#f472b6", "#06b6d4"],
        )
        fig.update_layout(template="plotly_dark", xaxis_tickangle=-90, height=550)
        st.plotly_chart(fig, use_container_width=True)
        n_zero = (bundle["coef_table"]["Lasso"] == 0).sum()
        st.caption(f"Lasso zeroed out **{n_zero}** of {len(bundle['coef_table'])} features (automatic feature selection).")

    with tab3:
        c1, c2 = st.columns(2)
        for c, name, grid in [(c1, "Lasso", bundle["lasso_grid"]), (c2, "Ridge", bundle["ridge_grid"])]:
            cv_df = pd.DataFrame(grid.cv_results_)
            cv_df["CV_MSE"] = -cv_df["mean_test_score"]
            fig = px.line(
                cv_df, x="param_alpha", y="CV_MSE", markers=True, log_x=True,
                title=f"{name}: Alpha vs Cross-Validation MSE",
                color_discrete_sequence=["#f472b6" if name == "Lasso" else "#06b6d4"],
            )
            fig.add_vline(x=grid.best_params_["alpha"], line_dash="dash", line_color="#34d399",
                           annotation_text=f"best α={grid.best_params_['alpha']}")
            fig.update_layout(template="plotly_dark")
            c.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE: LIVE ALPHA TUNING
# ----------------------------------------------------------------------
elif page == "🎛️ Live Alpha Tuning":
    st.subheader("Adjust Alpha and Watch the Model React")
    st.caption("This retrains a single Lasso/Ridge model at your chosen alpha (outside the grid search) so you can feel the regularization effect directly.")

    model_type = st.radio("Regularization type", ["Lasso", "Ridge"], horizontal=True)
    alpha = st.slider("Alpha (regularization strength)", 0.001, 100.0, 1.0, step=0.1)

    Model = Lasso if model_type == "Lasso" else Ridge
    kwargs = {"max_iter": 10000} if model_type == "Lasso" else {}
    live_model = Model(alpha=alpha, **kwargs).fit(
        bundle["scaler"].transform(bundle["X_train"]), bundle["y_train"]
    )
    live_pred = live_model.predict(bundle["scaler"].transform(bundle["X_test"]))

    live_r2 = r2_score(bundle["y_test"], live_pred)
    live_mae = mean_absolute_error(bundle["y_test"], live_pred)
    live_rmse = np.sqrt(mean_squared_error(bundle["y_test"], live_pred))
    n_zero = (live_model.coef_ == 0).sum() if model_type == "Lasso" else 0

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "R²", f"{live_r2:.4f}")
    metric_card(c2, "MAE", f"{live_mae:.2f}")
    metric_card(c3, "RMSE", f"{live_rmse:.2f}")
    metric_card(c4, "Zeroed coefficients", f"{n_zero}" if model_type == "Lasso" else "—")

    coef_series = pd.Series(live_model.coef_, index=bundle["X"].columns).sort_values()
    fig = px.bar(
        coef_series, orientation="h", title=f"{model_type} Coefficients at alpha={alpha}",
        color=coef_series.values, color_continuous_scale="RdBu",
    )
    fig.update_layout(template="plotly_dark", height=700, coloraxis_showscale=False, transition_duration=300)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Drag the alpha slider toward 100 to watch coefficients shrink toward zero "
        "(and, for Lasso, actually hit zero). Drag it toward 0.001 to see the model "
        "converge back to plain Linear Regression behaviour."
    )

# ----------------------------------------------------------------------
# PAGE: PREDICT A NEW READING
# ----------------------------------------------------------------------
elif page == "🔮 Predict a New Reading":
    st.subheader("Simulate a New Sensor Reading")
    model_choice = st.selectbox("Model to use", results["Model"].tolist(), index=int(results["R2"].idxmax()))
    model = bundle["models"][model_choice]

    st.caption("Sliders default to the median of each feature — adjust the ones you care about.")
    key_features = ["T1", "RH_1", "T_out", "RH_out", "hour", "Windspeed"]
    key_features = [f for f in key_features if f in bundle["X"].columns]

    input_row = bundle["X"].median().copy()
    cols = st.columns(3)
    for i, feat in enumerate(key_features):
        lo, hi = float(bundle["X"][feat].min()), float(bundle["X"][feat].max())
        input_row[feat] = cols[i % 3].slider(feat, lo, hi, float(input_row[feat]))

    input_df = pd.DataFrame([input_row])[bundle["X"].columns]
    input_scaled = bundle["scaler"].transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("###")
    metric_card(st, "Predicted Appliance Energy Use", f"{prediction:.1f} Wh")
    st.caption(f"Prediction made using **{model_choice}**.")
