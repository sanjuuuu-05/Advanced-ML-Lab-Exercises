import time

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="AML Lab 1 | Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------
st.markdown("""
<style>
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(18px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 0 0 rgba(99,102,241,.45); }
    70% { box-shadow: 0 0 0 14px rgba(99,102,241,0); }
    100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
}
@keyframes floatCar {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
    100% { transform: translateY(0px); }
}
@keyframes shimmer {
    0% { background-position: -400px 0; }
    100% { background-position: 400px 0; }
}
@keyframes popIn {
    0% { opacity: 0; transform: scale(.85); }
    80% { opacity: 1; transform: scale(1.03); }
    100% { opacity: 1; transform: scale(1); }
}

html, body, [class*="css"] { scroll-behavior: smooth; }

.block-container {
    padding-top: 1.2rem;
    max-width: 1450px;
    animation: fadeIn .6s ease-in-out;
}

.hero {
    padding: 2.2rem 2.4rem;
    border-radius: 24px;
    color: white;
    background: linear-gradient(120deg,#0f172a,#312e81,#1e293b,#4338ca);
    background-size: 300% 300%;
    animation: gradientShift 10s ease infinite, fadeInUp .7s ease-out;
    margin-bottom: 1.3rem;
    position: relative;
    overflow: hidden;
    transition: transform .35s ease, box-shadow .35s ease;
}
.hero:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(67,56,202,.35);
}
.hero h1 {
    margin: 0;
    font-size: 2.6rem;
    display: inline-block;
    animation: floatCar 3.5s ease-in-out infinite;
}
.hero p { margin: .45rem 0 0; opacity: .85; animation: fadeIn 1.2s ease-in-out; }

.card {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 16px;
    background: rgba(128,128,128,.055);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeInUp .5s ease-out;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 26px rgba(0,0,0,.12);
    border-color: rgba(99,102,241,.45);
}

.prediction {
    padding: 1.8rem;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(99,102,241,.35);
    background: linear-gradient(135deg, rgba(99,102,241,.12), rgba(168,85,247,.10));
    animation: popIn .55s cubic-bezier(.34,1.56,.64,1), pulseGlow 2.2s ease-out 1;
}
.prediction .price {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg,#6366f1,#a855f7,#ec4899);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 3s ease infinite;
}
.small-muted { opacity: .7; font-size: .9rem; }

/* Metrics */
div[data-testid="stMetric"] {
    background: rgba(128,128,128,.06);
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 14px;
    padding: .7rem .9rem .5rem;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeInUp .55s ease-out;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px) scale(1.015);
    box-shadow: 0 10px 24px rgba(99,102,241,.18);
    border-color: rgba(99,102,241,.5);
}

/* Buttons */
.stButton > button {
    transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    border-radius: 12px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.015);
    box-shadow: 0 8px 20px rgba(99,102,241,.35);
    filter: brightness(1.05);
}
.stButton > button:active {
    transform: translateY(0px) scale(.98);
}

/* Tabs */
button[data-baseweb="tab"] {
    transition: color .2s ease, transform .2s ease;
}
button[data-baseweb="tab"]:hover {
    transform: translateY(-2px);
}
div[data-baseweb="tab-highlight"] {
    transition: left .3s cubic-bezier(.4,0,.2,1), width .3s cubic-bezier(.4,0,.2,1) !important;
}

/* Sliders / inputs subtle entrance */
div[data-testid="stSlider"], div[data-testid="stSelectbox"],
div[data-testid="stRadio"], div[data-testid="stNumberInput"] {
    animation: fadeInUp .5s ease-out;
}

/* Dataframe / expander */
div[data-testid="stExpander"] {
    transition: box-shadow .25s ease;
    border-radius: 12px;
    animation: fadeInUp .5s ease-out;
}
div[data-testid="stExpander"]:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    animation: fadeIn .6s ease-in-out;
}

/* Section headers get a little left-accent */
h2, h3 {
    position: relative;
    padding-left: .55rem;
}
h2::before, h3::before {
    content: "";
    position: absolute;
    left: 0; top: 8%;
    height: 84%;
    width: 4px;
    border-radius: 4px;
    background: linear-gradient(180deg,#6366f1,#a855f7);
}

.shimmer-bar {
    height: 6px;
    border-radius: 6px;
    background: linear-gradient(90deg,#6366f1 0%,#a855f7 25%,#6366f1 50%,#a855f7 75%,#6366f1 100%);
    background-size: 800px 100%;
    animation: shimmer 3.2s linear infinite;
    margin: .2rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"

# -------------------- Data/model helpers --------------------
@st.cache_data
def load_csv(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def preprocess(df):
    data = df.copy()

    if "car_ID" in data.columns:
        data = data.drop(columns=["car_ID"])

    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].median())

    for col in data.select_dtypes(include="object").columns:
        mode = data[col].mode()
        if not mode.empty:
            data[col] = data[col].fillna(mode.iloc[0])

    return data

@st.cache_data
def train_cached(data_json, test_size, random_state):
    df = pd.read_json(data_json)

    encoded = pd.get_dummies(df, drop_first=True)
    X = encoded.drop(columns=["price"])
    y = encoded["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)

    return {
        "model": model,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred": pred,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

def predict_new_car(model, df, X_columns, values):
    template = df.iloc[0].copy()

    for key, value in values.items():
        if key in template.index:
            template[key] = value

    new_df = pd.DataFrame([template])
    encoded = pd.get_dummies(new_df, drop_first=True)
    encoded = encoded.reindex(columns=X_columns, fill_value=0)

    return float(model.predict(encoded)[0])

def animated_metric_row(columns_specs):
    """columns_specs: list of (label, value_str). Renders with a tiny staggered reveal."""
    cols = st.columns(len(columns_specs))
    for i, (col, (label, value)) in enumerate(zip(cols, columns_specs)):
        with col:
            st.markdown(f'<div style="animation-delay:{i*0.08}s">', unsafe_allow_html=True)
            st.metric(label, value)
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
    <h1>🚗 Car Price Prediction</h1>
    <p>AML Lab Exercise 1 • Interactive Multiple Linear Regression</p>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Experiment Controls")

uploaded = st.sidebar.file_uploader(
    "Upload CarPrice_Assignment.csv",
    type=["csv"]
)

if uploaded is None:
    st.info("Upload **CarPrice_Assignment.csv** using the sidebar to begin.")
    st.markdown("""
    ### Interactive features

    📊 Explore the dataset with live, hoverable Plotly charts
    🔎 Adjust EDA visualizations in real time
    🎛️ Change train/test split
    🧠 Retrain the regression model
    📈 Inspect evaluation metrics with animated reveals
    🚗 Adjust a new car's specifications
    💰 Get an instant predicted price with a celebratory animation
    """)
    st.stop()

with st.spinner("Reading and cleaning your dataset..."):
    try:
        raw = load_csv(uploaded.getvalue())
        df = preprocess(raw)

        if "price" not in df.columns:
            st.error("The dataset must contain a `price` column.")
            st.stop()
    except Exception as e:
        st.error(f"Dataset error: {e}")
        st.stop()

# Model controls
st.sidebar.divider()
st.sidebar.subheader("🤖 Model Controls")

test_size_pct = st.sidebar.slider(
    "Testing data (%)",
    min_value=10,
    max_value=40,
    value=20,
    step=5,
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=999,
    value=42,
    step=1,
)

st.sidebar.caption(
    f"Training: {100-test_size_pct}%  |  Testing: {test_size_pct}%"
)

st.sidebar.progress(100 - test_size_pct, text=f"Train split preview: {100-test_size_pct}%")

test_size = test_size_pct / 100

with st.spinner("Training the regression model..."):
    results = train_cached(
        df.to_json(),
        test_size,
        int(random_state),
    )
    time.sleep(0.15)

st.markdown('<div class="shimmer-bar"></div>', unsafe_allow_html=True)

# -------------------- Global metrics --------------------
animated_metric_row([
    ("R² Score", f"{results['r2']:.4f}"),
    ("MAE", f"{results['mae']:,.0f}"),
    ("RMSE", f"{results['rmse']:,.0f}"),
    ("Test Records", str(len(results["y_test"]))),
])

tabs = st.tabs([
    "📊 Dashboard",
    "🔎 Interactive EDA",
    "🤖 Model Lab",
    "📈 Evaluation",
    "🚘 Price Predictor",
])

# -------------------- Dashboard --------------------
with tabs[0]:
    st.header("Dataset Dashboard")

    animated_metric_row([
        ("Records", str(len(raw))),
        ("Attributes", str(len(raw.columns))),
        ("Missing Values", str(int(raw.isnull().sum().sum()))),
        ("Duplicates", str(int(raw.duplicated().sum()))),
        ("Encoded Features", str(results["X"].shape[1])),
    ])

    st.subheader("Dataset Preview")

    preview_rows = st.slider(
        "Rows to display",
        5,
        min(50, len(raw)),
        5,
        key="preview_rows"
    )
    st.dataframe(raw.head(preview_rows), use_container_width=True)

    st.subheader("Attribute Information")

    info = pd.DataFrame({
        "Attribute": raw.columns,
        "Type": raw.dtypes.astype(str),
        "Non-Null": raw.notna().sum(),
        "Missing": raw.isna().sum(),
        "Unique": raw.nunique(),
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

    st.subheader("Numerical Attributes")
    st.write(df.select_dtypes(include=np.number).columns.tolist())

    st.subheader("Categorical Attributes")
    st.write(df.select_dtypes(include="object").columns.tolist())

    with st.expander("📋 Descriptive Statistics"):
        st.dataframe(raw.describe().T, use_container_width=True)

# -------------------- Interactive EDA --------------------
with tabs[1]:
    st.header("Interactive Exploratory Data Analysis")

    left, right = st.columns([1, 2])

    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c != "price"
    ]

    with left:
        st.subheader("🎛️ Visualization Controls")

        feature = st.selectbox(
            "Feature vs Price",
            numeric_cols,
            index=numeric_cols.index("horsepower")
            if "horsepower" in numeric_cols else 0
        )

        chart_type = st.radio(
            "Chart type",
            ["Scatter Plot", "Line Plot"],
            horizontal=True
        )

        point_size = st.slider(
            "Point size",
            min_value=6,
            max_value=40,
            value=14
        )

        opacity = st.slider(
            "Point opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.75,
            step=0.05
        )

        color_by = st.selectbox(
            "Color points by",
            ["None"] + df.select_dtypes(include="object").columns.tolist(),
        )

    with right:
        color_arg = None if color_by == "None" else color_by

        if chart_type == "Scatter Plot":
            fig = px.scatter(
                df,
                x=feature,
                y="price",
                color=color_arg,
                opacity=opacity,
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"{feature.title()} vs Price",
                trendline="ols" if color_arg is None else None,
            )
            fig.update_traces(marker=dict(size=point_size, line=dict(width=0)))
        else:
            sorted_df = df.sort_values(feature)
            fig = px.line(
                sorted_df,
                x=feature,
                y="price",
                markers=True,
                template=PLOTLY_TEMPLATE,
                title=f"{feature.title()} vs Price",
            )
            fig.update_traces(opacity=opacity, line=dict(width=2.5))

        fig.update_layout(
            transition_duration=400,
            transition_easing="cubic-in-out",
            margin=dict(t=60, l=10, r=10, b=10),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    st.divider()

    st.subheader("💰 Price Distribution")

    bins = st.slider(
        "Number of histogram bins",
        min_value=5,
        max_value=60,
        value=20
    )

    show_marginal = st.checkbox("Show box-plot margin (like a KDE companion)", value=True)

    fig = px.histogram(
        df,
        x="price",
        nbins=bins,
        template=PLOTLY_TEMPLATE,
        marginal="box" if show_marginal else None,
        color_discrete_sequence=["#a855f7"],
        title="Distribution of Car Prices",
    )
    fig.update_layout(
        transition_duration=400,
        bargap=0.05,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.divider()

    st.subheader("🔥 Interactive Correlation Heatmap")

    corr_cols = st.multiselect(
        "Select attributes for correlation analysis",
        df.select_dtypes(include=np.number).columns.tolist(),
        default=[
            c for c in [
                "price",
                "horsepower",
                "enginesize",
                "curbweight",
                "citympg",
                "highwaympg"
            ] if c in df.columns
        ]
    )

    if len(corr_cols) >= 2:
        corr_matrix = df[corr_cols].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            template=PLOTLY_TEMPLATE,
            title="Selected Feature Correlations",
            aspect="auto",
        )
        fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True, theme=None)
    else:
        st.warning("Select at least two attributes.")

    st.subheader("📌 Price Correlation Ranking")

    price_corr = (
        df.select_dtypes(include=np.number)
        .corr()["price"]
        .drop("price")
        .sort_values()
    )

    fig = px.bar(
        price_corr,
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color=price_corr.values,
        color_continuous_scale="Viridis",
        title="Correlation of Numerical Attributes with Price",
        labels={"value": "Correlation", "index": "Attribute"},
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        margin=dict(t=60, l=10, r=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

# -------------------- Model Lab --------------------
with tabs[2]:
    st.header("Interactive Regression Model Lab")

    st.markdown("""
    Change the **testing percentage** or **random state** in the sidebar.
    The model automatically retrains with the new settings.
    """)

    animated_metric_row([
        ("Training Records", str(len(results["X_train"]))),
        ("Testing Records", str(len(results["X_test"]))),
        ("Features", str(results["X"].shape[1])),
    ])

    st.subheader("Train/Test Split")

    split_data = pd.DataFrame({
        "Dataset": ["Training", "Testing"],
        "Records": [len(results["X_train"]), len(results["X_test"])]
    })

    fig = px.pie(
        split_data,
        names="Dataset",
        values="Records",
        hole=0.55,
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#6366f1", "#ec4899"],
        title="Train / Test Split",
    )
    fig.update_traces(
        textinfo="percent+label",
        pull=[0.02, 0.02],
        marker=dict(line=dict(color="#0f172a", width=2)),
    )
    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.subheader("Model Coefficients")

    coef_df = pd.DataFrame({
        "Feature": results["X"].columns,
        "Coefficient": results["model"].coef_
    })

    coef_df["Absolute Impact"] = coef_df["Coefficient"].abs()
    coef_df = coef_df.sort_values("Absolute Impact", ascending=False)

    show_n = st.slider(
        "Number of coefficients to display",
        min_value=5,
        max_value=min(25, len(coef_df)),
        value=min(10, len(coef_df))
    )

    top_coefs = coef_df.head(show_n).sort_values("Coefficient")
    fig = px.bar(
        top_coefs,
        x="Coefficient",
        y="Feature",
        orientation="h",
        template=PLOTLY_TEMPLATE,
        color="Coefficient",
        color_continuous_scale="RdBu",
        title="Top Coefficients by Absolute Impact",
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=60, l=10, r=10, b=10),
        transition_duration=400,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    with st.expander("📋 View coefficient table"):
        st.dataframe(coef_df.head(show_n), use_container_width=True, hide_index=True)

    st.subheader("Prediction Preview")

    comparison = pd.DataFrame({
        "Actual Price": results["y_test"].values,
        "Predicted Price": results["pred"]
    })

    comparison["Error"] = (
        comparison["Actual Price"] -
        comparison["Predicted Price"]
    )

    st.dataframe(
        comparison.head(
            st.slider("Prediction rows", 5, min(50, len(comparison)), 10)
        ),
        use_container_width=True
    )

# -------------------- Evaluation --------------------
with tabs[3]:
    st.header("Interactive Model Evaluation")

    metrics = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R² Score"],
        "Value": [
            results["mae"],
            results["mse"],
            results["rmse"],
            results["r2"]
        ]
    })

    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.subheader("🎯 Actual vs Predicted Price")

    show_reference = st.checkbox(
        "Show perfect-prediction reference line",
        value=True
    )

    eval_df = pd.DataFrame({
        "Actual": results["y_test"].values,
        "Predicted": results["pred"],
    })

    fig = px.scatter(
        eval_df,
        x="Actual",
        y="Predicted",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=["#22d3ee"],
        opacity=0.8,
        title="Actual Price vs Predicted Price",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0)))

    if show_reference:
        lo = float(min(eval_df["Actual"].min(), eval_df["Predicted"].min()))
        hi = float(max(eval_df["Actual"].max(), eval_df["Predicted"].max()))
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Perfect Prediction",
        ))

    fig.update_layout(margin=dict(t=60, l=10, r=10, b=10), transition_duration=400)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.subheader("📉 Residual / Prediction Error Plot")

    residual_df = pd.DataFrame({
        "Predicted": results["pred"],
        "Residual": results["y_test"].values - results["pred"],
    })

    fig = px.scatter(
        residual_df,
        x="Predicted",
        y="Residual",
        template=PLOTLY_TEMPLATE,
        color="Residual",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Prediction Errors / Residual Plot",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0)))
    fig.add_hline(y=0, line_dash="dash", line_color="#ef4444")
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=60, l=10, r=10, b=10),
        transition_duration=400,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.subheader("Metric Interpretation")

    st.info(
        f"MAE = {results['mae']:,.2f} | "
        f"RMSE = {results['rmse']:,.2f} | "
        f"R² = {results['r2']:.4f}"
    )

    st.caption(
        "Lower MAE/RMSE indicate smaller prediction errors. "
        "A higher R² indicates that a larger proportion of price variation "
        "is explained by the model."
    )

# -------------------- New Car Predictor --------------------
with tabs[4]:
    st.header("🚘 Interactive New Car Price Predictor")

    st.markdown("""
    Adjust the specifications below and click **Predict Price**.
    The model uses the same preprocessing and trained regression model used
    in the experiment.
    """)

    values = {}

    priority = [
        "enginesize",
        "horsepower",
        "curbweight",
        "citympg",
        "highwaympg"
    ]

    available = [c for c in priority if c in df.columns]

    st.subheader("⚙️ Main Car Specifications")

    cols = st.columns(3)

    for i, col in enumerate(available):
        minimum = float(df[col].min())
        maximum = float(df[col].max())
        median = float(df[col].median())

        if pd.api.types.is_integer_dtype(df[col]):
            values[col] = cols[i % 3].slider(
                col,
                min_value=int(minimum),
                max_value=int(maximum),
                value=int(median),
                key=f"main_{col}"
            )
        else:
            values[col] = cols[i % 3].slider(
                col,
                min_value=float(minimum),
                max_value=float(maximum),
                value=float(median),
                key=f"main_{col}"
            )

    other_numeric = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in available and c != "price"
    ]

    if other_numeric:
        with st.expander("🔧 Adjust Other Numerical Attributes"):
            cols = st.columns(3)

            for i, col in enumerate(other_numeric):
                minimum = float(df[col].min())
                maximum = float(df[col].max())
                median = float(df[col].median())

                if pd.api.types.is_integer_dtype(df[col]):
                    values[col] = cols[i % 3].slider(
                        col,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(median),
                        key=f"other_{col}"
                    )
                else:
                    values[col] = cols[i % 3].slider(
                        col,
                        min_value=minimum,
                        max_value=maximum,
                        value=median,
                        key=f"other_{col}"
                    )

    categorical = df.select_dtypes(include="object").columns.tolist()

    if categorical:
        st.subheader("🎨 Car Categories")

        cols = st.columns(3)

        for i, col in enumerate(categorical):
            options = df[col].dropna().unique().tolist()

            if not options:
                continue

            mode = df[col].mode()
            default = mode.iloc[0] if not mode.empty else options[0]
            index = options.index(default) if default in options else 0

            values[col] = cols[i % 3].selectbox(
                col,
                options,
                index=index,
                key=f"select_{col}"
            )

    st.divider()

    if st.button(
        "🚗 Predict Car Price",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Running the trained model..."):
            time.sleep(0.35)
            try:
                prediction = predict_new_car(
                    results["model"],
                    df,
                    results["X"].columns,
                    values
                )

                st.markdown(
                    f"""
                    <div class="prediction">
                        <div class="small-muted">Estimated Selling Price</div>
                        <div class="price">{prediction:,.2f}</div>
                        <div class="small-muted">Based on the trained Multiple Linear Regression model</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.balloons()

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.divider()

    st.subheader("📋 Current Car Configuration")

    config = pd.DataFrame({
        "Attribute": list(values.keys()),
        "Selected Value": list(values.values())
    })

    st.dataframe(config, use_container_width=True, hide_index=True)

# -------------------- Footer --------------------
st.divider()
st.caption(
    "AML Lab Exercise 1 • Car Price Prediction Using Regression • "
    "Interactive Streamlit Implementation"
)
