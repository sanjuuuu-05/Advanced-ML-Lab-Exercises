"""
Lab Exercise 4 — PCA, LDA and ICA  (interactive Plotly edition)
Run:  streamlit run app.py
Needs: streamlit>=1.50, plotly>=5.18, scikit-learn>=1.1, pandas, numpy, scipy
Tip : keep .streamlit/config.toml next to this file (dark theme) for the intended look.
"""
import warnings
from time import perf_counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import kurtosis
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="PCA, LDA & ICA Lab", page_icon="🧬", layout="wide",
                   initial_sidebar_state="expanded")

# ──────────────────────────────────────────────────────────────────────────────
# Look & feel: CSS animations + one shared Plotly template
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html { scroll-behavior: smooth; }
.stApp {
    font-family: 'Inter', sans-serif;
    background:
        radial-gradient(1100px 560px at 8% -8%, rgba(124,92,255,.20), transparent 60%),
        radial-gradient(900px 520px at 100% 0%, rgba(56,224,196,.13), transparent 55%),
        radial-gradient(800px 500px at 50% 120%, rgba(255,93,143,.10), transparent 60%),
        #0b0f1a;
    color: #e8ecf8;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] { color: #e8ecf8; }
header[data-testid="stHeader"] { background: transparent; }

@keyframes fadeUp   { from { opacity: 0; transform: translateY(16px) scale(.985); } to { opacity: 1; transform: none; } }
@keyframes fadeIn   { from { opacity: 0; } to { opacity: 1; } }
@keyframes gradShift{ 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes floaty   { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes glowPulse{ 0%,100% { box-shadow: 0 0 0 0 rgba(124,92,255,.0); } 50% { box-shadow: 0 0 34px 2px rgba(124,92,255,.22); } }

.hero {
    position: relative; overflow: hidden;
    padding: 1.9rem 2.1rem; margin-bottom: 1.1rem; border-radius: 24px;
    background: linear-gradient(135deg, rgba(20,26,46,.92), rgba(14,18,34,.92));
    border: 1px solid rgba(255,255,255,.09);
    animation: fadeUp .9s cubic-bezier(.2,.8,.2,1) both, glowPulse 6s ease-in-out infinite;
}
.hero::after {
    content: ""; position: absolute; inset: -40% -10% auto auto; width: 420px; height: 420px;
    background: radial-gradient(circle, rgba(124,92,255,.35), transparent 65%);
    filter: blur(20px); animation: floaty 7s ease-in-out infinite; pointer-events: none;
}
.hero .badge {
    display: inline-block; padding: .28rem .8rem; border-radius: 999px; font-size: .72rem;
    letter-spacing: .12em; font-weight: 700; color: #9fe9dc;
    background: rgba(56,224,196,.10); border: 1px solid rgba(56,224,196,.35);
}
.hero h1 {
    margin: .7rem 0 .2rem 0; font-size: 2.5rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(100deg, #8b7cff, #38e0c4, #ff5d8f, #8b7cff);
    background-size: 300% 100%; -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; animation: gradShift 9s ease infinite;
}
.hero p { margin: 0; color: #aab3d1 !important; font-size: 1.02rem; }

div[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
    border: 1px solid rgba(255,255,255,.09); border-radius: 18px; padding: 14px 18px;
    animation: fadeUp .7s cubic-bezier(.2,.8,.2,1) both;
    transition: transform .28s cubic-bezier(.2,.8,.2,1), box-shadow .28s, border-color .28s;
}
div[data-testid="stMetric"]:hover { transform: translateY(-5px); border-color: rgba(124,92,255,.55);
    box-shadow: 0 14px 34px rgba(124,92,255,.25); }

div[data-baseweb="tab-list"] { gap: 6px; background: rgba(255,255,255,.03); padding: 6px; border-radius: 16px;
    border: 1px solid rgba(255,255,255,.06); }
button[data-baseweb="tab"] { border-radius: 12px !important; padding: 8px 16px !important;
    transition: background .3s, color .3s, transform .25s; }
button[data-baseweb="tab"]:hover { background: rgba(124,92,255,.14) !important; transform: translateY(-1px); }
button[data-baseweb="tab"][aria-selected="true"] { background: rgba(124,92,255,.24) !important; }
div[data-baseweb="tab-highlight"] { transition: all .45s cubic-bezier(.4,0,.2,1); background: linear-gradient(90deg,#8b7cff,#38e0c4) !important; }
div[data-baseweb="tab-panel"] { animation: fadeUp .6s cubic-bezier(.2,.8,.2,1) both; padding-top: 1rem; }

div[data-testid="stPlotlyChart"] {
    border-radius: 20px; border: 1px solid rgba(255,255,255,.07); background: rgba(255,255,255,.022);
    padding: 6px; animation: fadeUp .8s cubic-bezier(.2,.8,.2,1) both;
    transition: border-color .35s, box-shadow .35s, transform .35s;
}
div[data-testid="stPlotlyChart"]:hover { border-color: rgba(124,92,255,.4); box-shadow: 0 18px 44px rgba(0,0,0,.35); }
div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; animation: fadeIn .8s both;
    border: 1px solid rgba(255,255,255,.07); }
div[data-testid="stExpander"] { border-radius: 16px; border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.025); transition: border-color .3s, transform .3s; animation: fadeUp .6s both; }
div[data-testid="stExpander"]:hover { border-color: rgba(124,92,255,.45); }
div[data-testid="stAlert"] { border-radius: 16px; animation: fadeUp .6s both; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(20,26,46,.97), rgba(12,16,30,.97));
    border-right: 1px solid rgba(255,255,255,.07); }
section[data-testid="stSidebar"] > div { animation: fadeIn .9s both; }
.stButton button, div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    transition: transform .2s, box-shadow .25s, border-color .25s; }
.stButton button:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(124,92,255,.3); }
div[data-testid="stSlider"] div[role="slider"] { transition: transform .2s, box-shadow .2s; }
div[data-testid="stSlider"] div[role="slider"]:hover { transform: scale(1.25); box-shadow: 0 0 0 8px rgba(124,92,255,.18); }

.step-chip { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .75rem; font-weight: 700;
    letter-spacing: .08em; color: #c9c0ff; background: rgba(124,92,255,.14); border: 1px solid rgba(124,92,255,.4);
    margin-bottom: .4rem; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)

tpl = go.layout.Template(pio.templates["plotly_dark"])
tpl.layout.update(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#dbe2ff", size=13),
    colorway=["#8b7cff", "#38e0c4", "#ff5d8f", "#ffb347", "#5ab0ff", "#c3f26b"],
    margin=dict(l=48, r=24, t=64, b=48),
    hoverlabel=dict(bgcolor="#141a2e", bordercolor="#8b7cff", font_size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    transition=dict(duration=600, easing="cubic-in-out"),
    title=dict(x=0.02, font=dict(size=17)),
)
tpl.layout.xaxis.update(gridcolor="rgba(255,255,255,.07)", zeroline=False, linecolor="rgba(255,255,255,.15)")
tpl.layout.yaxis.update(gridcolor="rgba(255,255,255,.07)", zeroline=False, linecolor="rgba(255,255,255,.15)")
pio.templates["lab"] = tpl
pio.templates.default = "lab"

CLS = {"malignant": "#ff5d8f", "benign": "#38e0c4"}
CFG = {"displaylogo": False, "responsive": True,
       "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}
PLAY = lambda ms, redraw=False: dict(frame=dict(duration=ms, redraw=redraw),
                                     transition=dict(duration=ms, easing="linear"),
                                     fromcurrent=True, mode="immediate")
STOP = dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")
BTN_STYLE = dict(bgcolor="#1b2240", bordercolor="#8b7cff", font=dict(color="#e8ecf8"))

# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = load_breast_cancer()
    return pd.DataFrame(data.data, columns=data.feature_names), data.target, data.target_names


X, y, names = load_data()
labels = pd.Series(y).map(dict(enumerate(names)))

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Experiment settings")
method = st.sidebar.selectbox("Technique", ["PCA", "LDA", "ICA", "Compare all"])
scale = st.sidebar.toggle("Standardise features (StandardScaler)", value=True,
                          help="Turn this off to see why scaling matters for PCA / ICA.")
n_ica = st.sidebar.slider("ICA components", 2, 10, 2)
seed = st.sidebar.number_input("Random state", 0, 999, 42)
st.sidebar.caption("Every control re-fits the models instantly (results are cached).")


@st.cache_data(show_spinner="Fitting PCA, LDA and ICA …")
def run_models(X, y, scale, n_ica, seed):
    Xin = StandardScaler().fit_transform(X) if scale else X.to_numpy()
    t0 = perf_counter()
    pca = PCA(n_components=2, random_state=seed)
    Zp = pca.fit_transform(Xin)
    t_pca = perf_counter() - t0
    t0 = perf_counter()
    lda = LinearDiscriminantAnalysis(n_components=1)
    Zl = lda.fit_transform(Xin, y)
    t_lda = perf_counter() - t0
    t0 = perf_counter()
    ica = FastICA(n_components=n_ica, random_state=seed, max_iter=2000, tol=0.001, whiten="unit-variance")
    Zi = ica.fit_transform(Xin)
    t_ica = perf_counter() - t0
    pca_full = PCA(random_state=seed).fit(Xin)
    return dict(Xin=Xin, pca=pca, Zp=Zp, t_pca=t_pca, lda=lda, Zl=Zl, t_lda=t_lda,
                ica=ica, Zi=Zi, t_ica=t_ica, pca_full=pca_full)


res = run_models(X, y, scale, n_ica, seed)
Xin, pca, Zp, lda, Zl, ica, Zi, pca_full = (res[k] for k in ["Xin", "pca", "Zp", "lda", "Zl", "ica", "Zi", "pca_full"])

# ──────────────────────────────────────────────────────────────────────────────
# Hero + KPI row
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="badge">LAB EXERCISE 4 · MAI511-2 · ADVANCED MACHINE LEARNING</span>
  <h1>PCA · LDA · ICA</h1>
  <p>Comparative dimensionality reduction on the Breast Cancer Wisconsin dataset — fully interactive, animated and Plotly-powered.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", X.shape[0])
c2.metric("Original features", X.shape[1])
c3.metric("Classes", len(names))
c4.metric("Preprocessing", "StandardScaler ✓" if scale else "None (raw)")

tabs = st.tabs(["🧼 Preprocessing", "🎯 Projections", "🎞️ Morph animation", "📉 PCA deep-dive",
                "🔬 ICA & LDA", "⚖️ Comparison", "📘 About & Viva"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<span class="step-chip">STEP 1 · LOAD & INSPECT</span>', unsafe_allow_html=True)
    a, b = st.columns([3, 2])
    with a:
        st.dataframe(X.head(8), use_container_width=True)
        q1, q2, q3 = st.columns(3)
        q1.metric("Missing values", int(X.isna().sum().sum()))
        q2.metric("Duplicate rows", int(X.duplicated().sum()))
        q3.metric("Constant features", int((X.nunique() <= 1).sum()))
    with b:
        counts = labels.value_counts()
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.64, sort=False,
                               marker=dict(colors=[CLS[k] for k in counts.index], line=dict(color="#0b0f1a", width=3)),
                               pull=[.04] * len(counts), textinfo="label+percent", direction="clockwise"))
        fig.update_layout(title="Class distribution", height=340, showlegend=False,
                          annotations=[dict(text=f"<b>{len(X)}</b><br>samples", showarrow=False, font=dict(size=18))])
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pre_pie")

    st.markdown('<span class="step-chip">STEP 2 · FEATURE-SCALE DISPARITY</span>', unsafe_allow_html=True)
    raw_std = X.std(ddof=0)
    fig = go.Figure()
    fig.add_bar(x=raw_std.index, y=raw_std.values, name="Raw std-dev",
                marker=dict(color=np.log10(raw_std.values + 1e-9), colorscale=[[0, "#38e0c4"], [1, "#8b7cff"]]))
    fig.add_scatter(x=raw_std.index, y=np.ones(len(raw_std)), mode="lines+markers", name="After StandardScaler (= 1)",
                    line=dict(color="#ffb347", width=3), marker=dict(size=6))
    fig.update_yaxes(type="log", title="Standard deviation (log scale)")
    fig.update_xaxes(tickangle=-50)
    fig.update_layout(title="Raw features span several orders of magnitude", height=430,
                      legend=dict(orientation="h", y=1.12, x=0.02))
    st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pre_std")
    st.caption("PCA, LDA-free distances and ICA whitening are all scale-sensitive: a feature measured in hundreds (area) "
               "would otherwise swamp one measured in hundredths (smoothness).")

    st.markdown('<span class="step-chip">STEP 3 · STANDARDISE  z = (x − μ) / σ</span>', unsafe_allow_html=True)
    Xs_all = StandardScaler().fit_transform(X)
    feat = st.selectbox("Inspect a feature before vs after scaling", list(X.columns), index=list(X.columns).index("mean area"))
    j = list(X.columns).index(feat)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Raw", "Standardised"), horizontal_spacing=.09)
    for i, name in enumerate(names):
        fig.add_histogram(x=X.loc[y == i, feat], name=name, marker_color=CLS[name], opacity=.7,
                          nbinsx=35, row=1, col=1, legendgroup=name)
        fig.add_histogram(x=Xs_all[y == i, j], name=name, marker_color=CLS[name], opacity=.7,
                          nbinsx=35, row=1, col=2, legendgroup=name, showlegend=False)
    fig.update_layout(barmode="overlay", height=360, title=f"“{feat}” — same shape, new scale",
                      legend=dict(orientation="h", y=1.14, x=0.02))
    st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pre_hist")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Raw mean", f"{X[feat].mean():.3f}")
    m2.metric("Raw std", f"{X[feat].std(ddof=0):.3f}")
    m3.metric("Scaled mean", f"{Xs_all[:, j].mean():.4f}")
    m4.metric("Scaled std", f"{Xs_all[:, j].std():.4f}")
    st.dataframe(pd.DataFrame({"Raw mean": X.mean().head(5), "Raw std": X.std(ddof=0).head(5),
                               "Scaled mean": np.round(Xs_all.mean(axis=0)[:5], 4),
                               "Scaled std": np.round(Xs_all.std(axis=0)[:5], 4)}), use_container_width=True)
    st.info("For this dimensionality-reduction comparison the scaler is fitted on the whole dataset so the projections are "
            "easy to reproduce. In a predictive workflow, fit it on the training split only to avoid data leakage.")

    st.markdown('<span class="step-chip">STEP 4 · FEATURE CORRELATION</span>', unsafe_allow_html=True)
    fig = px.imshow(X.corr(), color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=640, title="Strong correlation between features is what PCA compresses",
                      coloraxis_colorbar=dict(thickness=12, len=.8))
    st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pre_corr")

    st.markdown('<span class="step-chip">STEP 5 · WHY SCALING MATTERS — LIVE PROOF</span>', unsafe_allow_html=True)
    ev_raw = PCA().fit(X.to_numpy()).explained_variance_ratio_[:6]
    ev_scl = PCA().fit(Xs_all).explained_variance_ratio_[:6]
    fig = go.Figure()
    fig.add_bar(x=[f"PC{i}" for i in range(1, 7)], y=ev_raw, name="Without scaling", marker_color="#ff5d8f")
    fig.add_bar(x=[f"PC{i}" for i in range(1, 7)], y=ev_scl, name="With StandardScaler", marker_color="#38e0c4")
    fig.update_layout(barmode="group", height=380, title="Explained variance ratio — first six components",
                      legend=dict(orientation="h", y=1.12, x=0.02))
    fig.update_yaxes(title="Explained variance ratio", tickformat=".0%")
    p1, p2 = st.columns([2, 1])
    with p1:
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pre_why")
    with p2:
        st.metric("PC1 share — raw data", f"{ev_raw[0] * 100:.2f}%")
        st.metric("PC1 share — standardised", f"{ev_scl[0] * 100:.2f}%")
        st.caption("Without scaling, PC1 is dominated by the few large-valued features instead of reflecting shared structure.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    o1, o2, o3 = st.columns([1, 1, 2])
    show_ell = o1.toggle("95% confidence ellipses", value=True)
    ic_x, ic_y = 0, 1
    if n_ica > 2:
        ic_x = o2.selectbox("ICA x-axis", range(n_ica), index=0, format_func=lambda k: f"IC{k + 1}")
        ic_y = o3.selectbox("ICA y-axis", range(n_ica), index=1, format_func=lambda k: f"IC{k + 1}")
    choices = ["PCA", "LDA", "ICA"] if method == "Compare all" else [method]
    cols = st.columns(len(choices))
    for col, choice in zip(cols, choices):
        with col:
            fig = go.Figure()
            if choice in ("PCA", "ICA"):
                Z2 = Zp if choice == "PCA" else Zi[:, [ic_x, ic_y]]
                ax_x, ax_y = ("PC1", "PC2") if choice == "PCA" else (f"IC{ic_x + 1}", f"IC{ic_y + 1}")
                for i, name in enumerate(names):
                    m = y == i
                    fig.add_scatter(x=Z2[m, 0], y=Z2[m, 1], mode="markers", name=name,
                                    marker=dict(size=8, color=CLS[name], opacity=.8, line=dict(width=.6, color="white")),
                                    customdata=np.where(m)[0],
                                    hovertemplate="sample %{customdata}<br>" + ax_x + " = %{x:.2f}<br>" + ax_y + " = %{y:.2f}<extra>" + name + "</extra>")
                    if show_ell:
                        mu = Z2[m].mean(axis=0)
                        vals, vecs = np.linalg.eigh(np.cov(Z2[m].T))
                        t = np.linspace(0, 2 * np.pi, 120)
                        ell = mu + (np.c_[np.cos(t), np.sin(t)] * np.sqrt(vals * 5.991)) @ vecs.T
                        fig.add_scatter(x=ell[:, 0], y=ell[:, 1], mode="lines", showlegend=False, hoverinfo="skip",
                                        line=dict(color=CLS[name], width=2, dash="dot"))
                ttl = "PCA — first two principal components" if choice == "PCA" else "ICA — two independent components"
                fig.update_xaxes(title=ax_x)
                fig.update_yaxes(title=ax_y)
            else:
                for i, name in enumerate(names):
                    fig.add_histogram(x=Zl[y == i, 0], histnorm="probability density", nbinsx=32, name=name,
                                      marker_color=CLS[name], opacity=.62)
                    fig.add_vline(x=Zl[y == i, 0].mean(), line=dict(color=CLS[name], dash="dash", width=2))
                fig.update_layout(barmode="overlay")
                fig.update_xaxes(title="LD1")
                fig.update_yaxes(title="Density")
                ttl = "LDA — discriminant score distribution"
            fig.update_layout(title=ttl, height=470, legend=dict(orientation="h", y=1.1, x=0.02))
            st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key=f"proj_{choice}")
    if method in ("LDA", "Compare all"):
        st.caption("LDA can return at most (classes − 1) components, so two classes give exactly one discriminant axis.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MORPH ANIMATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("**Watch the same 569 patients glide from raw feature space → PCA → ICA → LDA.** "
                "Press ▶ Play, drag the slider, or reset. Each stage is re-centred and rescaled so the motion is comparable.")
    k1, k2 = st.columns(2)
    morph_steps = k1.slider("Frames per transition", 8, 40, 22)
    frame_ms = k2.slider("Frame duration (ms)", 20, 140, 45)

    rng = np.random.default_rng(seed)
    short = ["Raw", "PCA", "ICA", "LDA"]
    raw2 = Xin[:, :2]
    lda2 = np.c_[Zl[:, 0], rng.normal(0, 1, len(y))]
    stage_titles = ["Raw features (mean radius vs mean texture)", "PCA — PC1 vs PC2",
                    "ICA — IC1 vs IC2", "LDA — LD1 (vertical jitter only for visibility)"]
    stage_data = []
    for Z in [raw2, Zp[:, :2], Zi[:, :2], lda2]:
        Zn = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        stage_data.append(Zn)
    stage_data[3][:, 1] *= 0.35
    lim = max(np.abs(Z).max() for Z in stage_data) * 1.08

    hold = 7
    fr_defs = [(stage_data[0], stage_titles[0], short[0])] + [(stage_data[0], stage_titles[0], "")] * (hold - 1)
    for s in range(3):
        A, B = stage_data[s], stage_data[s + 1]
        for jj in range(1, morph_steps + 1):
            tt = jj / morph_steps
            e = tt * tt * (3 - 2 * tt)
            fr_defs.append(((1 - e) * A + e * B,
                            f"{short[s]}  →  {short[s + 1]}" if jj < morph_steps else stage_titles[s + 1],
                            short[s + 1] if jj == morph_steps else ""))
        fr_defs += [(B, stage_titles[s + 1], "")] * (hold - 1)

    frames, slider_steps = [], []
    for kf, (Z, ttl, lab) in enumerate(fr_defs):
        frames.append(go.Frame(name=str(kf), traces=[0, 1], layout=dict(title=dict(text=ttl)),
                               data=[go.Scatter(x=Z[y == 0, 0], y=Z[y == 0, 1]),
                                     go.Scatter(x=Z[y == 1, 0], y=Z[y == 1, 1])]))
        slider_steps.append(dict(method="animate", label=lab,
                                 args=[[str(kf)], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                                       transition=dict(duration=0))]))
    Z0 = fr_defs[0][0]
    fig = go.Figure(data=[go.Scatter(x=Z0[y == i, 0], y=Z0[y == i, 1], mode="markers", name=names[i],
                                     marker=dict(size=8, color=CLS[names[i]], opacity=.8, line=dict(width=.6, color="white")))
                          for i in range(2)], frames=frames)
    fig.update_layout(
        title=dict(text=stage_titles[0]), height=640, margin=dict(b=150),
        xaxis=dict(range=[-lim, lim], title="axis 1 (standardised)"), yaxis=dict(range=[-lim, lim], title="axis 2 (standardised)"),
        legend=dict(orientation="h", y=1.08, x=0.02),
        updatemenus=[dict(type="buttons", direction="left", x=0.5, y=0, xanchor="center", yanchor="top",
                          pad=dict(t=95, r=10), showactive=False, **BTN_STYLE,
                          buttons=[dict(label="▶  Play", method="animate", args=[None, PLAY(frame_ms)]),
                                   dict(label="⏸  Pause", method="animate", args=[[None], STOP]),
                                   dict(label="↺  Reset", method="animate", args=[["0"], STOP])])],
        sliders=[dict(active=0, steps=slider_steps, x=0.04, len=0.92, xanchor="left", y=0, yanchor="top",
                      pad=dict(t=40, b=10), currentvalue=dict(visible=False), transition=dict(duration=0),
                      bgcolor="#1b2240", bordercolor="#8b7cff", activebgcolor="#8b7cff", ticklen=0,
                      font=dict(color="#c9c0ff", size=13))])
    st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="morph")
    st.caption("The in-between positions are linear interpolations (with ease-in-out) between coordinate systems — "
               "an illustration of how each method re-expresses the same data, not a trajectory the algorithms follow.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PCA DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    ev = pca_full.explained_variance_ratio_
    cum = np.cumsum(ev)
    n_pc = len(ev)
    xs = np.arange(1, n_pc + 1)
    n95 = int(np.argmax(cum >= .95) + 1)

    st.subheader("Explained variance")
    r1, r2 = st.columns([3, 2])
    with r1:
        fig = go.Figure()
        fig.add_bar(x=xs, y=ev, name="Per component", marker=dict(color=ev, colorscale=[[0, "#2c2670"], [1, "#8b7cff"]]))
        fig.add_scatter(x=xs, y=cum, mode="lines+markers", name="Cumulative",
                        line=dict(color="#ffb347", width=3, shape="spline"), marker=dict(size=6))
        fig.add_hline(y=.95, line=dict(color="#38e0c4", dash="dash"), annotation_text="95 %", annotation_position="bottom right")
        fig.add_vline(x=n95, line=dict(color="#ff5d8f", dash="dot"), annotation_text=f"{n95} PCs", annotation_position="top left")
        fig.frames = [go.Frame(name=str(kk), traces=[1], data=[go.Scatter(x=xs[:kk], y=cum[:kk])]) for kk in range(1, n_pc + 1)]
        fig.update_layout(title="Scree + cumulative variance", height=470, margin=dict(b=100),
                          legend=dict(orientation="h", y=1.1, x=0.02),
                          updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02, xanchor="left", yanchor="top",
                                            pad=dict(t=50), showactive=False, **BTN_STYLE,
                                            buttons=[dict(label="▶  Replay build-up", method="animate",
                                                          args=[None, dict(frame=dict(duration=90, redraw=False),
                                                                           transition=dict(duration=90, easing="cubic-in-out"),
                                                                           fromcurrent=False, mode="immediate")])])])
        fig.update_xaxes(title="Principal component")
        fig.update_yaxes(title="Explained variance ratio", tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pca_scree")
    with r2:
        k_keep = st.slider("Components to keep", 1, n_pc, 2)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=float(cum[k_keep - 1] * 100),
            number=dict(suffix=" %", valueformat=".2f"), delta=dict(reference=95, valueformat=".2f", suffix=" vs 95 %"),
            title=dict(text=f"Variance kept by {k_keep} PC{'s' if k_keep > 1 else ''}"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#8b7cff", thickness=.3), bgcolor="rgba(255,255,255,.04)",
                       steps=[dict(range=[0, 95], color="rgba(255,255,255,.05)"), dict(range=[95, 100], color="rgba(56,224,196,.28)")],
                       threshold=dict(line=dict(color="#ffb347", width=4), thickness=.8, value=95))))
        fig.update_layout(height=340, margin=dict(t=90, b=20))
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pca_gauge")
        g1, g2 = st.columns(2)
        g1.metric("First 2 PCs", f"{pca.explained_variance_ratio_.sum() * 100:.2f}%")
        g2.metric("PCs for 95 %", n95)

    st.subheader("Feature loadings")
    l1, l2 = st.columns([1, 1])
    n_load = l1.slider("Components shown in heatmap", 2, 6, 2)
    pc_sel = l2.selectbox("Rank features for", [f"PC{i}" for i in range(1, 7)])
    comps = pca_full.components_[:6]
    load_df = pd.DataFrame(comps[:n_load].T, index=X.columns, columns=[f"PC{i}" for i in range(1, n_load + 1)])
    h1, h2 = st.columns([1, 1])
    with h1:
        fig = px.imshow(load_df, color_continuous_scale="RdBu_r", zmin=-.6, zmax=.6, aspect="auto")
        fig.update_layout(height=700, title="Feature loadings heatmap", coloraxis_colorbar=dict(thickness=12))
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pca_load")
    with h2:
        ld = pd.Series(comps[int(pc_sel[2:]) - 1], index=X.columns)
        top = ld.reindex(ld.abs().sort_values(ascending=False).index)[:12][::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=["#38e0c4" if v > 0 else "#ff5d8f" for v in top.values]))
        fig.update_layout(height=700, title=f"Top 12 contributing features — {pc_sel}")
        fig.update_xaxes(title="Loading")
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pca_top")

    st.subheader("3-D view (auto-rotate)")
    Z3 = PCA(n_components=3, random_state=seed).fit_transform(Xin)
    ang = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    fig = go.Figure(data=[go.Scatter3d(x=Z3[y == i, 0], y=Z3[y == i, 1], z=Z3[y == i, 2], mode="markers", name=names[i],
                                       marker=dict(size=4, color=CLS[names[i]], opacity=.85, line=dict(width=.3, color="white")))
                          for i in range(2)],
                    frames=[go.Frame(name=str(i), layout=dict(scene_camera=dict(eye=dict(x=1.9 * np.cos(a), y=1.9 * np.sin(a), z=0.85))))
                            for i, a in enumerate(ang)])
    fig.update_layout(
        height=620, title="First three principal components", margin=dict(b=90),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", camera=dict(eye=dict(x=1.9, y=0, z=0.85)),
                   xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.1)"),
                   yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.1)"),
                   zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,.1)")),
        legend=dict(orientation="h", y=1.08, x=0.02),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=0, xanchor="left", yanchor="top", pad=dict(t=10),
                          showactive=False, **BTN_STYLE,
                          buttons=[dict(label="▶  Rotate", method="animate",
                                        args=[None, dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0),
                                                         fromcurrent=True, mode="immediate")]),
                                   dict(label="⏸  Stop", method="animate", args=[[None], STOP])])])
    st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="pca_3d")
    st.caption(f"Three PCs retain {PCA(n_components=3).fit(Xin).explained_variance_ratio_.sum() * 100:.2f}% of the variance.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ICA & LDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("ICA — independent components")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Iterations used", int(ica.n_iter_))
    i2.metric("max_iter", int(ica.max_iter))
    i3.metric("Converged", "Yes ✓" if ica.n_iter_ < ica.max_iter else "No ✗")
    i4.metric("Mixing / unmixing shape", f"{ica.mixing_.shape} / {ica.components_.shape}")

    n_show = min(n_ica, 4)
    fig = make_subplots(rows=1, cols=n_show, subplot_titles=[f"IC{jj + 1}" for jj in range(n_show)], horizontal_spacing=.06)
    for jj in range(n_show):
        for i, name in enumerate(names):
            fig.add_histogram(x=Zi[y == i, jj], nbinsx=35, marker_color=CLS[name], opacity=.7, name=name,
                              legendgroup=name, showlegend=(jj == 0), row=1, col=jj + 1)
    fig.update_layout(barmode="overlay", height=380, title="Component distributions by diagnosis" + (" (first 4 shown)" if n_ica > 4 else ""),
                      legend=dict(orientation="h", y=1.18, x=0.02))
    st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="ica_hist")

    e1, e2 = st.columns(2)
    with e1:
        ic_names = [f"IC{jj + 1}" for jj in range(n_ica)]
        corr_ic = np.corrcoef(Zi.T)
        fig = px.imshow(corr_ic, x=ic_names, y=ic_names, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f")
        fig.update_layout(height=420, title="Correlation between ICA components", coloraxis_colorbar=dict(thickness=12))
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="ica_corr")
    with e2:
        Zp_n = pca_full.transform(Xin)[:, :n_ica]
        fig = go.Figure()
        fig.add_bar(x=[f"PC{jj + 1}" for jj in range(n_ica)], y=kurtosis(Zp_n, axis=0), name="PCA components", marker_color="#8b7cff")
        fig.add_bar(x=ic_names, y=kurtosis(Zi, axis=0), name="ICA components", marker_color="#38e0c4")
        fig.update_layout(barmode="group", height=420, title="Excess kurtosis (0 = Gaussian)", legend=dict(orientation="h", y=1.12, x=0.02))
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="ica_kurt")
    st.caption("ICA searches for directions that are as non-Gaussian as possible — that is why its components typically show larger |kurtosis| than PCA's.")

    st.subheader("LDA — class separation")
    class_means = [Zl[y == i, 0].mean() for i in range(len(names))]
    d1, d2, d3 = st.columns(3)
    d1.metric(f"Mean LD1 — {names[0]}", f"{class_means[0]:.3f}")
    d2.metric(f"Mean LD1 — {names[1]}", f"{class_means[1]:.3f}")
    d3.metric("Distance between class means", f"{abs(class_means[0] - class_means[1]):.3f}")
    v1, v2 = st.columns(2)
    with v1:
        fig = go.Figure()
        for i, name in enumerate(names):
            fig.add_violin(x=Zl[y == i, 0], name=name, orientation="h", line_color=CLS[name], fillcolor=CLS[name], opacity=.55,
                           box_visible=True, meanline_visible=True, points="all", jitter=.45, pointpos=0, marker=dict(size=4))
        fig.update_layout(height=420, title="LD1 scores by diagnosis", showlegend=False)
        fig.update_xaxes(title="Linear Discriminant 1")
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="lda_violin")
    with v2:
        coef = pd.Series(lda.coef_[0], index=X.columns)
        top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:10][::-1]
        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                               marker_color=["#38e0c4" if v > 0 else "#ff5d8f" for v in top.values]))
        fig.update_layout(height=420, title="Top 10 LDA coefficients")
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="lda_coef")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    def sil(z):
        try:
            return silhouette_score(z, y)
        except Exception:
            return np.nan

    comp = pd.DataFrame({
        "Technique": ["PCA", "LDA", "ICA"],
        "Approach": ["Unsupervised", "Supervised", "Unsupervised"],
        "Uses class labels": ["No", "Yes", "No"],
        "Components": [Zp.shape[1], Zl.shape[1], Zi.shape[1]],
        "Reduced shape": [str(Zp.shape), str(Zl.shape), str(Zi.shape)],
        "Silhouette score": [sil(Zp), sil(Zl), sil(Zi)],
        "Execution time (ms)": [res["t_pca"] * 1000, res["t_lda"] * 1000, res["t_ica"] * 1000],
        "Primary objective": ["Maximum variance", "Maximum class separation", "Statistical independence"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True,
                 column_config={"Silhouette score": st.column_config.NumberColumn(format="%.4f"),
                                "Execution time (ms)": st.column_config.NumberColumn(format="%.2f")})
    TECH_COL = ["#8b7cff", "#ff5d8f", "#38e0c4"]
    s1, s2 = st.columns(2)
    with s1:
        fig = go.Figure(go.Bar(x=comp["Technique"], y=comp["Silhouette score"], marker_color=TECH_COL,
                               text=comp["Silhouette score"].round(3), textposition="outside"))
        fig.update_layout(height=400, title="Silhouette score (higher = tighter, better-separated classes)")
        fig.update_yaxes(range=[min(0, np.nanmin(comp["Silhouette score"]) - .1), np.nanmax(comp["Silhouette score"]) + .15])
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="cmp_sil")
    with s2:
        fig = go.Figure(go.Bar(x=comp["Technique"], y=comp["Execution time (ms)"], marker_color=TECH_COL,
                               text=comp["Execution time (ms)"].round(2), textposition="outside"))
        fig.update_layout(height=400, title="Execution time (ms)")
        fig.update_yaxes(range=[0, comp["Execution time (ms)"].max() * 1.25])
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CFG, key="cmp_time")

    best = comp.loc[comp["Silhouette score"].idxmax()]
    st.success(f"On this run the highest silhouette score is **{best['Technique']}** ({best['Silhouette score']:.3f}).")
    st.info("Silhouette score is a descriptive separation measure, not classifier accuracy. LDA explicitly uses labels; PCA and ICA do not, "
            "so overlap in their plots is expected and does not mean the algorithms failed.")

    st.write("LDA class means on LD1")
    st.dataframe(pd.DataFrame({"Class": names, "Mean LD1": class_means}), use_container_width=True, hide_index=True)
    st.write("Conceptual comparison")
    st.dataframe(pd.DataFrame([
        ["Objective", "Maximum variance", "Maximum class separation", "Statistical independence"],
        ["Uses labels", "No", "Yes", "No"],
        ["Main output", "Principal components", "Discriminant components", "Independent components"],
        ["Typical use", "Compression, noise reduction", "Classification, face recognition", "Signal processing, source separation"],
    ], columns=["Parameter", "PCA", "LDA", "ICA"]), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT & VIVA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown("""
- **PCA:** unsupervised; finds directions of maximum variance.
- **LDA:** supervised; uses class labels and maximises between-class separation relative to within-class variation.
- **ICA:** unsupervised; estimates statistically independent components.
- For two classes, LDA can produce at most one discriminant component.
""")
    st.subheader("Viva practice")
    viva = [
        ("What is the full form of PCA, LDA and ICA?", "Principal Component Analysis, Linear Discriminant Analysis, Independent Component Analysis."),
        ("Which of these methods is supervised?", "LDA — it needs the class labels. PCA and ICA are unsupervised."),
        ("What does explained variance ratio represent?", "The fraction of the total variance in the data captured by each principal component."),
        ("Why is LDA limited to one component for this binary dataset?", "LDA yields at most (number of classes − 1) components; two classes → one."),
        ("Does PCA use the target labels?", "No. It only looks at the feature covariance structure."),
        ("What is the purpose of StandardScaler?", "It rescales each feature to zero mean and unit variance so large-valued features do not dominate scale-sensitive methods."),
        ("What does ICA mean by statistical independence?", "The joint distribution of the components factorises into the product of their marginals — a stronger condition than being uncorrelated."),
        ("Why might FastICA fail to converge within the iteration limit?", "Tight tolerance, too few iterations, near-Gaussian sources, or a poor local optimum; inspect `n_iter_` rather than hiding the warning."),
        ("Is a silhouette score the same as classification accuracy?", "No. It measures cohesion vs separation of labelled groups in the projected space (−1 to 1); accuracy counts correct predictions."),
        ("Give one real-world application of each technique.", "PCA — compression/visualisation; LDA — face recognition/classification; ICA — audio source separation or EEG artefact removal."),
    ]
    for n_, (qq, aa) in enumerate(viva, 1):
        with st.expander(f"{n_}. {qq}"):
            st.write(aa)
    st.caption("Educational demonstration. The dataset is used here for dimensionality-reduction comparison, not clinical diagnosis.")