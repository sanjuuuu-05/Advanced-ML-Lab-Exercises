
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
import tensorflow as tf

st.set_page_config(page_title="NeuroLab | MLP Studio", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(circle at 10% 0%, #e9f3ff 0, transparent 32%), linear-gradient(135deg,#f7f9ff,#f2f5fb);}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;color:#15233b;}
.hero {padding:2rem 2.2rem;border-radius:24px;background:linear-gradient(120deg,#111d3b,#253d71 60%,#4969a5);color:white;box-shadow:0 18px 55px #1e376022;margin-bottom:1.2rem;animation:rise .7s ease-out;}
.hero h1 {color:white;font-size:2.5rem;margin:0;}
.hero p {color:#dce8ff;font-size:1rem;margin:.5rem 0 0;}
.kpi {background:white;border:1px solid #e4eaf4;border-radius:18px;padding:1rem 1.2rem;box-shadow:0 8px 24px #23395b0b;transition:transform .2s,box-shadow .2s;}
.kpi:hover {transform:translateY(-4px);box-shadow:0 14px 30px #23395b18;}
.kpi small {color:#66758e;font-weight:600;text-transform:uppercase;letter-spacing:.08em;}
.kpi strong {display:block;font-size:1.8rem;color:#172b4d;margin-top:.25rem;}
.panel {background:white;border:1px solid #e4eaf4;border-radius:20px;padding:1.2rem 1.3rem;margin:.5rem 0 1rem;box-shadow:0 8px 25px #23395b09;}
div[data-testid="stTabs"] button {font-weight:700;}
@keyframes rise {from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    d = load_breast_cancer(as_frame=True)
    return d.data, d.target, list(d.feature_names)

@st.cache_resource
def train_model(activation, hidden1, hidden2, epochs, test_size):
    X, y, _ = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    Xtr = prep.fit_transform(Xtr).astype("float32")
    Xte = prep.transform(Xte).astype("float32")
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1],)),
        tf.keras.layers.Dense(hidden1, activation=activation, name="hidden_1"),
        tf.keras.layers.Dense(hidden2, activation=activation, name="hidden_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_split=.2, epochs=epochs, batch_size=32,
                     callbacks=[es], verbose=0)
    prob = model.predict(Xte, verbose=0).ravel()
    pred = (prob >= .5).astype(int)
    return model, prep, Xte, yte.to_numpy(), prob, pred, pd.DataFrame(hist.history), prep

X, y, features = load_data()
st.markdown("""<div class="hero"><h1>🧠 NeuroLab — MLP Studio</h1>
<p>Interactive neural-network lab · preprocessing · architecture explorer · model evaluation</p></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Experiment controls")
    activation = st.selectbox("Hidden-layer activation", ["relu", "tanh", "elu"])
    h1 = st.slider("Hidden layer 1 neurons", 8, 128, 64, 8)
    h2 = st.slider("Hidden layer 2 neurons", 8, 128, 32, 8)
    epochs = st.slider("Maximum epochs", 30, 250, 150, 10)
    test_size = st.slider("Test split", .15, .35, .20, .05)
    train = st.button("🚀 Train / refresh model", use_container_width=True, type="primary")

if "trained" not in st.session_state or train:
    with st.spinner("Training the MLP and preparing evaluation…"):
        st.session_state.trained = train_model(activation, h1, h2, epochs, test_size)
        st.session_state.config = (activation, h1, h2, epochs, test_size)

model, prep, Xte, yte, prob, pred, hist, _ = st.session_state.trained
metrics = {
    "Accuracy": accuracy_score(yte, pred),
    "Precision": precision_score(yte, pred, zero_division=0),
    "Recall": recall_score(yte, pred, zero_division=0),
    "F1 score": f1_score(yte, pred, zero_division=0),
    "ROC-AUC": roc_auc_score(yte, prob)
}
st.caption("Dataset: Wisconsin Diagnostic Breast Cancer · 569 observations · 30 numeric features · binary target")
tabs = st.tabs(["Overview", "Preprocessing", "Architecture & parameters", "Training", "Evaluation", "Explore data", "Self-learning"])

with tabs[0]:
    st.markdown("### Model performance at a glance")
    cols = st.columns(5)
    for c, (k,v) in zip(cols, metrics.items()):
        c.markdown(f'<div class="kpi"><small>{k}</small><strong>{v:.3f}</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">The model predicts the dataset diagnosis label. This benchmark exercise is educational and is not a clinical diagnostic system.</div>', unsafe_allow_html=True)
    st.plotly_chart(px.bar(pd.DataFrame({"Metric":list(metrics), "Score":list(metrics.values())}),
                           x="Metric", y="Score", color="Metric", range_y=[0,1],
                           title="Held-out test metrics", template="plotly_white"), use_container_width=True)

with tabs[1]:
    st.markdown("### Preprocessing workflow")
    steps = pd.DataFrame({"Step":["Raw dataset","Stratified split","Median imputation","Standard scaling","MLP-ready arrays"],
                          "Purpose":["30 numeric features","Keep class proportions","Robust missing-value handling","Comparable feature scales","Float32 tensors"]})
    st.dataframe(steps, hide_index=True, use_container_width=True)
    st.info("The selected dataset contains no missing values and no categorical predictors. The pipeline still includes median imputation; categorical encoding is not applicable.")
    st.write("Training and testing are split before fitting the preprocessing pipeline to prevent leakage.")
    st.metric("Input features", len(features))

with tabs[2]:
    st.markdown("### Live architecture")
    layers = [("Input", len(features)), ("Hidden 1", h1), ("Hidden 2", h2), ("Output", 1)]
    fig = go.Figure()
    for i,(name,n) in enumerate(layers):
        fig.add_trace(go.Scatter(x=[i],y=[0],mode="markers+text",text=[f"{name}<br>{n} neurons"],
                                 textposition="bottom center",marker=dict(size=42+min(n,80),opacity=.85),
                                 showlegend=False))
        if i:
            fig.add_annotation(x=i,y=0,ax=i-1,ay=0,xref="x",yref="y",axref="x",ayref="y",
                               text="",showarrow=True,arrowhead=3)
    fig.update_layout(height=300,template="plotly_white",xaxis_visible=False,yaxis_visible=False,
                      title=f"Input → {activation.upper()} hidden layers → sigmoid output")
    st.plotly_chart(fig, use_container_width=True)
    rows=[]
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            ni, no = int(layer.kernel.shape[0]), layer.units
            rows.append({"Layer":layer.name,"Inputs":ni,"Neurons":no,
                         "Weights":ni*no,"Biases":no,"Formula":f"({ni} × {no}) + {no}",
                         "Parameters":ni*no+no})
    ptable=pd.DataFrame(rows)
    st.dataframe(ptable, hide_index=True, use_container_width=True)
    a,b,c=st.columns(3)
    a.metric("Total parameters", f"{model.count_params():,}")
    b.metric("Trainable parameters", f"{sum(int(np.prod(w.shape)) for w in model.trainable_weights):,}")
    c.metric("Non-trainable parameters", f"{sum(int(np.prod(w.shape)) for w in model.non_trainable_weights):,}")
    st.code("Dense parameters = (input connections × neurons) + biases", language="text")

with tabs[3]:
    st.markdown("### Learning dynamics")
    epochs_done=len(hist)
    st.metric("Epochs completed", epochs_done)
    fig=go.Figure()
    for col in ["accuracy","val_accuracy"]:
        fig.add_trace(go.Scatter(y=hist[col],mode="lines",name=col.replace("_"," ").title()))
    fig.update_layout(title="Training and validation accuracy",xaxis_title="Epoch",template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)
    fig=go.Figure()
    for col in ["loss","val_loss"]:
        fig.add_trace(go.Scatter(y=hist[col],mode="lines",name=col.replace("_"," ").title()))
    fig.update_layout(title="Training and validation loss",xaxis_title="Epoch",template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)
    st.caption("A growing gap between training and validation curves can indicate overfitting. Similar low scores may indicate underfitting.")

with tabs[4]:
    st.markdown("### Confusion matrix and ROC")
    cm=confusion_matrix(yte,pred)
    fig=px.imshow(cm,text_auto=True,x=["Predicted malignant","Predicted benign"],
                  y=["Actual malignant","Actual benign"],color_continuous_scale="Blues",
                  title="Confusion matrix",template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)
    fpr,tpr,_=roc_curve(yte,prob)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"MLP AUC {metrics['ROC-AUC']:.3f}"))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random baseline",line_dash="dash"))
    fig.update_layout(title="ROC curve",xaxis_title="False positive rate",yaxis_title="True positive rate",
                      template="plotly_white")
    st.plotly_chart(fig,use_container_width=True)

with tabs[5]:
    st.markdown("### Dataset explorer")
    sample = X.copy()
    sample["diagnosis"] = y.map({0:"malignant",1:"benign"})
    st.dataframe(sample.head(20), use_container_width=True)
    selected=st.selectbox("Choose a feature",features,index=0)
    st.plotly_chart(px.histogram(sample,x=selected,color="diagnosis",marginal="box",
                                 barmode="overlay",opacity=.72,title=f"Distribution of {selected}",
                                 template="plotly_white"),use_container_width=True)
    corr=X.corr()
    st.plotly_chart(px.imshow(corr,color_continuous_scale="RdBu_r",zmin=-1,zmax=1,
                              title="Feature correlation heatmap",aspect="auto",
                              template="plotly_white"),use_container_width=True)

with tabs[6]:
    st.markdown("### Self-learning: questions and answers")
    qa = [
    ("What is an MLP?","A feed-forward neural network with an input layer, one or more fully connected hidden layers, and an output layer."),
    ("Why scale features?","To place numeric inputs on comparable scales and support stable, efficient optimization."),
    ("Why ReLU in hidden layers?","It introduces non-linearity and is computationally inexpensive."),
    ("Why sigmoid at the output?","It maps a single output to a probability between 0 and 1 for binary classification."),
    ("Why binary cross-entropy?","It is a standard loss for binary labels and predicted probabilities."),
    ("What is data leakage?","Using information from held-out data during training or preprocessing; fit transformations only on training data."),
    ("How are Dense parameters calculated?","(Number of inputs × number of neurons) + number of biases."),
    ("What does precision measure?","The fraction of predicted positive examples that are actually positive."),
    ("What does recall measure?","The fraction of actual positive examples correctly identified."),
    ("What indicates overfitting?","Training performance improves while validation performance stalls or worsens, creating a widening gap."),
    ("How can overfitting be reduced?","Early stopping, dropout, regularization, simpler models, or more representative training data."),
    ("Why compare architectures?","To understand how capacity and activation affect parameter count, optimization, and generalization."),
    ("Is this a medical diagnostic tool?","No. It is an educational benchmark experiment, not clinically validated or suitable for patient care.")
    ]
    for q,a in qa:
        with st.expander(q):
            st.write(a)

st.markdown("---")
st.caption("NeuroLab · MAI511-2 Advanced Machine Learning · Lab Exercise 6")
