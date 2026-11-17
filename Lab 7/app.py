import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, pairwise_distances
from sklearn.decomposition import PCA

st.set_page_config(page_title="Lab 7 | K-Means vs K-Medoids", page_icon="📊", layout="wide")
st.title("📊 Lab Exercise 7 — Wholesale Customer Clustering")
st.caption("Interactive comparison of K-Means and K-Medoids using the UCI Wholesale Customers dataset.")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded = st.sidebar.file_uploader("Upload Wholesale customers data.csv", type=["csv"])
if uploaded is None:
    st.info("Upload the supplied Wholesale customers data.csv file to begin.")
    st.stop()

df = load_data(uploaded)
features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Required columns missing: {missing}")
    st.stop()

X = df[features].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
scaler_choice = st.sidebar.selectbox("Scaling method", ["StandardScaler", "RobustScaler"])
scaler = StandardScaler() if scaler_choice == "StandardScaler" else RobustScaler()
Xs = scaler.fit_transform(X)

def pam_kmedoids(X, k, max_iter=100, random_state=42):
    D = pairwise_distances(X, metric="euclidean")
    n = len(X)
    medoids = [int(np.argmin(D.sum(axis=1)))]
    while len(medoids) < k:
        nearest = D[:, medoids].min(axis=1)
        best_candidate, best_cost = None, np.inf
        for candidate in range(n):
            if candidate in medoids:
                continue
            cost = np.minimum(nearest, D[:, candidate]).sum()
            if cost < best_cost:
                best_candidate, best_cost = candidate, cost
        medoids.append(best_candidate)
    medoids = np.array(medoids, dtype=int)
    current_cost = D[:, medoids].min(axis=1).sum()
    for _ in range(max_iter):
        nearest = D[:, medoids]
        best_delta, best_swap = 0.0, None
        for pos in range(k):
            others = [j for j in range(k) if j != pos]
            second = nearest[:, others].min(axis=1) if others else np.full(n, np.inf)
            costs = np.minimum(second[:, None], D).sum(axis=0)
            costs[medoids] = np.inf
            cand = int(np.argmin(costs))
            delta = costs[cand] - current_cost
            if delta < best_delta:
                best_delta, best_swap = delta, (pos, cand)
        if best_swap is None:
            break
        medoids[best_swap[0]] = best_swap[1]
        current_cost = D[:, medoids].min(axis=1).sum()
    labels = D[:, medoids].argmin(axis=1)
    return labels, medoids

def metrics(X, labels):
    return {
        "Silhouette (↑)": silhouette_score(X, labels),
        "Davies-Bouldin (↓)": davies_bouldin_score(X, labels),
        "Calinski-Harabasz (↑)": calinski_harabasz_score(X, labels)
    }

st.sidebar.header("Experiment controls")
k = st.sidebar.slider("Number of clusters (K)", 2, 8, 3)
show_kmeans = st.sidebar.checkbox("Show K-Means", True)
show_kmedoids = st.sidebar.checkbox("Show K-Medoids", True)

c1, c2, c3 = st.columns(3)
c1.metric("Customer records", f"{len(df):,}")
c2.metric("Clustering features", len(features))
c3.metric("Missing feature values", int(df[features].isna().sum().sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Dataset & EDA", "Cluster visualization", "Evaluation", "Cluster profiles"])

with tab1:
    st.subheader("Dataset preview")
    st.dataframe(df.head(12), use_container_width=True)
    st.subheader("Spending distributions")
    long_df = X.melt(var_name="Spending category", value_name="Annual spending")
    st.plotly_chart(px.box(long_df, x="Spending category", y="Annual spending",
                           color="Spending category", points="outliers",
                           title="Spending distribution and potential outliers"),
                    use_container_width=True)
    st.plotly_chart(px.imshow(X.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Feature correlation heatmap"),
                    use_container_width=True)

results, labels_map, medoids_map = {}, {}, {}
if show_kmeans:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels_map["K-Means"] = km.fit_predict(Xs)
    results["K-Means"] = metrics(Xs, labels_map["K-Means"])
if show_kmedoids:
    med_labels, medoid_ids = pam_kmedoids(Xs, k)
    labels_map["K-Medoids"] = med_labels
    medoids_map["K-Medoids"] = medoid_ids
    results["K-Medoids"] = metrics(Xs, med_labels)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(Xs)
plot_frames = []
for name, labs in labels_map.items():
    frame = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                          "Cluster": labs.astype(str), "Algorithm": name})
    plot_frames.append(frame)
plot_df = pd.concat(plot_frames, ignore_index=True) if plot_frames else pd.DataFrame()

with tab2:
    st.subheader(f"2D PCA cluster projection (K={k})")
    st.caption(f"First two PCs explain {pca.explained_variance_ratio_.sum():.1%} of standardized feature variance.")
    if not plot_df.empty:
        st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                                   facet_col="Algorithm", opacity=.75,
                                   title="Customer clusters in PCA space"),
                        use_container_width=True)
    st.subheader("3D PCA projection")
    pca3 = PCA(n_components=3, random_state=42)
    xyz = pca3.fit_transform(Xs)
    if labels_map:
        name = st.selectbox("Algorithm for 3D view", list(labels_map))
        f3 = pd.DataFrame({"PC1": xyz[:, 0], "PC2": xyz[:, 1], "PC3": xyz[:, 2],
                           "Cluster": labels_map[name].astype(str)})
        fig3 = px.scatter_3d(f3, x="PC1", y="PC2", z="PC3", color="Cluster",
                             opacity=.75, title=f"{name}: 3D PCA view")
        if name in medoids_map:
            pts = xyz[medoids_map[name]]
            fig3.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                                       mode="markers", marker=dict(size=8, symbol="diamond", color="black"),
                                       name="Medoids"))
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Evaluation metrics")
    if results:
        metric_df = pd.DataFrame(results).T
        st.dataframe(metric_df.style.format("{:.4f}"), use_container_width=True)
        st.plotly_chart(px.bar(metric_df.reset_index().melt(id_vars="index"),
                               x="variable", y="value", color="index", barmode="group",
                               title="Algorithm metric comparison"),
                        use_container_width=True)
    else:
        st.warning("Select at least one algorithm in the sidebar.")

with tab4:
    st.subheader("Average spending profile by cluster")
    for name, labs in labels_map.items():
        profile = X.assign(Cluster=labs).groupby("Cluster")[features].mean()
        st.markdown(f"**{name}**")
        st.dataframe(profile.round(1), use_container_width=True)
        st.plotly_chart(px.imshow(np.log1p(profile), x=features,
                                  y=[f"Cluster {i}" for i in profile.index],
                                  color_continuous_scale="YlGnBu", aspect="auto",
                                  title=f"{name}: log-scaled average spending profile"),
                        use_container_width=True)
        if name == "K-Means":
            centers = scaler.inverse_transform(km.cluster_centers_)
            st.caption("K-Means centroids (original spending units)")
            st.dataframe(pd.DataFrame(centers, columns=features).round(1), use_container_width=True)
        elif name == "K-Medoids":
            st.caption("K-Medoids are actual customer records (original spending units)")
            st.dataframe(X.iloc[medoids_map[name]].reset_index(drop=True), use_container_width=True)

st.divider()
st.caption("Interpretation reminder: higher Silhouette and Calinski–Harabasz are generally preferred; lower Davies–Bouldin is preferred. Interpret metrics together with cluster sizes, visual separation and business meaning.")
