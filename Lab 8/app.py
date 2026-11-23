import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

st.set_page_config(
    page_title="Hierarchical Clustering Explorer",
    page_icon="🌿",
    layout="wide",
)

# Use a dark Plotly theme so charts match Streamlit's dark interface.
st.title("Hierarchical Clustering Explorer")
st.caption("Explore Agglomerative and Divisive-style clustering on the Wine dataset.")

wine = load_wine(as_frame=True)
df = wine.data
X = StandardScaler().fit_transform(df)

with st.sidebar:
    st.header("Clustering settings")
    method = st.selectbox(
        "Choose clustering approach",
        ["Agglomerative", "Divisive (Bisecting K-Means)"],
    )
    k = st.slider("Number of clusters", min_value=2, max_value=6, value=3)

if method == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
else:
    model = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)

labels = model.fit_predict(X)
score = silhouette_score(X, labels)
sizes = pd.Series(labels, name="Cluster").value_counts().sort_index()
sil_values = silhouette_samples(X, labels)

pca = PCA(n_components=2, random_state=42)
points = pca.fit_transform(X)
plot_df = pd.DataFrame({
    "PC1": points[:, 0],
    "PC2": points[:, 1],
    "Cluster": labels.astype(str),
    "Wine class": wine.target.astype(str),
    "Silhouette": sil_values,
})

m1, m2, m3 = st.columns(3)
m1.metric("Silhouette score", f"{score:.3f}")
m2.metric("Clusters", k)
m3.metric("Samples", len(labels))

st.subheader("Cluster sizes")
size_df = sizes.rename("Count").reset_index()
size_df.columns = ["Cluster", "Count"]
fig_sizes = px.bar(
    size_df, x="Cluster", y="Count", color="Cluster",
    text="Count", template="plotly_dark",
    title="Number of wine samples in each cluster",
)
fig_sizes.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
fig_sizes.update_traces(textposition="outside")
st.plotly_chart(fig_sizes, use_container_width=True)

st.subheader("PCA cluster visualization")
fig_pca = px.scatter(
    plot_df, x="PC1", y="PC2", color="Cluster",
    hover_data=["Wine class", "Silhouette"],
    template="plotly_dark",
    title="Wine samples projected onto the first two principal components",
)
fig_pca.update_traces(marker=dict(size=9, opacity=0.85))
fig_pca.update_layout(height=560, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig_pca, use_container_width=True)

st.subheader("Silhouette values by cluster")
fig_sil = px.box(
    plot_df, x="Cluster", y="Silhouette", color="Cluster",
    points="outliers", template="plotly_dark",
    title="Within-cluster cohesion and separation",
)
fig_sil.add_hline(y=0, line_dash="dash", line_color="#f87171")
fig_sil.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig_sil, use_container_width=True)

with st.expander("View cluster-size table"):
    st.dataframe(size_df.set_index("Cluster"), use_container_width=True)

st.caption(
    "Note: PCA is used only to display the clusters in 2D. "
    "Clustering and silhouette score are calculated using all standardized features."
)
