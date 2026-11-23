
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.title("Hierarchical Clustering Explorer")
st.write("Explore Agglomerative and Divisive-style clustering on the Wine dataset.")

wine = load_wine()
X = StandardScaler().fit_transform(wine.data)

method = st.selectbox("Choose clustering approach",
                      ["Agglomerative", "Divisive (Bisecting K-Means)"])
k = st.slider("Number of clusters", 2, 6, 3)

if method == "Agglomerative":
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
else:
    model = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)

labels = model.fit_predict(X)
points = PCA(n_components=2).fit_transform(X)

st.metric("Silhouette score", round(silhouette_score(X, labels), 3))
st.write("Cluster sizes")
st.write(pd.Series(labels).value_counts().sort_index())

fig, ax = plt.subplots()
ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="viridis")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title(method + " clustering")
st.pyplot(fig)
