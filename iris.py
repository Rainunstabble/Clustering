
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("./IRIS.csv")

print(df)
print(df.head())
print(df.info())
print(df.describe())

print("Shape = ", df.shape)

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
model.fit(X_scaled)
df["Clusters"] = model.labels_

print("Cluster Result")
print(df[["Clusters"]].head(20))

score = silhouette_score(X_scaled, model.labels_)
print("Silhoutte Score = ", score)

plt.figure(figsize=(8,5))

plt.scatter(
    df["sepal_length"],
    df["sepal_width"],
    c=df["Clusters"],
    cmap="viridis"
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Flower Clustering using K-Means")
plt.colorbar(label="Cluster")
plt.show()

df.to_csv("Iris_Clustering_Results.csv", index=False)
print("Results saved!")