# ============================================================
# IMPORT LIBRARY
# ============================================================

import os
import joblib
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# membuat folder output 
os.makedirs("output", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("LOAD DATASET")
print("=" * 60)

df = pd.read_csv("online_retail_cleaned.csv")

print("\n5 Data Pertama")
print(df.head())

print("\nUkuran Dataset")
print(df.shape)

# ============================================================
# DATA UNDERSTANDING
# ============================================================

print("\nInformasi Dataset")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nStatistik Deskriptif")
print(df.describe())

# ============================================================
# DATA CLEANING
# ============================================================

print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

print(f"\nJumlah data sebelum cleaning : {df.shape}")

# Menghapus transaksi yang dibatalkan
df = df[df["is_cancelled"] == False]
print(f"Setelah menghapus transaksi yang dibatalkan : {df.shape}")

# Menghapus customer_id yang kosong
df = df.dropna(subset=["customer_id"])
print(f"Setelah menghapus Customer ID kosong       : {df.shape}")

# Mengubah tipe data
df["invoicedate"] = pd.to_datetime(df["invoicedate"])
print("\nTipe data InvoiceDate berhasil diubah menjadi datetime.")

print("\nJumlah missing value setelah cleaning:")
print(df.isnull().sum())

print("\nUkuran Dataset Setelah Cleaning")
print(df.shape)

print("\n5 Data Pertama Setelah Cleaning")
print(df.head())

# ============================================================
# PERHITUNGAN RFM
# ============================================================

print("=" * 60)
print("PERHITUNGAN RFM")
print("=" * 60)

# Menentukan tanggal acuan (1 hari setelah transaksi terakhir)
snapshot = df["invoicedate"].max() + pd.Timedelta(days=1)

# Menghitung nilai RFM setiap pelanggan
rfm = df.groupby("customer_id").agg({
    "invoicedate": lambda x: (snapshot - x.max()).days,
    "invoice": "nunique",
    "total_price": "sum"
})

# Mengubah nama kolom
rfm.columns = ["Recency", "Frequency", "Monetary"]

print("\n5 Data Pertama RFM")
print(rfm.head())

print("\nJumlah Customer")
print(rfm.shape)

print("\nStatistik RFM")
print(rfm.describe())

# ============================================================
# TRANSFORMASI LOGARITMIK
# ============================================================

print("=" * 60)
print("TRANSFORMASI LOGARITMIK")
print("=" * 60)

# Transformasi log
rfm_log = np.log1p(rfm)

print("\n5 Data Pertama Setelah Transformasi Log")
print(rfm_log.head())

print("\nStatistik Setelah Transformasi Log")
print(rfm_log.describe())

# ============================================================
# MIN-MAX SCALING
# ============================================================

print("=" * 60)
print("MIN-MAX SCALING")
print("=" * 60)

# Inisialisasi scaler
scaler = MinMaxScaler()

# Normalisasi data
rfm_scaled = scaler.fit_transform(rfm_log)

# Mengubah kembali menjadi DataFrame
rfm_scaled = pd.DataFrame(
    rfm_scaled,
    columns=rfm_log.columns,
    index=rfm_log.index
)

print("\n5 Data Pertama Setelah Scaling")
print(rfm_scaled.head())

print("\nStatistik Setelah Scaling")
print(rfm_scaled.describe())


joblib.dump(
    scaler,
    "output/scaler.pkl"
)

print("\nScaler berhasil disimpan.")


# ============================================================
# ELBOW METHOD
# ============================================================

print("=" * 60)
print("ELBOW METHOD")
print("=" * 60)

sse = []

for k in range(2, 11):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(rfm_scaled)
    sse.append(model.inertia_)

print("\nNilai SSE")
for i, nilai in enumerate(sse, start=2):
    print(f"Cluster {i} : {nilai:.2f}")

# VISUALISASI ELBOW
plt.figure(figsize=(8,5))

plt.plot(
    range(2,11),
    sse,
    marker='o'
)

plt.title("Elbow Method")
plt.xlabel("Jumlah Cluster (k)")
plt.ylabel("SSE")

plt.grid(True)

plt.savefig("images/elbow.png")

plt.show()

print("\nGrafik Elbow berhasil disimpan.")


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

print("=" * 60)
print("K-MEANS CLUSTERING")
print("=" * 60)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

print("\n5 Data Pertama")

print(rfm.head())

joblib.dump(
    kmeans,
    "output/kmeans.pkl"
)

print("\nModel KMeans berhasil disimpan.")

rfm.to_csv(
    "output/hasil_segmentasi_pelanggan.csv"
)

print("\nHasil segmentasi berhasil disimpan.")


# ============================================================
# EVALUATION
# ============================================================

print("=" * 60)
print("EVALUATION")
print("=" * 60)

# Davies Bouldin Index
dbi = davies_bouldin_score(
    rfm_scaled,
    rfm["Cluster"]
)

print(f"\nDavies Bouldin Index : {dbi:.4f}")

# Silhouette Score
sil = silhouette_score(
    rfm_scaled,
    rfm["Cluster"]
)

print(f"Silhouette Score : {sil:.4f}")

# ANALISI RATA RATA TIAP CLUSTER
cluster_analysis = rfm.groupby("Cluster").mean(numeric_only=True)

print("\nRata-rata Nilai RFM Setiap Cluster")

print(cluster_analysis)

cluster_analysis.to_csv(
    "output/cluster_analysis.csv"
)

print("\nCluster analysis berhasil disimpan.")

# ============================================================
# SCATTER PLOT CLUSTER
# ============================================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="viridis"
)

plt.title("Visualisasi Cluster Pelanggan")
plt.xlabel("Frequency")
plt.ylabel("Monetary")

plt.savefig(
    "images/scatter_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Scatter Plot berhasil disimpan.")

# ============================================================
# DISTRIBUSI CLUSTER
# ============================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=rfm,
    x="Cluster",
    palette="viridis"
)

plt.title("Distribusi Jumlah Customer")
plt.xlabel("Cluster")
plt.ylabel("Jumlah Customer")

plt.savefig(
    "images/distribusi_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Distribusi Cluster berhasil disimpan.")

# ============================================================
# BOXPLOT RFM
# ============================================================

fig, axes = plt.subplots(1,3,figsize=(18,5))

sns.boxplot(
    data=rfm,
    x="Cluster",
    y="Recency",
    ax=axes[0]
)

axes[0].set_title("Recency")

sns.boxplot(
    data=rfm,
    x="Cluster",
    y="Frequency",
    ax=axes[1]
)

axes[1].set_title("Frequency")

sns.boxplot(
    data=rfm,
    x="Cluster",
    y="Monetary",
    ax=axes[2]
)

axes[2].set_title("Monetary")

plt.suptitle("Distribusi Nilai RFM")

plt.tight_layout()

plt.savefig(
    "images/boxplot_rfm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Boxplot berhasil disimpan.")

# ============================================================
# HEATMAP KORELASI
# ============================================================

plt.figure(figsize=(6,4))

sns.heatmap(
    rfm[
        ["Recency","Frequency","Monetary"]
    ].corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Korelasi Antar Variabel RFM")

plt.savefig(
    "images/heatmap_korelasi.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Heatmap Korelasi berhasil disimpan.")

# ============================================================
# HEATMAP CLUSTER
# ============================================================

plt.figure(figsize=(8,4))

sns.heatmap(
    cluster_analysis,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)

plt.title("Rata-rata Nilai RFM Tiap Cluster")

plt.savefig(
    "images/heatmap_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Heatmap Cluster berhasil disimpan.")

# ============================================================
# SIMPAN HASIL
# ============================================================

rfm.to_csv(
    "output/hasil_segmentasi_pelanggan.csv"
)

print("\nHasil segmentasi berhasil disimpan.")
print("\nAnalisis selesai.")