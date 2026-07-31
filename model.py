import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

sns.set_style("whitegrid")

# ============================================================
# LOAD DATA
# ============================================================

print("="*60)
print("LOAD HASIL CLUSTERING")
print("="*60)

rfm = pd.read_csv(
    "output/hasil_segmentasi_pelanggan.csv",
    index_col="customer_id"
)

print("\n5 Data Pertama")
print(rfm.head())

print("\nUkuran Dataset")
print(rfm.shape)

# ============================================================
# MENENTUKAN FITUR DAN TARGET
# ============================================================

print("="*60)
print("MENENTUKAN FITUR DAN TARGET")
print("="*60)

X = rfm[["Recency","Frequency","Monetary"]]
y = rfm["Cluster"]

print("\nFitur")
print(X.head())

print("\nTarget")
print(y.head())

# ============================================================
# SPLIT DATA
# ============================================================

print("="*60)
print("SPLIT DATA")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nJumlah Data Training :", X_train.shape)
print("Jumlah Data Testing  :", X_test.shape)

# ============================================================
# MEMBANGUN MODEL RANDOM FOREST
# ============================================================

print("="*60)
print("MEMBANGUN MODEL RANDOM FOREST")
print("="*60)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Random Forest berhasil dibuat.")

# ============================================================
# PREDIKSI DATA TESTING
# ============================================================

print("="*60)
print("PREDIKSI DATA TESTING")
print("="*60)

pred = model.predict(X_test)

print("\n20 Hasil Prediksi Pertama")

print(pred[:20])

# ============================================================
# EVALUASI MODEL
# ============================================================

print("="*60)
print("EVALUASI MODEL")
print("="*60)

accuracy = accuracy_score(
    y_test,
    pred
)

print(f"\nAccuracy : {accuracy:.4f}")

print("\nClassification Report")

print(classification_report(
    y_test,
    pred
))


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("="*60)
print("CONFUSION MATRIX")
print("="*60)

cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.savefig(
    "images/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion Matrix berhasil disimpan.")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print("\nFeature Importance")

print(importance)

# ============================================================
# VISUALISASI FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(7,4))

sns.barplot(
    x=importance.values,
    y=importance.index,
    hue=importance.index,
    palette="viridis",
    legend=False
)

plt.title("Feature Importance Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Fitur")

plt.savefig(
    "images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFeature Importance berhasil disimpan.")

# ============================================================
# SIMPAN MODEL
# ============================================================

joblib.dump(
    model,
    "output/random_forest.pkl"
)

print("\nModel Random Forest berhasil disimpan.")


# ============================================================
# PREDIKSI CUSTOMER BARU
# ============================================================

print("=" * 60)
print("PREDIKSI CUSTOMER BARU")
print("=" * 60)

# Contoh data customer baru
customer_baru = pd.DataFrame({
    "Recency": [326],
    "Frequency": [12],
    "Monetary": [77556.46]
})

hasil = model.predict(customer_baru)[0]

print(f"\nCluster Hasil Prediksi : {hasil}")

# ============================================================
# MENENTUKAN KATEGORI CLUSTER
# ============================================================

cluster_analysis = pd.read_csv(
    "output/cluster_analysis.csv",
    index_col="Cluster"
)

cluster_mean = cluster_analysis.sort_values(
    by=["Monetary", "Frequency"],
    ascending=False
)

mapping = {}

for i, cluster in enumerate(cluster_mean.index):

    if i == 0:
        mapping[cluster] = "Pelanggan Loyal"

    elif i == 1:
        mapping[cluster] = "Pelanggan Potensial"

    else:
        mapping[cluster] = "Pelanggan Berisiko"

print("\nKategori Cluster")

for key, value in mapping.items():
    print(f"Cluster {key} : {value}")

# ============================================================
# REKOMENDASI STRATEGI
# ============================================================

def rekomendasi(cluster):

    kategori = mapping[cluster]

    if kategori == "Pelanggan Loyal":
        return "Berikan reward, cashback, membership, dan promo eksklusif."

    elif kategori == "Pelanggan Potensial":
        return "Berikan promo personal, cross-selling, dan diskon agar menjadi pelanggan loyal."

    else:
        return "Berikan voucher reaktivasi, email marketing, dan promo khusus untuk menarik pelanggan kembali."

# ============================================================
# HASIL PREDIKSI
# ============================================================

print("=" * 60)
print("HASIL PREDIKSI")
print("=" * 60)

print(f"Cluster      : {hasil}")
print(f"Kategori     : {mapping[hasil]}")
print(f"Rekomendasi  : {rekomendasi(hasil)}")


# ============================================================
# MENAMBAHKAN KATEGORI DAN REKOMENDASI
# ============================================================

rfm["Kategori"] = rfm["Cluster"].map(mapping)
rfm["Rekomendasi"] = rfm["Cluster"].apply(rekomendasi)

rfm.to_csv(
    "output/hasil_segmentasi_pelanggan.csv"
)

print("\nFile hasil_segmentasi_pelanggan.csv berhasil diperbarui.")
