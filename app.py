import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# ======================================
# KONFIGURASI HALAMAN
# ======================================

st.set_page_config(
    page_title="Dashboard Segmentasi Pelanggan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================
# LOAD CSS
# ======================================

with open("styles/style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

# ======================================
# LOAD DATA
# ======================================

rfm = pd.read_csv("output/hasil_segmentasi_pelanggan.csv")
cluster = pd.read_csv("output/cluster_analysis.csv")

model = joblib.load("output/random_forest.pkl")

# ======================================
# MAPPING KATEGORI CLUSTER
# ======================================

cluster_mean = cluster.sort_values(
    by=["Monetary", "Frequency"],
    ascending=False
)

mapping = {}

for i, c in enumerate(cluster_mean["Cluster"]):

    if i == 0:
        mapping[c] = "Pelanggan Loyal"

    elif i == 1:
        mapping[c] = "Pelanggan Potensial"

    else:
        mapping[c] = "Pelanggan Berisiko"

# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.markdown("""
    <h2 style="
        font-weight:700;
        font-size:30px;
        margin-bottom:20px;
    ">
     Dashboard
    </h2>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title="",
        options=[
            "Beranda",
            "Model",
            "Prediksi Customer",
            "Data Segmentasi"
        ],
        icons=[
            "house-fill",
            "diagram-3-fill",
            "robot",
            "table"
        ],
        default_index=0
    )

# ======================================
# BERANDA
# ======================================

if selected == "Beranda":

    st.title(" Dashboard Segmentasi Pelanggan")
    st.write("")

    # =============================
    # CARD
    # =============================

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="title-card">TOTAL CUSTOMER</div>
            <div class="value-card">{len(rfm):,}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="title-card">JUMLAH CLUSTER</div>
            <div class="value-card">{rfm['Cluster'].nunique()}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
            <div class="title-card">DBI</div>
            <div class="value-card">0.96</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="card">
            <div class="title-card">ACCURACY RF</div>
            <div class="value-card">99.07%</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ======================================
    # HASIL CLUSTERING & ANALISIS
    # ======================================

    kiri, kanan = st.columns([0.8, 1.2], gap="large")

    # ==========================
    # KOLOM KIRI
    # ==========================
    with kiri:

        st.subheader("Hasil Clustering")

        img1, img2 = st.columns(2)

        with img1:
            st.image(
                "images/elbow.png",
                use_container_width=True
            )

        with img2:
            st.image(
                "images/scatter_cluster.png",
                use_container_width=True
            )

    # ==========================
    # KOLOM KANAN
    # ==========================
    with kanan:

        st.subheader("Analisis Cluster")

        st.dataframe(
            cluster,
            hide_index=True,
            height=150,
            use_container_width=True
        )

        img1, img2 = st.columns(2)

        with img1:
            st.image(
                "images/heatmap_cluster.png",
                use_container_width=True
            )

        with img2:
            st.image(
                "images/distribusi_cluster.png",
                use_container_width=True
            )

# ======================================
# MODEL
# ======================================

elif selected == "Model":

    st.title(" Model")

    # ======================================
    # K-MEANS
    # ======================================

    st.subheader(" K-Means Clustering")

    km1,km2 = st.columns(2)

    with km1:
        st.image(
            "images/elbow.png",
             width=500
            
        )

    with km2:
        st.image(
            "images/scatter_cluster.png",
            width=500
        )

    st.divider()

    # ======================================
    # RANDOM FOREST
    # ======================================

    st.subheader(" Random Forest")

    rf1,rf2 = st.columns(2)

    with rf1:
        st.image(
            "images/confusion_matrix.png",
             width=500
        )

    with rf2:
        st.image(
            "images/feature_importance.png",
             width=500,
        )

# ======================================
# PREDIKSI CUSTOMER
# ======================================

elif selected == "Prediksi Customer":
    st.title(" Prediksi Customer")

    input_col, hasil_col = st.columns([1, 1.2])

    # ==========================
    # INPUT
    # ==========================

    with input_col:

        st.subheader("Input Data RFM")

        recency = st.number_input(
            "Recency",
            min_value=0,
            value=30
        )

        frequency = st.number_input(
            "Frequency",
            min_value=1,
            value=5
        )

        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            value=1000.0
        )

        prediksi = st.button(
            " Prediksi Customer",
            use_container_width=True
        )

    # ==========================
    # HASIL
    # ==========================

    with hasil_col:

        st.subheader("Hasil Prediksi")

        if prediksi:

            customer = pd.DataFrame({
                "Recency": [recency],
                "Frequency": [frequency],
                "Monetary": [monetary]
            })

            hasil = model.predict(customer)[0]
            kategori = mapping[hasil]

    # Tampilkan hasil
            st.metric("Cluster", hasil)
            st.metric("Kategori", kategori)

            if kategori == "Pelanggan Loyal":

                st.markdown("### Rekomendasi")
                st.markdown("""
                - Reward pelanggan
                - Cashback atau voucher
                - Membership
                - Promo eksklusif
                """)

            elif kategori == "Pelanggan Potensial":
                st.markdown("### Rekomendasi")
                st.markdown("""
                - Promo personal
                - Cross Selling
                - Diskon
                """)
                
            else:
                st.markdown("### Rekomendasi")
                st.markdown("""
                - Voucher reaktivasi
                - Email marketing
                - Promo khusus
                """)

# ======================================
# DATA SEGMENTASI
# ======================================

elif selected == "Data Segmentasi":
    st.title(" Data Segmentasi Pelanggan")

    st.divider()

    # ==========================
    # FILTER
    # ==========================

    col1, col2 = st.columns([3,1])

    with col1:
        keyword = st.text_input(
            " Cari Customer ID",
            placeholder="Masukkan Customer ID..."
        )

    with col2:
        pilih_cluster = st.selectbox(
            "Filter Cluster",
            ["Semua"] + sorted(rfm["Cluster"].unique().tolist())
        )

    data = rfm.copy()

    # Filter Customer ID
    if keyword:
        data = data[
            data.index.astype(str).str.contains(
                keyword,
                case=False
            )
        ]

    # Filter Cluster
    if pilih_cluster != "Semua":
        data = data[
            data["Cluster"] == pilih_cluster
        ]

    st.write("")

    # ==========================
    # RINGKASAN
    # ==========================

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Jumlah Customer",
            f"{len(data):,}"
        )

    with c2:
        st.metric(
            "Jumlah Cluster",
            data["Cluster"].nunique()
        )

    st.divider()

    # ==========================
    # DATAFRAME
    # ==========================

    st.dataframe(
        data,
        use_container_width=True,
        height=450,
        hide_index=True
    )

    st.write("")

    st.download_button(
        " Download CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="hasil_segmentasi_pelanggan.csv",
        mime="text/csv",
        use_container_width=True
    )