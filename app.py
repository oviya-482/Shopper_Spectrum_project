import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

APP_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(APP_DIR / "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(APP_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(APP_DIR / "cluster_label_map.pkl", "rb") as f:
        cluster_label_map = pickle.load(f)
    similarity_df = pd.read_pickle(APP_DIR / "product_similarity.pkl")
    with open(APP_DIR / "product_list.pkl", "rb") as f:
        product_list = pickle.load(f)
    return kmeans, scaler, cluster_label_map, similarity_df, product_list


try:
    kmeans, scaler, cluster_label_map, similarity_df, product_list = load_artifacts()
    artifacts_ok = True
except FileNotFoundError as e:
    artifacts_ok = False
    missing_error = e

SEGMENT_INFO = {
    "High-Value": ("🌟", "Recent, frequent, big spenders — your best customers. Prioritize loyalty & VIP perks."),
    "Regular": ("🙂", "Steady, moderate purchasers. Good upsell / cross-sell targets."),
    "Occasional": ("🕒", "Infrequent, lower spend. Nurture with targeted promotions."),
    "At-Risk": ("⚠️", "Haven't purchased in a long time. Target with win-back / retention campaigns."),
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendations for E-Commerce")

if not artifacts_ok:
    st.error(
        "Model artifacts not found next to app.py. Please run the analysis notebook "
        "first so the following files are generated in this folder:\n\n"
        "`kmeans_model.pkl`, `scaler.pkl`, `cluster_label_map.pkl`, "
        "`product_similarity.pkl`, `product_list.pkl`"
    )
    st.stop()

tab1, tab2 = st.tabs(["🔎 Product Recommendations", "👥 Customer Segmentation"])

# ---------------------------------------------------------------------------
# Tab 1: Product Recommendation Module
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Find similar products")
    st.write("Enter a product name to get 5 similar products, based on collaborative filtering "
             "over customer purchase history.")

    col1, col2 = st.columns([3, 1])
    with col1:
        product_name = st.selectbox(
            "Product Name",
            options=product_list,
            index=None,
            placeholder="Start typing or select a product...",
        )
    with col2:
        st.write("")
        st.write("")
        get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)

    if get_recs:
        if not product_name:
            st.warning("Please select a product name first.")
        else:
            sims = (
                similarity_df[product_name]
                .drop(product_name)
                .sort_values(ascending=False)
                .head(5)
            )
            st.success(f"Top 5 products similar to **{product_name}**:")
            cols = st.columns(5)
            for c, (prod, score) in zip(cols, sims.items()):
                with c:
                    st.markdown(
                        f"""
                        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:14px;
                                    min-height:170px; background-color:#fafafa;">
                            <div style="font-size:14px; font-weight:600; margin-bottom:8px; color:#111111;">
                                {prod.title()}
                            </div>
                            <div style="font-size:12px; color:#666666;">
                                Similarity score
                            </div>
                            <div style="font-size:18px; font-weight:700; color:#2e7d32;">
                                {score:.2f}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Tab 2: Customer Segmentation Module
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Predict a customer's segment")
    st.write("Enter a customer's RFM values to predict which behavioral segment they belong to.")

    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30, step=1)
    with c2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5, step=1)
    with c3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, step=10.0, format="%.2f")

    predict = st.button("Predict Cluster", type="primary")

    if predict:
        X = np.array([[recency, frequency, monetary]])
        X_scaled = scaler.transform(X)
        cluster = kmeans.predict(X_scaled)[0]
        segment = cluster_label_map.get(cluster, f"Cluster {cluster}")
        emoji, desc = SEGMENT_INFO.get(segment, ("📦", ""))

        st.markdown("---")
        st.markdown(
            f"""
            <div style="border:2px solid #4caf50; border-radius:12px; padding:24px; text-align:center;
                        background-color:#f1f8e9;">
                <div style="font-size:40px;">{emoji}</div>
                <div style="font-size:26px; font-weight:700; color:#2e7d32;">{segment}</div>
                <div style="font-size:14px; color:#444; margin-top:8px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")
st.caption("Built with Streamlit • KMeans clustering + item-based collaborative filtering")
