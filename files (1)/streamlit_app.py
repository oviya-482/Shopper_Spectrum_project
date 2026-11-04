"""
🛒 Shopper Spectrum – Streamlit App
Customer Segmentation & Product Recommendation System
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    with open(f"{models_dir}/kmeans.pkl", "rb") as f:
        km = pickle.load(f)
    with open(f"{models_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{models_dir}/label_map.pkl", "rb") as f:
        label_map = pickle.load(f)
    sim_df = pd.read_pickle(f"{models_dir}/similarity_matrix.pkl")
    return km, scaler, label_map, sim_df

km, scaler, label_map, sim_df = load_models()

SEGMENT_INFO = {
    "High-Value": {
        "icon": "🏆",
        "color": "#2ecc71",
        "bg": "#eafaf1",
        "desc": "Recent, frequent, and big spenders. Your VIP customers — reward and retain them.",
        "actions": ["Offer exclusive loyalty rewards", "Early access to new products", "Personalized VIP offers"],
    },
    "Regular": {
        "icon": "🔄",
        "color": "#3498db",
        "bg": "#eaf4fb",
        "desc": "Steady purchasers with moderate spend. Nurture them toward High-Value status.",
        "actions": ["Cross-sell complementary products", "Volume discount promotions", "Subscription programs"],
    },
    "Occasional": {
        "icon": "💤",
        "color": "#f39c12",
        "bg": "#fef9e7",
        "desc": "Rare, low-value purchasers. Activate with targeted campaigns.",
        "actions": ["Win-back email campaigns", "Limited-time discounts", "Product discovery newsletters"],
    },
    "At-Risk": {
        "icon": "⚠️",
        "color": "#e74c3c",
        "bg": "#fdedec",
        "desc": "Once-active customers who haven't purchased recently. Act fast to retain them.",
        "actions": ["Re-engagement campaigns", "Personalized 'we miss you' offers", "Feedback surveys"],
    },
}


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shopping-cart.png", width=70)
    st.title("Shopper Spectrum")
    st.caption("E-Commerce Analytics Platform")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🎯 Product Recommendations", "👥 Customer Segmentation", "ℹ️ About"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**Dataset:** UCI Online Retail")
    st.markdown("**Model:** KMeans (k=4) + Cosine Similarity CF")


# ── Page: Product Recommendations ─────────────────────────
if page == "🎯 Product Recommendations":
    st.title("🎯 Product Recommendation Engine")
    st.markdown(
        "Enter a product name to discover **5 similar products** based on "
        "customer purchasing patterns (Item-Based Collaborative Filtering)."
    )
    st.markdown("---")

    # Autocomplete-style selectbox from available products
    available_products = sorted(sim_df.index.tolist())
    col1, col2 = st.columns([3, 1])
    with col1:
        product_input = st.selectbox(
            "🔍 Select or search a product",
            options=[""] + available_products,
            index=0,
            help="Start typing to filter products",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        get_recs = st.button("Get Recommendations →", type="primary", use_container_width=True)

    if get_recs:
        if not product_input:
            st.warning("Please select a product first.")
        else:
            with st.spinner("Finding similar products..."):
                scores = sim_df[product_input].sort_values(ascending=False).drop(product_input)
                top5 = scores.head(5)

            st.success(f"Top 5 products similar to **{product_input}**")
            for rank, (prod, score) in enumerate(top5.items(), 1):
                bar_width = int(score * 100)
                st.markdown(
                    f"""
                    <div style="background:#f8f9fa;border-radius:10px;padding:14px 18px;margin-bottom:10px;
                                border-left:5px solid #3498db;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:16px;font-weight:600;">#{rank} &nbsp; {prod}</span>
                            <span style="color:#3498db;font-weight:700;font-size:15px;">
                                {score:.2%} similarity
                            </span>
                        </div>
                        <div style="height:6px;background:#dee2e6;border-radius:3px;margin-top:8px;">
                            <div style="width:{bar_width}%;height:6px;background:#3498db;border-radius:3px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        # Show sample products
        st.markdown("#### 💡 Popular Products to Try")
        samples = [
            "WHITE HANGING HEART T-LIGHT HOLDER",
            "REGENCY CAKESTAND 3 TIER",
            "JUMBO BAG RED RETROSPOT",
            "PARTY BUNTING",
            "ASSORTED COLOUR BIRD ORNAMENT",
        ]
        cols = st.columns(len(samples))
        for col, prod in zip(cols, samples):
            with col:
                if st.button(prod, key=prod, use_container_width=True):
                    st.session_state["selected_product"] = prod
                    st.rerun()


# ── Page: Customer Segmentation ────────────────────────────
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation Predictor")
    st.markdown(
        "Enter a customer's **RFM values** to predict their segment and "
        "receive tailored marketing recommendations."
    )
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📥 Customer RFM Input")
        recency = st.number_input(
            "🕐 Recency (days since last purchase)",
            min_value=1, max_value=730, value=30, step=1,
            help="Fewer days = more recent customer",
        )
        frequency = st.number_input(
            "🔁 Frequency (number of purchases)",
            min_value=1, max_value=500, value=5, step=1,
            help="Total distinct invoices/orders",
        )
        monetary = st.number_input(
            "💷 Monetary (total spend in £)",
            min_value=1.0, max_value=300000.0, value=500.0, step=50.0,
            help="Total £ spent by the customer",
        )
        predict_btn = st.button("🔮 Predict Cluster", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📊 Segment Reference Guide")
        for seg, info in SEGMENT_INFO.items():
            st.markdown(
                f"""<div style="background:{info['bg']};border-left:4px solid {info['color']};
                    border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                    <strong style="color:{info['color']};">{info['icon']} {seg}</strong><br>
                    <small>{info['desc']}</small>
                </div>""",
                unsafe_allow_html=True,
            )

    if predict_btn:
        rfm_input = np.array([[recency, frequency, monetary]])
        rfm_scaled_input = scaler.transform(rfm_input)
        cluster_id = km.predict(rfm_scaled_input)[0]
        segment = label_map[cluster_id]
        info = SEGMENT_INFO[segment]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        st.markdown(
            f"""
            <div style="background:{info['bg']};border:2px solid {info['color']};
                border-radius:14px;padding:24px 30px;text-align:center;">
                <div style="font-size:52px;">{info['icon']}</div>
                <div style="font-size:28px;font-weight:700;color:{info['color']};margin:8px 0;">
                    {segment} Customer
                </div>
                <div style="font-size:15px;color:#555;max-width:480px;margin:0 auto;">
                    {info['desc']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### 🚀 Recommended Actions")
        for action in info["actions"]:
            st.markdown(f"- ✅ {action}")

        st.markdown("#### 📊 Your Input Summary")
        metrics_cols = st.columns(3)
        metrics_cols[0].metric("Recency", f"{recency} days")
        metrics_cols[1].metric("Frequency", f"{frequency} orders")
        metrics_cols[2].metric("Monetary", f"£{monetary:,.2f}")


# ── Page: About ────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About Shopper Spectrum")
    st.markdown("""
    ## 🛒 Project Overview

    **Shopper Spectrum** is an end-to-end E-Commerce analytics system that combines:
    - **RFM Analysis** for customer value scoring
    - **KMeans Clustering** for customer segmentation
    - **Item-Based Collaborative Filtering** for product recommendations

    ---

    ## 📊 Dataset
    - **Source:** UCI Online Retail Dataset
    - **Period:** December 2022 – December 2023
    - **Raw records:** ~541,909 transactions
    - **Clean records after preprocessing:** ~397,884

    ---

    ## 🔧 Technical Stack
    | Component | Technology |
    |---|---|
    | Data Processing | Pandas, NumPy |
    | Machine Learning | Scikit-learn (KMeans, StandardScaler) |
    | Recommendation | Cosine Similarity (Collaborative Filtering) |
    | Visualization | Matplotlib, Seaborn |
    | Web App | Streamlit |

    ---

    ## 📌 Customer Segments

    | Segment | Characteristics |
    |---|---|
    | 🏆 High-Value | Recent, frequent, high spenders — VIP customers |
    | 🔄 Regular | Steady purchasers with moderate spend |
    | 💤 Occasional | Rare, low-frequency buyers |
    | ⚠️ At-Risk | Previously active, now lapsing customers |

    ---

    ## 🎯 How the Recommendation Works
    1. Build a **CustomerID × Product** purchase matrix
    2. Compute **cosine similarity** between all product pairs
    3. For any input product, return the **top 5 most similar** products
    """)
