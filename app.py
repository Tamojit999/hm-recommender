import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from src.loader import load_models, load_data, build_sparse_matrix, build_item_lookup
from src.recommender import generate_candidates, rank_candidates, enrich_with_metadata
from src.utils import (
    price_tier_label, price_tier_color,
    season_emoji, dept_emoji, get_user_profile
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="H&M Fashion Feed",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  —  Deep Purple × Rose Gold Dark
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Outfit:wght@300;400;500&display=swap');

/* ── Global dark canvas ── */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background-color: #0d0b14 !important;
    color: #e8dff5 !important;
}
.block-container { padding-top: 1.5rem !important; }

/* ── Page background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 10%, rgba(90,30,140,0.25) 0%, transparent 55%),
                radial-gradient(ellipse at 80% 80%, rgba(180,100,120,0.15) 0%, transparent 50%),
                #0d0b14 !important;
}

/* ── Header ── */
.hm-header {
    background: linear-gradient(135deg, #1a0e2e 0%, #2a1245 50%, #1e0d38 100%);
    border: 1px solid rgba(196,143,180,0.25);
    border-radius: 20px;
    padding: 2.8rem 2.4rem 2.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hm-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(196,143,80,0.2) 0%, transparent 65%);
    border-radius: 50%;
}
.hm-header::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(140,50,180,0.25) 0%, transparent 65%);
    border-radius: 50%;
}
.hm-header h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e8c97a, #d4a0c0, #b87acc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 0.5px;
}
.hm-header p {
    color: rgba(220,200,240,0.55);
    margin: 0.5rem 0 0;
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}

/* ── Profile card ── */
.profile-card {
    background: linear-gradient(145deg, #1a0e2e, #221540);
    border: 1px solid rgba(196,143,80,0.3);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.profile-card .name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    background: linear-gradient(90deg, #e8c97a, #d4a0c0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.profile-stat {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid rgba(196,143,180,0.1);
    color: rgba(220,200,240,0.55);
}
.profile-stat span:last-child {
    color: #e8dff5;
    font-weight: 500;
}

/* ── Product card ── */
.product-card {
    background: linear-gradient(145deg, #160f28, #1e1436);
    border: 1px solid rgba(196,143,180,0.18);
    border-radius: 16px;
    padding: 1.3rem;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    height: 100%;
}
.product-card:hover {
    transform: translateY(-4px);
    border-color: rgba(196,143,80,0.5);
    box-shadow: 0 12px 32px rgba(100,30,150,0.35);
}
.product-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #f0e6ff;
    margin-bottom: 0.3rem;
    line-height: 1.3;
}
.product-type {
    font-size: 0.72rem;
    color: rgba(196,143,180,0.7);
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.product-price {
    font-size: 1.2rem;
    font-weight: 500;
    background: linear-gradient(90deg, #e8c97a, #c8a060);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.7rem;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    margin-right: 4px;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
}
.prob-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    height: 4px;
    margin-top: 0.9rem;
}
.prob-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, #7b2d8b, #c8608a, #e8c97a);
}
.prob-label {
    font-size: 0.7rem;
    color: rgba(196,143,180,0.6);
    margin-top: 0.35rem;
}

/* ── Section title ── */
.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8dff5;
    margin-bottom: 1.2rem;
    border-left: 3px solid #e8c97a;
    padding-left: 0.9rem;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(145deg, #160f28, #1e1436) !important;
    border: 1px solid rgba(196,143,180,0.2) !important;
    border-radius: 14px !important;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="metric-container"] label {
    color: rgba(196,143,180,0.7) !important;
    font-size: 0.78rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e8c97a !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #120b22 0%, #0d0818 100%) !important;
    border-right: 1px solid rgba(196,143,180,0.15) !important;
}
[data-testid="stSidebar"] * { color: #e8dff5 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    color: rgba(196,143,180,0.7) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(196,143,180,0.15) !important;
}

/* ── Progress bars ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #7b2d8b, #e8c97a) !important;
    border-radius: 4px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
}

/* ── Dividers ── */
hr { border-color: rgba(196,143,180,0.15) !important; }

/* ── Headings in main area ── */
h1, h2, h3 { color: #e8dff5 !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, #4a1a6e, #7b2d8b) !important;
    border: 1px solid rgba(196,143,80,0.4) !important;
    color: #e8c97a !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: 0.3px !important;
}

/* ── Primary button ── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #5a1e8c, #8b3da8) !important;
    border: 1px solid rgba(196,143,80,0.5) !important;
    color: #e8c97a !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: rgba(196,143,180,0.6);
    font-size: 1rem;
}
.empty-state .ready-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    color: #e8dff5;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
@st.cache_resource
def get_all_resources():
    knn_model, xgb_model, user_encoder, item_encoder = load_models()
    cf_data = load_data()
    sparse_matrix, user_item = build_sparse_matrix(cf_data)
    item_user_matrix = sparse_matrix.T
    item_lookup = build_item_lookup(cf_data)
    return knn_model, xgb_model, user_encoder, item_encoder, cf_data, item_user_matrix, item_lookup

with st.spinner("Loading models..."):
    knn_model, xgb_model, user_encoder, item_encoder, cf_data, item_user_matrix, item_lookup = get_all_resources()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛍️ H&M Recommender")
    st.markdown("---")

    all_customers = sorted(cf_data["customer_id"].unique().tolist())
    selected_customer = st.selectbox(
        "Select Customer ID",
        all_customers,
        index=0,
    )

    st.markdown("---")
    st.markdown("**Feed Settings**")
    top_n = st.slider("Number of recommendations", 5, 20, 10)
    recent_items = st.slider("Recent items to consider", 1, 5, 3)

    st.markdown("---")
    st.markdown("**Filter by Department**")
    depts = ["All"] + sorted(cf_data["department_name"].dropna().unique().tolist())
    selected_dept = st.selectbox("Department", depts)

    st.markdown("**Filter by Season**")
    seasons = ["All"] + sorted(cf_data["season"].dropna().unique().tolist())
    selected_season = st.selectbox("Season", seasons)

    st.markdown("---")
    run_btn = st.button("🚀 Generate Feed", use_container_width=True, type="primary")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="hm-header">
    <h1>🛍️ H&M Fashion Feed</h1>
    <p>Personalised recommendations powered by collaborative filtering + XGBoost ranking</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
if run_btn or "feed_data" in st.session_state:

    if run_btn:
        # Generate fresh feed
        with st.spinner("Finding similar items..."):
            candidates = generate_candidates(
                selected_customer, cf_data,
                item_encoder, knn_model,
                item_user_matrix, item_lookup,
                recent_items=recent_items,
                top_n=50,
            )

        if candidates.empty:
            st.markdown('<div class="empty-state">⚠️ No candidates found for this user. Try a different customer ID.</div>', unsafe_allow_html=True)
            st.stop()

        with st.spinner("Ranking by purchase probability..."):
            ranked = rank_candidates(selected_customer, candidates, cf_data, xgb_model)
            enriched = enrich_with_metadata(ranked, cf_data)

        st.session_state["feed_data"] = enriched
        st.session_state["selected_customer"] = selected_customer

    feed = st.session_state["feed_data"]
    cust = st.session_state.get("selected_customer", selected_customer)

    # Apply filters
    if selected_dept != "All" and "department_name" in feed.columns:
        feed = feed[feed["department_name"] == selected_dept]
    if selected_season != "All" and "season" in feed.columns:
        feed = feed[feed["season"] == selected_season]

    feed = feed.head(top_n)

    # ── User Profile + Summary Metrics ──
    col_profile, col_metrics = st.columns([1, 2])

    with col_profile:
        profile = get_user_profile(cust, cf_data)
        if profile:
            st.markdown(f"""
            <div class="profile-card">
                <div class="name">👤 {cust}</div>
                <div class="profile-stat"><span>Total purchases</span><span>{profile['total_purchases']}</span></div>
                <div class="profile-stat"><span>Avg spend</span><span>£{profile['avg_spend']}</span></div>
                <div class="profile-stat"><span>Age group</span><span>{profile['age_group']}</span></div>
                <div class="profile-stat"><span>Fav dept</span><span>{dept_emoji(profile['fav_dept'])} {profile['fav_dept']}</span></div>
                <div class="profile-stat"><span>Fav type</span><span>{profile['fav_type']}</span></div>
                <div class="profile-stat"><span>Last purchase</span><span>{profile['last_purchase']}</span></div>
            </div>
            """, unsafe_allow_html=True)

    with col_metrics:
        if not feed.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items in Feed", len(feed))
            m2.metric("Avg Probability", f"{feed['purchase_probability'].mean():.1%}")
            m3.metric("Avg Price", f"£{feed['avg_product_price'].mean():.0f}")
            m4.metric("Top Score", f"{feed['purchase_probability'].max():.1%}")

            # Top category breakdown
            if "product_type_name" in feed.columns:
                st.markdown("**Category Breakdown**")
                type_counts = feed["product_type_name"].value_counts().head(4)
                for t, c in type_counts.items():
                    pct = c / len(feed)
                    st.progress(pct, text=f"{t}  ·  {c} items")

    st.markdown("---")

    # ── Feed Grid ──
    st.markdown(f'<div class="section-title">Your Personalised Feed — {len(feed)} items</div>', unsafe_allow_html=True)

    if feed.empty:
        st.markdown('<div class="empty-state">No items match the selected filters.</div>', unsafe_allow_html=True)
    else:
        cols_per_row = 4
        rows = [feed.iloc[i:i+cols_per_row] for i in range(0, len(feed), cols_per_row)]

        for row_df in rows:
            cols = st.columns(cols_per_row)
            for col, (_, item) in zip(cols, row_df.iterrows()):
                prob   = item.get("purchase_probability", 0)
                price  = item.get("avg_product_price", item.get("price_x", 0))
                tier   = price_tier_label(price)
                t_color = price_tier_color(tier)
                season = item.get("season", "")
                dept   = item.get("department_name", "")
                s_emoji = season_emoji(season)
                d_emoji = dept_emoji(dept)
                name   = item.get("prod_name", item.get("article_id", "Unknown"))
                ptype  = item.get("product_type_name", "")
                colour = item.get("colour_group_name", "")
                bar_w  = int(prob * 100)

                with col:
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="product-name">{name}</div>
                        <div class="product-type">{ptype} · {colour}</div>
                        <div class="product-price">£{price:.2f}</div>
                        <div>
                            <span class="badge" style="background:{t_color}22;color:{t_color};border:1px solid {t_color}55">{tier}</span>
                            <span class="badge" style="background:rgba(123,45,139,0.25);color:#d4a0c0;border:1px solid rgba(196,143,180,0.3)">{d_emoji} {dept}</span>
                            <span class="badge" style="background:rgba(196,143,80,0.15);color:#e8c97a;border:1px solid rgba(232,201,122,0.3)">{s_emoji} {season}</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{bar_w}%"></div>
                        </div>
                        <div class="prob-label">Purchase probability: {prob:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Download button ──
        st.markdown("---")
        csv = feed[["article_id", "prod_name", "product_type_name",
                    "department_name", "season", "avg_product_price",
                    "purchase_probability"]].to_csv(index=False)
        st.download_button(
            "⬇️ Download Feed as CSV",
            data=csv,
            file_name=f"feed_{cust}.csv",
            mime="text/csv",
        )

else:
    # Welcome state
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:3rem;margin-bottom:1rem">✦</div>
        <div class="ready-title">Ready to style your feed</div>
        <div>Select a customer from the sidebar and click <strong style="color:#e8c97a">Generate Feed</strong></div>
    </div>
    """, unsafe_allow_html=True)

    # Show quick stats
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", cf_data["customer_id"].nunique())
    c2.metric("Total Articles",  cf_data["article_id"].nunique())
    c3.metric("Transactions",    len(cf_data))
    c4.metric("Departments",     cf_data["department_name"].nunique())
