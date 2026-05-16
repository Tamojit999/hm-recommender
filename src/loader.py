import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix

@st.cache_resource
def load_models():
    knn_model    = joblib.load("models/knn_model.pkl")
    xgb_model    = joblib.load("models/xgb_model.pkl")
    user_encoder = joblib.load("models/user_encoder.pkl")
    item_encoder = joblib.load("models/item_encoder.pkl")
    return knn_model, xgb_model, user_encoder, item_encoder

@st.cache_data
def load_data():
    df = pd.read_csv("data/cf_data.csv")
    df["t_dat"] = pd.to_datetime(df["t_dat"])
    return df

@st.cache_resource
def build_sparse_matrix(_cf_data):
    user_item = (
        _cf_data
        .groupby(["user_idx", "item_idx"])["interaction_score"]
        .sum()
        .unstack(fill_value=0)
    )
    sparse = csr_matrix(user_item.values)
    return sparse, user_item

@st.cache_data
def build_item_lookup(_cf_data):
    return (
        _cf_data[["item_idx", "article_id"]]
        .drop_duplicates()
        .set_index("item_idx")
    )
