import numpy as np
import pandas as pd


def get_similar_items(article_id, item_encoder, knn_model, item_user_matrix, item_lookup, n=20):
    try:
        encoded = item_encoder.transform([article_id])[0]
        distances, indices = knn_model.kneighbors(
            item_user_matrix[encoded], n_neighbors=n + 1
        )
        results = []
        for idx, dist in zip(indices.flatten()[1:], distances.flatten()[1:]):
            rec_article = item_lookup.loc[idx, "article_id"]
            results.append({
                "recommended_article": rec_article,
                "similarity_score": 1 - dist,
            })
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()


def generate_candidates(customer_id, cf_data, item_encoder, knn_model, item_user_matrix, item_lookup, recent_items=3, top_n=50):
    user_history = (
        cf_data[cf_data["customer_id"] == customer_id]
        .sort_values("t_dat", ascending=False)
    )
    if user_history.empty:
        return pd.DataFrame()

    recent_products = user_history["article_id"].unique()[:recent_items]
    candidate_scores = {}

    for item in recent_products:
        similar = get_similar_items(item, item_encoder, knn_model, item_user_matrix, item_lookup, n=20)
        for _, row in similar.iterrows():
            art = row["recommended_article"]
            sc  = row["similarity_score"]
            candidate_scores[art] = candidate_scores.get(art, 0) + sc

    candidate_df = pd.DataFrame({
        "article_id": list(candidate_scores.keys()),
        "candidate_score": list(candidate_scores.values()),
    })

    purchased = set(str(x) for x in user_history["article_id"])
    candidate_df = candidate_df[~candidate_df["article_id"].isin(purchased)]
    return candidate_df.sort_values("candidate_score", ascending=False).head(top_n)


def rank_candidates(customer_id, candidate_df, cf_data, xgb_model):
    if candidate_df.empty:
        return pd.DataFrame()

    user_rows = cf_data[cf_data["customer_id"] == customer_id]
    user_avg_spend     = user_rows["price_x"].mean()
    purchase_frequency = len(user_rows)

    product_stats = (
        cf_data.groupby("article_id")
        .agg(
            purchase_count=("article_id", "count"),
            avg_product_price=("price_x", "mean"),
            recency_score=("recency_score", "mean"),
        )
        .reset_index()
    )

    ranked = candidate_df.merge(product_stats, on="article_id", how="left").fillna(0)
    ranked["user_avg_spend"]      = user_avg_spend
    ranked["purchase_frequency"]  = purchase_frequency
    ranked["price_affinity"]      = abs(ranked["avg_product_price"] - user_avg_spend)

    feature_cols = [
        "user_avg_spend",
        "purchase_frequency",
        "purchase_count",
        "avg_product_price",
        "recency_score",
        "price_affinity",
    ]
    X = ranked[feature_cols]
    ranked["purchase_probability"] = xgb_model.predict_proba(X)[:, 1]

    return ranked.sort_values("purchase_probability", ascending=False)


def enrich_with_metadata(ranked_df, cf_data):
    meta_cols = ["article_id", "prod_name", "product_type_name",
                 "colour_group_name", "department_name", "season", "price_x"]
    meta = (
        cf_data[meta_cols]
        .drop_duplicates("article_id")
    )
    return ranked_df.merge(meta, on="article_id", how="left")
