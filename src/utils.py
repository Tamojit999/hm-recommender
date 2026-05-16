def price_tier_label(price):
    if price < 40:
        return "Budget"
    elif price < 80:
        return "Mid"
    else:
        return "Premium"

def price_tier_color(tier):
    return {
        "Budget":  "#4ade80",
        "Mid":     "#e8c97a",
        "Premium": "#c084fc",
    }.get(tier, "#a78bfa")

def season_emoji(season):
    return {
        "Spring": "🌸",
        "Summer": "☀️",
        "Autumn": "🍂",
        "Winter": "❄️",
    }.get(season, "")

def dept_emoji(dept):
    return {
        "Mens":   "👔",
        "Womens": "👗",
        "Kids":   "🎒",
    }.get(dept, "🛍️")

def get_user_profile(customer_id, cf_data):
    rows = cf_data[cf_data["customer_id"] == customer_id]
    if rows.empty:
        return None
    return {
        "total_purchases": len(rows),
        "avg_spend":       round(rows["price_x"].mean(), 2),
        "age_group":       rows["age_group"].mode()[0] if not rows["age_group"].empty else "N/A",
        "fav_dept":        rows["department_name"].mode()[0] if not rows["department_name"].empty else "N/A",
        "fav_type":        rows["product_type_name"].mode()[0] if not rows["product_type_name"].empty else "N/A",
        "spend_tier":      rows["spend_tier"].mode()[0] if "spend_tier" in rows.columns and not rows["spend_tier"].empty else "N/A",
        "last_purchase":   rows["t_dat"].max().strftime("%d %b %Y"),
    }
