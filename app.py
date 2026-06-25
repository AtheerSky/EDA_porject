# -*- coding: utf-8 -*-

"""
Riyadh Restaurants — Interactive EDA Dashboard
Run with:  streamlit run r
iyadh_restaurants_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Riyadh Restaurants EDA",
    page_icon="🍽️",
    layout="wide",
)

# ── Consistent plot style ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 110,
})

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load dataset and apply the same cleaning steps used in the notebook."""
    df_raw = pd.read_csv("riyadh_resturants_clean.csv")

    # --- same cleaning steps as the notebook ---
    df = df_raw.copy()
    df = df.drop_duplicates()
    df["likes"] = df["likes"].fillna(0)
    if "ratingSignals" in df.columns:
        df = df.drop(columns=["ratingSignals"])

    # Asian flag
    asian_categories = [
        "Japanese Restaurant", "Chinese Restaurant", "Korean Restaurant",
        "Thai Restaurant", "Indian Restaurant", "Vietnamese Restaurant",
        "Asian Restaurant", "Sushi Restaurant", "Ramen Restaurant",
        "Dim Sum Restaurant",
    ]
    df["Asian"] = df["categories"].apply(
        lambda x: any(cat in str(x) for cat in asian_categories)
    )
    
    df["price_numeric"] = df["price"].map({
        "Cheap": 1,
        "Moderate": 2,
        "Expensive": 3,
        "Very Expensive": 4
    })
    
    return df, asian_categories
df, ASIAN_CATS = load_data()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Flag_of_Saudi_Arabia.svg/320px-Flag_of_Saudi_Arabia.svg.png",
    width=80,
)
st.sidebar.title("🍽️ Riyadh Restaurants")
st.sidebar.markdown("**EDA Dashboard** — Atheer")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "Go to section",
    [
        "📋 Dataset Overview",
        "⭐ Ratings & Prices",
        "❤️ Customer Engagement",
        "🍱 Restaurant Categories",
        "🌏 Asian Restaurants",
        "⚙️ Feature Engineering",
        "🗺️ Geographic Map",
    ],
)

st.sidebar.markdown("---")
price_filter = st.sidebar.selectbox(
    "Filter by Price",
    ["All"] + sorted(df["price"].dropna().unique().tolist())
)

if price_filter != "All":
    df = df[df["price"] == price_filter]
st.sidebar.caption(
    "Dataset: [Riyadh Restaurants 20K](https://www.kaggle.com/datasets/fahd09/riyadh-restaurants-20k)"
)

# ── Helper ────────────────────────────────────────────────────────────────────
def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════
if section == "📋 Dataset Overview":
    st.title("📋 Dataset Overview")
    st.markdown(
        "A quick look at the cleaned dataset before diving into the analysis."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Restaurants", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("With Rating", f"{df['rating'].notna().sum():,}")
    col4.metric("With Price", f"{df['price'].notna().sum():,}")

    st.markdown("### First 5 rows")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Missing Values")
    missing = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2),
    }).sort_values("Missing Count", ascending=False)
    st.dataframe(missing, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(df.describe(include="all"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Ratings & Prices
# ══════════════════════════════════════════════════════════════════════════════
elif section == "⭐ Ratings & Prices":
    st.title("⭐ Ratings & Prices")

    # --- Rating distribution ---
    st.subheader("Distribution of Restaurant Ratings")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="rating", bins=15, ax=ax)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Restaurants")
    ax.set_title("Distribution of Restaurant Ratings")
    show_fig(fig)

    st.info(
        f"Only **{df['rating'].notna().sum():,}** of {len(df):,} restaurants "
        "have a rating. Missing ratings were kept as-is to avoid bias."
    )

    # --- Price distribution ---
    st.subheader("Distribution of Price Levels")
    fig, ax = plt.subplots(figsize=(6, 5))
    price_order = sorted(df["price"].dropna().unique())
    sns.countplot(data=df, x="price", order=price_order, ax=ax)
    ax.set_xlabel("Price Level")
    ax.set_ylabel("Number of Restaurants")
    ax.set_title("Distribution of Restaurant Price Levels")
    show_fig(fig)

    # --- Price vs Rating ---
    st.subheader("Ratings by Price Level")

    price_summary = (
        df.groupby("price")["rating"]
        .agg(["count", "mean", "median"])
        .rename(columns={"count": "Count", "mean": "Mean Rating", "median": "Median Rating"})
    )
    st.dataframe(price_summary.style.format({"Mean Rating": "{:.2f}", "Median Rating": "{:.2f}"}),
                 use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df.dropna(subset=["price", "rating"]),
                x="price", y="rating", order=price_order, ax=ax)
    ax.set_xlabel("Price Level")
    ax.set_ylabel("Rating")
    ax.set_title("Restaurant Ratings by Price Level")
    show_fig(fig)

    st.success(
        "💡 Higher price levels do **not** always guarantee higher ratings — "
        "mid-range restaurants often perform comparably to expensive ones."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Customer Engagement
# ══════════════════════════════════════════════════════════════════════════════
elif section == "❤️ Customer Engagement":
    st.title("❤️ Customer Engagement")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Distribution of Likes")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data=df, x="likes", bins=20, ax=ax)
        ax.set_xlabel("Likes")
        ax.set_ylabel("Number of Restaurants")
        ax.set_title("Distribution of Restaurant Likes")
        show_fig(fig)

    with col_right:
        st.subheader("Distribution of Photos")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(data=df, x="photos", bins=20, ax=ax)
        ax.set_xlabel("Number of Photos")
        ax.set_ylabel("Number of Restaurants")
        ax.set_title("Distribution of Restaurant Photos")
        show_fig(fig)

    # Scatter: Likes vs Rating
    st.subheader("Likes vs. Rating")

    # Slider to filter number of likes (avoids overplotting)
    max_likes = int(df["likes"].quantile(0.99))
    like_cap = st.slider(
        "Show restaurants with likes up to:",
        min_value=10, max_value=max_likes, value=max_likes,
        help="Trim extreme outliers to see the main trend more clearly.",
    )
    df_scatter = df[df["likes"] <= like_cap].dropna(subset=["rating"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df_scatter, x="likes", y="rating", alpha=0.4, ax=ax)
    ax.set_xlabel("Likes")
    ax.set_ylabel("Rating")
    ax.set_title("Relationship Between Likes and Ratings")
    show_fig(fig)

    # Correlation heatmap
    st.subheader("Correlation Between Numerical Variables")
    num_cols = [c for c in ["rating", "likes", "photos", "tips"] if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Correlation Between Numerical Variables")
    show_fig(fig)

    st.success(
        "💡 Customer engagement (likes, photos, tips) shows a **positive** "
        "correlation with ratings — more engaged customers tend to rate higher."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Restaurant Categories
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🍱 Restaurant Categories":
    st.title("🍱 Restaurant Categories")

    all_categories = (
        df["categories"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
    )
    category_counts = all_categories.value_counts()

    top_n = st.slider("Show top N categories:", 5, 30, 15)

    fig, ax = plt.subplots(figsize=(10, 6))
    category_counts.head(top_n).plot(kind="bar", ax=ax)
    ax.set_title(f"Top {top_n} Restaurant Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Restaurants")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    show_fig(fig)

    st.markdown("### Full Category Table")
    cat_df = category_counts.reset_index()
    cat_df.columns = ["Category", "Count"]
    st.dataframe(cat_df, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Asian Restaurants
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🌏 Asian Restaurants":
    st.title("🌏 Asian Restaurants")

    asian_df = df[df["Asian"]]
    non_asian = len(df) - len(asian_df)
    pct = len(asian_df) / len(df) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Asian Restaurants", f"{len(asian_df):,}")
    col2.metric("Non-Asian Restaurants", f"{non_asian:,}")
    col3.metric("Asian Share", f"{pct:.1f}%")

    # Asian vs Non-Asian bar
    st.subheader("Asian vs Non-Asian Restaurants")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=df, x="Asian", ax=ax)
    ax.set_xticklabels(["Non-Asian", "Asian"])
    ax.set_xlabel("")
    ax.set_ylabel("Number of Restaurants")
    ax.set_title("Asian vs Non-Asian Restaurants")
    show_fig(fig)

    # Asian sub-categories
    st.subheader("Breakdown of Asian Restaurant Types")
    asian_cat_series = (
        asian_df["categories"]
        .str.split(",")
        .explode()
        .str.strip()
    )
    asian_cat_series = asian_cat_series[asian_cat_series.isin(ASIAN_CATS)]
    asian_cat_counts = asian_cat_series.value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    asian_cat_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top Asian Restaurant Categories")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of Restaurants")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    show_fig(fig)

    st.markdown("### Asian Restaurants — Ratings vs Non-Asian")
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_df = df.dropna(subset=["rating"])
    sns.boxplot(data=plot_df, x="Asian", y="rating",
                order=[False, True], ax=ax)
    ax.set_xticklabels(["Non-Asian", "Asian"])
    ax.set_xlabel("")
    ax.set_ylabel("Rating")
    ax.set_title("Rating Distribution: Asian vs Non-Asian")
    show_fig(fig)

    st.success(
        "💡 Japanese, Chinese, Indian, Korean, and Thai restaurants are the "
        "most common Asian categories in Riyadh."
    )
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
elif section == "⚙️ Feature Engineering":

    st.title("⚙️ Feature Engineering")

    st.markdown(
        """
        Feature Engineering means creating new variables
        from existing data to improve analysis.
        """
    )

    st.subheader("Asian Restaurant Feature")

    st.dataframe(
        df[["categories", "Asian"]].head(10),
        use_container_width=True
    )

    st.subheader("Price Numeric Feature")
    
    st.dataframe(
        df[["price", "price_numeric"]]
        .drop_duplicates()
        .sort_values("price_numeric")
    )

    st.success(
        """
        Created two new features:

        • Asian (True / False)

        • price_numeric (1–4)

        These engineered features simplify analysis and visualization.
        """
    )
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Geographic Map
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🗺️ Geographic Map":
    st.title("🗺️ Geographic Distribution of Restaurants")

    map_df = df.dropna(subset=["lat", "lng"])

    # Filter by rating
    show_unrated = st.checkbox("Include unrated restaurants", value=False)
    if not show_unrated:
        map_df = map_df.dropna(subset=["rating"])

    # Optional: filter by Asian
    asian_only = st.checkbox("Show only Asian restaurants", value=False)
    if asian_only:
        map_df = map_df[map_df["Asian"]]


    min_rating = st.slider( "Minimum Rating", min_value=0.0, max_value=5.0, value=3.0, step=0.1)
    map_df = map_df[ (map_df["rating"].isna()) | (map_df["rating"] >= min_rating)]

    st.markdown(f"Showing **{len(map_df):,}** restaurants on the map.")

    # Streamlit built-in map (requires lat/lon columns)
    st.map(map_df.rename(columns={"lat": "latitude", "lng": "longitude"})[["latitude", "longitude"]])

    # Matplotlib scatter coloured by rating
    st.subheader("Rating Heatmap (Scatter)")
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        map_df["lng"], map_df["lat"],
        c=map_df["rating"], cmap="viridis",
        alpha=0.5, s=5,
    )
    plt.colorbar(scatter, ax=ax, label="Rating")
    ax.set_title("Distribution of Restaurants Across Riyadh")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    show_fig(fig)

    st.success(
        "💡 Restaurants are concentrated in specific districts of Riyadh "
        "rather than being evenly spread across the city."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("EDA Project · Riyadh Restaurants 20K · Student: Atheer")
