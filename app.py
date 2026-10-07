"""
Smart Crop Advisor - ML mini project
Analysis: crop-profile matching and K-Means
UI: Streamlit
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATHS = (
    BASE_DIR / "data" / "karnataka_crop_dataset_V2.csv",
    BASE_DIR / "karnataka_crop_dataset_V2.csv",
    BASE_DIR / "Crop_recommendation.csv",
)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
FEATURE_LABELS = {
    "N": "Nitrogen (N) (kg/ha)",
    "P": "Phosphorus (P) (kg/ha)",
    "K": "Potassium (K) (kg/ha)",
    "temperature": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "ph": "Soil pH (unitless)",
    "rainfall": "Rainfall (mm)",
}

st.set_page_config(page_title="Smart Crop Advisor", page_icon="🌾", layout="wide")


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner="Evaluating cluster counts...")
def cluster_diagnostics(values, seed):
    ks = list(range(2, min(10, len(values) - 1) + 1))
    inertia, silhouette = [], []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(values)
        inertia.append(model.inertia_)
        label_count = len(set(model.labels_))
        if 1 < label_count < len(values):
            score = silhouette_score(
                values,
                model.labels_,
                sample_size=min(2000, len(values)),
                random_state=seed,
            )
        else:
            score = float("nan")
        silhouette.append(score)
    return ks, inertia, silhouette


@st.cache_data(show_spinner="Grouping field conditions...")
def fit_clusters(values, n_clusters, seed):
    return KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    ).fit_predict(values)


def crop_profile_suggestions(df, sample):
    """Rank crops by the standardized distance from their average field conditions."""
    feature_std = df[FEATURES].std().replace(0, 1)
    crop_profiles = df.groupby(TARGET)[FEATURES].mean()
    distances = crop_profiles.sub(sample.iloc[0], axis="columns").div(feature_std).abs().mean(axis=1)
    suggestions = pd.DataFrame({
        "Crop": distances.index,
        "Profile match": 100 / (1 + distances.values),
        "Average standardized difference": distances.values,
    }).sort_values("Average standardized difference")
    return suggestions.reset_index(drop=True), feature_std


def extreme_input_alerts(df, values):
    """Describe inputs outside dataset percentile ranges and flag extreme pH."""
    typical_ranges = df[FEATURES].quantile([0.05, 0.95])
    alerts = []
    for feature in FEATURES:
        value = values[feature]
        lower, upper = typical_ranges.loc[0.05, feature], typical_ranges.loc[0.95, feature]
        reasons = []
        if value < lower:
            reasons.append(f"below the typical range ({lower:.1f}–{upper:.1f})")
        elif value > upper:
            reasons.append(f"above the typical range ({lower:.1f}–{upper:.1f})")

        if feature == "ph" and value < 5.5:
            reasons.append("too acidic (pH below 5.5)")
        elif feature == "ph" and value > 7.5:
            reasons.append("too basic (pH above 7.5)")

        if reasons:
            alerts.append(f"{FEATURE_LABELS[feature]}: {', and '.join(reasons)}.")
    return alerts


# ---------------------------- Sidebar ---------------------------------
st.sidebar.title("🌾 Smart Crop Advisor")
st.session_state.setdefault("random_seed", 42)
with st.sidebar.form("analysis_settings"):
    uploaded = st.file_uploader("Or upload the CSV here", type="csv")
    seed_input = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999,
        value=st.session_state["random_seed"],
    )
    apply_settings = st.form_submit_button("Load dataset and apply settings")
if apply_settings:
    st.session_state["random_seed"] = seed_input
dataset_path = next((path for path in DATASET_PATHS if path.is_file()), None)
try:
    if uploaded:
        df = pd.read_csv(uploaded)
    elif dataset_path is not None:
        df = load_data(str(dataset_path))
    else:
        st.error(
            "Dataset not found. Place karnataka_crop_dataset_V2.csv in the "
            "'data' folder, or Crop_recommendation.csv beside app.py, "
            "or upload a CSV in the sidebar."
        )
        st.stop()
except FileNotFoundError:
    st.error("The selected dataset file could not be found. Check the file location and try again.")
    st.stop()
required_columns = FEATURES + [TARGET]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"Dataset is missing required columns: {', '.join(missing_columns)}.")
    st.stop()
try:
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="raise")
except (TypeError, ValueError) as error:
    st.error(f"All seven model input columns must contain numeric values. Details: {error}")
    st.stop()
if df[required_columns].isna().any().any():
    st.error("The dataset contains missing values in required model columns. Please clean the CSV and try again.")
    st.stop()
if df.empty:
    st.error("The selected dataset contains no rows.")
    st.stop()

seed = st.session_state["random_seed"]
classes = sorted(df[TARGET].unique())

st.title("🌾 Smart Crop Advisor")
st.caption("Explore crop data, match field conditions to crop profiles, and discover farming zones.")
st.caption(f"Using {len(df):,} records from `{uploaded.name if uploaded else dataset_path.name}`.")

tabs = st.tabs(["📊 Data", "🧩 K-Means", "🔮 Crop Advisor"])

# ----------------------------- Data tab -------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Features", len(FEATURES))
    c3.metric("Crops", len(classes))
    st.dataframe(df.head(20).rename(columns=FEATURE_LABELS), width="stretch")
    st.subheader("Feature distribution")
    st.session_state.setdefault("data_feature", FEATURES[0])
    with st.form("data_feature_settings"):
        feature_input = st.selectbox(
            "Feature",
            FEATURES,
            index=FEATURES.index(st.session_state["data_feature"]),
            format_func=lambda feature: FEATURE_LABELS[feature],
        )
        apply_feature = st.form_submit_button("Show distribution")
    if apply_feature:
        st.session_state["data_feature"] = feature_input
    feat = st.session_state["data_feature"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df[feat], bins=30, color="seagreen")
    ax.set_xlabel(FEATURE_LABELS[feat])
    st.pyplot(fig)
    st.subheader("Average conditions per crop")
    st.dataframe(
        df.groupby(TARGET)[FEATURES].mean().round(1).rename(columns=FEATURE_LABELS),
        width="stretch",
    )

# ---------------------------- K-Means tab -----------------------------
with tabs[1]:
    st.subheader("K-Means: discover natural farming zones")
    st.caption("Unsupervised: the crop label is NOT used for clustering.")
    st.session_state.setdefault("kmeans_features", FEATURES)
    st.session_state.setdefault("kmeans_clusters", 4)
    with st.form("kmeans_settings"):
        chosen_input = st.multiselect(
            "Features to cluster on",
            FEATURES,
            default=st.session_state["kmeans_features"],
            format_func=lambda feature: FEATURE_LABELS[feature],
        )
        cluster_input = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=st.session_state["kmeans_clusters"],
        )
        run_kmeans = st.form_submit_button("Apply settings and run K-Means")
    if run_kmeans:
        st.session_state["kmeans_features"] = chosen_input
        st.session_state["kmeans_clusters"] = cluster_input
        st.session_state["kmeans_run"] = True
    chosen = st.session_state["kmeans_features"]
    if len(chosen) < 2:
        st.info("Select at least 2 features and apply settings to view clustering.")
    elif len(df) < 3:
        st.warning("At least 3 rows are required to evaluate K-Means clusters.")
    elif not st.session_state.get("kmeans_run", False):
        st.info("Choose features and the number of clusters, then apply settings to run the analysis.")
    else:
        Xs = StandardScaler().fit_transform(df[chosen])
        ks, inertia, sil = cluster_diagnostics(Xs, seed)
        kc = st.session_state["kmeans_clusters"]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(ks, inertia, marker="o", color="seagreen")
            ax.axvline(kc, color="gray", linestyle="--", label=f"Selected k = {kc}")
            ax.set_title("Elbow method (inertia)")
            ax.set_xlabel("k")
            ax.set_ylabel("Inertia (lower is tighter)")
            ax.set_xticks(ks)
            ax.grid(alpha=0.25)
            ax.legend()
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(ks, sil, marker="o", color="orange")
            ax.axvline(kc, color="gray", linestyle="--", label=f"Selected k = {kc}")
            ax.set_title("Silhouette score")
            ax.set_xlabel("k")
            ax.set_ylabel("Score (higher is better)")
            ax.set_xticks(ks)
            ax.grid(alpha=0.25)
            ax.legend()
            st.pyplot(fig)
        st.caption(
            "Use the elbow chart to look for diminishing returns as k increases; "
            "a higher silhouette score generally indicates better-separated clusters. "
            "These are guides, not an automatic guarantee of the best k."
        )

        cluster_labels = fit_clusters(Xs, kc, seed)
        out = df.copy()
        out["cluster"] = cluster_labels
        pca = PCA(n_components=2)
        pts = pca.fit_transform(Xs)
        cluster_col, size_col = st.columns(2)
        with cluster_col:
            fig, ax = plt.subplots(figsize=(7, 4))
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=cluster_labels, cmap="viridis", s=14, alpha=0.75)
            ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.set_title("Cluster map (PCA projection)")
            ax.legend(*sc.legend_elements(), title="Cluster")
            ax.grid(alpha=0.2)
            st.pyplot(fig)
        with size_col:
            cluster_sizes = out["cluster"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.bar(cluster_sizes.index.astype(str), cluster_sizes.values, color="seagreen")
            ax.bar_label(bars, padding=3)
            ax.set_title("Number of records in each cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Records")
            ax.grid(axis="y", alpha=0.25)
            st.pyplot(fig)
        st.caption(
            "The cluster map uses PCA to compress the selected features into two dimensions "
            "for visualization; points that look close on this chart may differ in the full feature space."
        )

        st.markdown("**Cluster profile (average values)**")
        cluster_profiles = out.groupby("cluster")[FEATURES].mean()
        st.dataframe(
            cluster_profiles.round(1).rename(columns=FEATURE_LABELS),
            width="stretch",
        )
        st.markdown("**Cluster profile heatmap (standard deviations from dataset average)**")
        profile_std = df[FEATURES].std().replace(0, 1)
        profile_z = cluster_profiles.sub(df[FEATURES].mean()).div(profile_std)
        fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * kc + 1)))
        image = ax.imshow(profile_z, cmap="RdYlBu_r", aspect="auto", vmin=-2, vmax=2)
        ax.set_xticks(range(len(FEATURES)))
        ax.set_xticklabels(
            [FEATURE_LABELS[feature] for feature in FEATURES],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(range(kc), [f"Cluster {cluster}" for cluster in profile_z.index])
        ax.set_title("How each cluster profile differs from the overall average")
        fig.colorbar(image, ax=ax, label="Standard deviations from overall average")
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("**Top crops in each cluster**")
        top = out.groupby("cluster")[TARGET].agg(lambda s: ", ".join(s.value_counts().head(4).index))
        st.dataframe(top.rename("Most common crops"), width="stretch")

# --------------------------- Crop Advisor tab -------------------------
with tabs[2]:
    st.subheader("Enter your field conditions")
    st.write(
        "The smart suggestion compares your inputs with the dataset's average "
        "conditions for each crop and ranks the closest crop profiles."
    )
    with st.form("crop_advisor_inputs"):
        cols = st.columns(4)
        vals = {}
        for i, f in enumerate(FEATURES):
            lo, hi, mean = float(df[f].min()), float(df[f].max()), float(df[f].mean())
            vals[f] = cols[i % 4].slider(
                FEATURE_LABELS[f],
                lo,
                hi,
                round(mean, 1),
                key=f"advisor_{f}",
            )
        recommend = st.form_submit_button("🌱 Recommend crop", type="primary")
    if recommend:
        st.session_state["advisor_values"] = vals
        st.session_state["advisor_recommendation_ready"] = True
    vals = st.session_state.get("advisor_values", vals)
    sample = pd.DataFrame([vals])

    st.subheader("Extreme input alerts")
    st.caption(
        "An input is flagged when it falls outside the dataset's 5th–95th percentile "
        "range. Soil pH below 5.5 is also flagged as too acidic; above 7.5 is too basic."
    )
    alerts = extreme_input_alerts(df, vals)
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("No extreme input values detected.")

    if st.session_state.get("advisor_recommendation_ready", False):
        suggestions, feature_std = crop_profile_suggestions(df, sample)
        best = suggestions.iloc[0]
        best_crop = best["Crop"]
        crop_profile = df.groupby(TARGET)[FEATURES].mean().loc[best_crop]
        differences = (sample.iloc[0] - crop_profile).abs().div(feature_std).sort_values()
        closest_factors = ", ".join(FEATURE_LABELS[feature] for feature in differences.head(3).index)

        st.markdown("### Smart suggestion overview")
        summary, score = st.columns([3, 1])
        summary.success(f"Suggested crop: **{best_crop}**")
        score.metric("Profile match", f"{best['Profile match']:.1f}/100")
        st.caption(
            "The profile match is a relative fit score based on average standardized "
            "differences, not a probability or guaranteed yield estimate."
        )
        st.write(f"Closest-matching conditions: **{closest_factors}**.")
        st.markdown("**Other crop profile matches**")
        st.dataframe(
            suggestions.head(5)[["Crop", "Profile match"]].style.format(
                {"Profile match": "{:.1f}/100"}
            ),
            width="stretch",
            hide_index=True,
        )

        # Keep K-Means as the separate unsupervised farming-zone overview.
        sc_all = StandardScaler().fit(df[FEATURES])
        km_final = KMeans(n_clusters=4, n_init=10, random_state=seed).fit(sc_all.transform(df[FEATURES]))
        zone = int(km_final.predict(sc_all.transform(sample))[0])
        st.info(f"Farming zone: **Cluster {zone}** (K-Means)")

        st.markdown(f"**Crops commonly grown in Cluster {zone}**")
        tmp = df.copy()
        tmp["cluster"] = km_final.labels_
        st.write(", ".join(tmp[tmp.cluster == zone][TARGET].value_counts().head(5).index))
