"""
Smart Crop Advisor - ML mini project
Analysis: crop-profile matching and K-Means
UI: Streamlit
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------
# >>> PASTE YOUR DATASET FILE ADDRESS HERE <<<
# Example (Windows): r"C:\Users\you\Downloads\Crop_recommendation.csv"
# For deployment keep the CSV next to app.py and leave it as below.
CSV_PATH = "Crop_recommendation.csv"
# ----------------------------------------------------------------------

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

st.set_page_config(page_title="Smart Crop Advisor", page_icon="🌾", layout="wide")


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


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


# ---------------------------- Sidebar ---------------------------------
st.sidebar.title("🌾 Smart Crop Advisor")
uploaded = st.sidebar.file_uploader("Or upload the CSV here", type="csv")
try:
    df = pd.read_csv(uploaded) if uploaded else load_data(CSV_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found at '{CSV_PATH}'. Set CSV_PATH in app.py or upload the file in the sidebar.")
    st.stop()

seed = st.sidebar.number_input("Random seed", 0, 999, 42)
classes = sorted(df[TARGET].unique())

st.title("🌾 Smart Crop Advisor")
st.caption("Explore crop data, match field conditions to crop profiles, and discover farming zones.")

tabs = st.tabs(["📊 Data", "🧩 K-Means", "🔮 Crop Advisor"])

# ----------------------------- Data tab -------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Features", len(FEATURES))
    c3.metric("Crops", len(classes))
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("Feature distribution")
    feat = st.selectbox("Feature", FEATURES)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(df[feat], bins=30, color="seagreen")
    ax.set_xlabel(feat)
    st.pyplot(fig)
    st.subheader("Average conditions per crop")
    st.dataframe(df.groupby(TARGET)[FEATURES].mean().round(1), use_container_width=True)

# ---------------------------- K-Means tab -----------------------------
with tabs[1]:
    st.subheader("K-Means: discover natural farming zones")
    st.caption("Unsupervised: the crop label is NOT used for clustering.")
    chosen = st.multiselect("Features to cluster on", FEATURES, default=FEATURES)
    if len(chosen) < 2:
        st.warning("Select at least 2 features.")
    else:
        Xs = StandardScaler().fit_transform(df[chosen])
        ks = range(2, 11)
        inertia, sil = [], []
        for i in ks:
            km_i = KMeans(n_clusters=i, n_init=10, random_state=seed).fit(Xs)
            inertia.append(km_i.inertia_)
            sil.append(silhouette_score(Xs, km_i.labels_))
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(ks, inertia, marker="o")
            ax.set_title("Elbow method (inertia)")
            ax.set_xlabel("k")
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.plot(ks, sil, marker="o", color="orange")
            ax.set_title("Silhouette score")
            ax.set_xlabel("k")
            st.pyplot(fig)

        kc = st.slider("Number of clusters", 2, 10, 4)
        km = KMeans(n_clusters=kc, n_init=10, random_state=seed).fit(Xs)
        out = df.copy()
        out["cluster"] = km.labels_
        pts = PCA(n_components=2).fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(7, 4))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=km.labels_, cmap="viridis", s=10)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.legend(*sc.legend_elements(), title="Cluster")
        st.pyplot(fig)

        st.markdown("**Cluster profile (average values)**")
        st.dataframe(out.groupby("cluster")[FEATURES].mean().round(1), use_container_width=True)
        st.markdown("**Top crops in each cluster**")
        top = out.groupby("cluster")[TARGET].agg(lambda s: ", ".join(s.value_counts().head(4).index))
        st.dataframe(top.rename("Most common crops"), use_container_width=True)

# --------------------------- Crop Advisor tab -------------------------
with tabs[2]:
    st.subheader("Enter your field conditions")
    st.write(
        "The smart suggestion compares your inputs with the dataset's average "
        "conditions for each crop and ranks the closest crop profiles."
    )
    cols = st.columns(4)
    vals = {}
    for i, f in enumerate(FEATURES):
        lo, hi, mean = float(df[f].min()), float(df[f].max()), float(df[f].mean())
        vals[f] = cols[i % 4].slider(f, lo, hi, round(mean, 1))
    sample = pd.DataFrame([vals])

    if st.button("🌱 Recommend crop", type="primary"):
        suggestions, feature_std = crop_profile_suggestions(df, sample)
        best = suggestions.iloc[0]
        best_crop = best["Crop"]
        crop_profile = df.groupby(TARGET)[FEATURES].mean().loc[best_crop]
        differences = (sample.iloc[0] - crop_profile).abs().div(feature_std).sort_values()
        closest_factors = ", ".join(differences.head(3).index)

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
            use_container_width=True,
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
