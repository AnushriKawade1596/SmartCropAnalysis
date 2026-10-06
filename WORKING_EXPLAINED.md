# How the Code Works

This document explains the data flow, crop suggestions, and farming-zone analysis in `app.py`.

## 1. Application setup

- `streamlit` creates the interactive web interface.
- `pandas` loads and summarizes the crop dataset.
- scikit-learn provides feature scaling, K-Means clustering, PCA, and silhouette scoring.
- `FEATURES` lists the seven soil and weather inputs; `TARGET` is the dataset's `label` column.
- `CSV_PATH` points to the dataset file. The sidebar also accepts a CSV upload.
- `load_data()` is cached so the file does not need to be reread on every UI interaction.

## 2. Data tab

The Data tab reports the row, feature, and crop counts; previews the dataset; plots the distribution of a selected feature; and displays average feature values for each crop.

## 3. K-Means tab

K-Means groups fields using the selected input features; it does not use the crop label to create clusters.

- `StandardScaler` brings features onto comparable scales before distance-based clustering.
- The elbow chart displays cluster inertia for several cluster counts.
- The silhouette chart shows how well-separated the clusters are.
- PCA reduces the selected features to two dimensions for a scatter plot.
- The app also displays average feature profiles and common crop labels for each cluster.

## 4. Crop Advisor tab

The user sets values for nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall. On recommendation:

1. `crop_profile_suggestions()` calculates the average input profile for every crop.
2. It standardizes the absolute difference between the user's values and each crop profile using the dataset's feature standard deviations.
3. It ranks the crops by their mean absolute standardized difference. The closest profile is the suggested crop.
4. The displayed profile-match score is `100 / (1 + mean difference)`. It is a relative fit score, **not** a probability or a yield estimate.
5. The Advisor separately applies K-Means to show the field's farming zone and the crops commonly found in that zone.

The crop profile is an easy-to-explain comparison against historical averages. It does not account for local farming practices, season, water availability, pest pressure, or market conditions, so users should treat it as guidance rather than a guarantee.

## 5. Common problems

| Problem | Fix |
|---|---|
| Dataset not found | Check `CSV_PATH` or upload the CSV in the sidebar. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt`. |
| Deployed app cannot find the dataset | Include `Crop_recommendation.csv` beside `app.py` in the repository. |
