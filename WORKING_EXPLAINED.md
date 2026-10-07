# How the Code Works

This document explains the data flow, crop suggestions, and farming-zone analysis in `app.py`.

## 1. Application setup

- `streamlit` creates the interactive web interface.
- `pandas` loads and summarizes the crop dataset.
- scikit-learn provides feature scaling, K-Means clustering, PCA, and silhouette scoring.
- `FEATURES` lists the seven soil and weather inputs; `TARGET` is the dataset's `label` column.
- The app looks for `data/karnataka_crop_dataset_V2.csv` beside `app.py`, then falls back to `Crop_recommendation.csv` beside `app.py`. The sidebar also accepts a CSV upload.
- The active Karnataka dataset has 21,960 rows and 12 columns. The app uses its seven numeric soil/weather inputs and crop `label`; extra columns are allowed.
- `load_data()` is cached so the file does not need to be reread on every UI interaction.
- Dataset selection and random-seed changes are grouped in a sidebar form and applied when submitted. Data-tab feature choice, K-Means settings, and crop-advisor inputs are also form-based, so changing controls does not rerun calculations until submission. Submitted K-Means settings and crop recommendations are retained across later Streamlit reruns.

## 2. Data tab

The Data tab reports the row, feature, and crop counts; previews the dataset; plots the distribution of a selected feature; and displays average feature values for each crop.

## 3. K-Means tab

K-Means groups fields using the selected input features; it does not use the crop label to create clusters.

- `StandardScaler` brings features onto comparable scales before distance-based clustering. Silhouette scores use a deterministic sample of up to 2,000 records to keep diagnostics practical on the larger dataset.
- The elbow chart displays inertia for several cluster counts, and the silhouette chart compares cluster separation. Both mark the selected cluster count; they help guide the choice of `k` but do not automatically determine the best value.
- A PCA scatter plot shows the clusters in two dimensions, with the explained variance shown on each axis. It is a projection of the selected features, so visual closeness may not represent distance in the full feature space.
- A bar chart compares the number of records in each cluster.
- A table shows average feature values per cluster, and a heatmap shows how those means differ from the dataset average in standard deviations.
- The app also displays the most common crop labels for each cluster. Crop labels are not used when fitting K-Means.

## 4. Crop Advisor tab

The user enters nitrogen, phosphorus, and potassium in kg/ha; temperature in °C; humidity in %; rainfall in mm; and soil pH (unitless).

While entering field conditions, the Advisor flags every feature below its dataset 5th percentile or above its 95th percentile. It also flags pH below 5.5 as too acidic and pH above 7.5 as too basic. These are input alerts to encourage review, not a guarantee that a crop will or will not grow.

On recommendation:

1. `crop_profile_suggestions()` calculates the average input profile for every crop.
2. It standardizes the absolute difference between the user's values and each crop profile using the dataset's feature standard deviations.
3. It ranks the crops by their mean absolute standardized difference. The closest profile is the suggested crop.
4. The displayed profile-match score is `100 / (1 + mean difference)`. It is a relative fit score, **not** a probability or a yield estimate.
5. The Advisor separately applies K-Means to show the field's farming zone and the crops commonly found in that zone.

The crop profile is an easy-to-explain comparison against historical averages. It does not account for local farming practices, season, water availability, pest pressure, or market conditions, so users should treat it as guidance rather than a guarantee.

## 5. Common problems

| Problem | Fix |
|---|---|
| Dataset not found | Put `karnataka_crop_dataset_V2.csv` in the `data` folder, put `Crop_recommendation.csv` beside `app.py`, or upload a CSV in the sidebar. |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt`. |
| Deployed app cannot find the dataset | Include `data/karnataka_crop_dataset_V2.csv` (or the fallback `Crop_recommendation.csv` beside `app.py`) in the repository. |
