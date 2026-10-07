# Smart Crop Advisor

A Streamlit app that explores crop data, suggests crops by comparing field conditions with crop profiles, and groups fields into farming zones.

## Features

- **Data:** preview the dataset, inspect feature distributions, and compare average conditions by crop.
- **K-Means:** explore unsupervised farming zones with elbow and silhouette charts, a PCA cluster plot, and cluster profiles.
- **Crop Advisor:** enter seven soil and weather measurements to see a ranked crop-profile match overview, the closest-matching conditions, and the field's K-Means zone.
- **Apply-on-submit controls:** adjust crop inputs, K-Means settings, feature distributions, and dataset settings in forms; calculations update when you submit each form instead of on every adjustment.

The crop suggestion is based on the average feature profile for each crop. It standardizes differences across features, ranks crops by their mean absolute standardized difference, and displays a relative profile-match score. This score is not a probability, yield prediction, or guarantee of suitability.

## Dataset

| Item | Details |
|---|---|
| Name | Karnataka Crop Dataset for ML models |
| Source | [Kaggle](https://www.kaggle.com/datasets/rmeghaganesh/karnataka-crop-dataset-for-ml-models) |
| File | `data/karnataka_crop_dataset_V2.csv` |
| Size | 21,960 rows × 12 columns, 39 crops |

This dataset is described as semi-synthetic: records are generated using ranges informed by cited government and research sources, rather than being direct field measurements. It is specific to Karnataka. The app uses the seven numeric soil and weather features; additional columns are retained for display and crop summaries.

The app labels input features with these units:

| Feature | Unit |
|---|---|
| Nitrogen (N), Phosphorus (P), Potassium (K) | kg/ha |
| Temperature | °C |
| Humidity | % |
| Soil pH | Unitless |
| Rainfall | mm |

The `label` column contains the crop name.

The Crop Advisor flags inputs below the dataset's 5th percentile or above its 95th percentile. It also flags soil pH below 5.5 as too acidic and above 7.5 as too basic.

## Project files

```text
crop-advisor/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── data/
│   └── karnataka_crop_dataset_V2.csv
├── README.md
└── WORKING_EXPLAINED.md      # Code walkthrough
```

## Run locally

Place `karnataka_crop_dataset_V2.csv` in the `data` folder, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Alternatively, upload a CSV in the sidebar. The app checks for the seven expected numeric input columns and the `label` column.

## Deploy

Streamlit Community Cloud supports this app:

1. Upload the project files, including `data/karnataka_crop_dataset_V2.csv`, to a GitHub repository.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io), select the repository and `app.py`, and deploy.

## Technology

Python · Streamlit · pandas · scikit-learn · Matplotlib

## Future improvements

- Add yield and market-price information.
- Support live weather and soil-test inputs.
- Add Marathi and Hindi language options.
