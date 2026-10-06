# Smart Crop Advisor

A Streamlit app that explores crop data, suggests crops by comparing field conditions with crop profiles, and groups fields into farming zones.

## Features

- **Data:** preview the dataset, inspect feature distributions, and compare average conditions by crop.
- **K-Means:** explore unsupervised farming zones with elbow and silhouette charts, a PCA cluster plot, and cluster profiles.
- **Crop Advisor:** enter seven soil and weather measurements to see a ranked crop-profile match overview, the closest-matching conditions, and the field's K-Means zone.

The crop suggestion is based on the average feature profile for each crop. It standardizes differences across features, ranks crops by their mean absolute standardized difference, and displays a relative profile-match score. This score is not a probability, yield prediction, or guarantee of suitability.

## Dataset

| Item | Details |
|---|---|
| Name | Crop Recommendation Dataset |
| Source | [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) |
| File | `Crop_recommendation.csv` |
| Size | 2,200 rows × 8 columns, 22 crops |

The input columns are `N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`. The `label` column contains the crop name.

## Project files

```text
crop-advisor/
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── Crop_recommendation.csv   # Crop recommendation dataset
├── README.md
└── WORKING_EXPLAINED.md      # Code walkthrough
```

## Run locally

Place `Crop_recommendation.csv` next to `app.py`, then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Alternatively, set `CSV_PATH` near the top of `app.py` to the dataset's location or upload a CSV in the sidebar.

## Deploy

Streamlit Community Cloud supports this app:

1. Upload the project files, including `Crop_recommendation.csv`, to a GitHub repository.
2. Create a new app at [share.streamlit.io](https://share.streamlit.io), select the repository and `app.py`, and deploy.

## Technology

Python · Streamlit · pandas · scikit-learn · Matplotlib

## Future improvements

- Add yield and market-price information.
- Support live weather and soil-test inputs.
- Add Marathi and Hindi language options.
