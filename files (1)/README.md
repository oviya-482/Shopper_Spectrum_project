# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

## Project Structure
```
project/
├── shopper_spectrum_notebook.py   # Full analysis script (run first)
├── streamlit_app.py               # Interactive web app
├── requirements.txt               # Dependencies
├── online_retail.csv              # Dataset (place here)
├── models/                        # Saved models (auto-generated)
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   ├── label_map.pkl
│   └── similarity_matrix.pkl
└── *.png                          # EDA & clustering plots (auto-generated)
```

## Setup & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Step 1: Run the analysis notebook (generates models + plots)
python shopper_spectrum_notebook.py

# Step 2: Launch the Streamlit app
streamlit run streamlit_app.py
```

## Features
- **EDA**: Country analysis, top products, monthly revenue, RFM distributions
- **RFM Segmentation**: KMeans clustering → High-Value, Regular, Occasional, At-Risk
- **Product Recommendations**: Item-based collaborative filtering (cosine similarity)
- **Streamlit App**: Interactive UI for real-time predictions & recommendations
