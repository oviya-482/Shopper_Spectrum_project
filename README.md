# 🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations for E-Commerce

A customer analytics project built on e-commerce retail transaction data. It cleans raw transactions, segments customers using **RFM analysis + KMeans clustering**, recommends products using **item-based collaborative filtering (cosine similarity)**, and serves both models through an interactive **Streamlit** web app.

> Developed as part of a Business Analytics internship at **Labmentix** (01 May 2026 – 30 June 2026) for the MBA programme at RVS College of Arts and Science (Autonomous), Bharathiar University.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Key Results](#-key-results)
- [Streamlit App](#-streamlit-app)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Business Recommendations](#-business-recommendations)
- [Limitations](#-limitations)
- [Future Scope](#-future-scope)
- [References](#-references)
- [Author](#-author)

---

## 🔍 Overview

Every purchase leaves a trace of customer intent. This project combines those traces to answer two practical questions for an online retailer:

1. **Who should we target?** – Customer segmentation groups shoppers by Recency, Frequency and Monetary value.
2. **What should we offer?** – A recommender suggests products bought by similar sets of customers.

The project covers the full analytics life cycle: data cleaning → EDA → feature engineering → modelling → model persistence → deployment.

## 🎯 Objectives

- Clean and prepare a large e-commerce transaction dataset for reliable analysis.
- Explore purchasing patterns across countries, products, time periods and spending levels.
- Engineer **Recency, Frequency, Monetary (RFM)** features for every customer.
- Segment customers into four business-friendly groups: **High-Value, Regular, Occasional, At-Risk**.
- Build an **item-based collaborative filtering** engine that suggests similar products.
- Deliver both models through an interactive **Streamlit** application.

## 📊 Dataset

**UCI Online Retail dataset** (`online_retail.csv`) – [link](https://archive.ics.uci.edu/dataset/352/online+retail)

| Column | Description |
|---|---|
| `InvoiceNo` | Invoice number (prefix `C` = cancellation) |
| `StockCode` | Unique product code |
| `Description` | Product name |
| `Quantity` | Units purchased in the line |
| `InvoiceDate` | Date and time of the transaction |
| `UnitPrice` | Price per unit |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |

- **Raw size:** 541,909 rows × 8 columns
- **After cleaning:** 392,692 valid purchase lines · 4,338 customers · 3,639 products · 38 countries

### Data cleaning steps

| Step | Rows remaining |
|---|---|
| Raw data | 541,909 |
| Drop missing `CustomerID` | 406,829 |
| Exclude cancellations (`InvoiceNo` starts with `C`) | 397,924 |
| Remove non-positive quantity / price | 397,884 |
| Drop missing description | 397,884 |
| Remove exact duplicates | 392,692 |

Additionally, the most frequent description for each `StockCode` was used as its standard product name, and a `TotalPrice = Quantity × UnitPrice` column was added.

## 🛠 Tech Stack

| Tool / Library | Purpose |
|---|---|
| Python (Jupyter Notebook) | Cleaning, analysis and modelling |
| pandas, NumPy | Data handling, aggregation, pivot tables |
| Matplotlib, Seaborn | Exploratory charts and cluster visualisations |
| scikit-learn | `StandardScaler`, `KMeans`, silhouette score, cosine similarity |
| pickle | Saving trained models for the app |
| Streamlit | Interactive web application |

## 🔄 Project Workflow

1. **Data preparation** – load with `latin1` encoding, inspect quality, clean.
2. **Exploratory analysis** – countries, top products, monthly revenue, spend distributions.
3. **RFM feature engineering** – computed per customer relative to a snapshot date (last transaction date + 1 day).
   - **Recency:** days since last purchase
   - **Frequency:** number of distinct invoices
   - **Monetary:** total amount spent
4. **Scaling** – `StandardScaler` so all three measures contribute equally to KMeans distances.
5. **Clustering** – KMeans with k chosen via the elbow method and silhouette score (k = 2 to 10); **k = 4** selected.
6. **Recommendation** – Customer × Product quantity matrix (4,338 × 3,639) → transposed → cosine similarity → 3,639 × 3,639 similarity matrix.
7. **Deployment** – models saved as pickle files and loaded into a two-tab Streamlit app.

```python
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('TotalPrice', 'sum')
).reset_index()
```

```python
def recommend_products(product_name, top_n=5):
    sims = similarity_df[product_name].drop(product_name)
    return list(sims.sort_values(ascending=False).head(top_n).items())
```

## 📈 Key Results

### Exploratory findings

- Sales are **heavily concentrated in the United Kingdom**; Germany, France and EIRE follow far behind.
- Revenue stays around 0.45–0.7M from December to August, then **rises sharply from September and peaks in November (>1.1M)** – consistent with pre-holiday shopping. (The final December dip is because the data ends part-way through the month.)
- Top sellers by quantity are low-priced decorative/household items (e.g., *Paper Craft Little Birdie*, *Medium Ceramic Top Storage Jar*), suggesting many bulk/wholesale buyers.
- Spending is **strongly right-skewed** – a few very large customers account for a big share of revenue.

### Customer segments (KMeans, k = 4)

Silhouette score for k = 4 is **0.6162**, essentially equal to the best value among k = 3–6 (0.6165 at k = 5).

| Segment | Customers | Avg. Recency (days) | Avg. Frequency | Avg. Monetary |
|---|---|---|---|---|
| High-Value | 13 | 7.4 | 82.5 | 127,187.96 |
| Regular | 204 | 15.5 | 22.3 | 12,690.50 |
| Occasional | 3,054 | 43.7 | 3.7 | 1,353.63 |
| At-Risk | 1,067 | 248.1 | 1.6 | 478.85 |

About **1 in 4 customers (1,067 of 4,338) is At-Risk**, a clear retention opportunity.

### Sample recommendations

Top 5 products similar to **WHITE HANGING HEART T-LIGHT HOLDER**:

| Rank | Recommended Product | Similarity |
|---|---|---|
| 1 | GIN + TONIC DIET METAL SIGN | 0.75 |
| 2 | RED HANGING HEART T-LIGHT HOLDER | 0.66 |
| 3 | WASHROOM METAL SIGN | 0.64 |
| 4 | LAUNDRY 15C METAL SIGN | 0.64 |
| 5 | GREEN VINTAGE SPOT BEAKER | 0.63 |

## 🖥 Streamlit App

The app (`app.py`) has two tabs:

| Tab | Input | Output |
|---|---|---|
| **Product Recommendations** | Product chosen from a searchable list of 3,639 products | Five most similar products as cards with similarity scores |
| **Customer Segmentation** | Recency (days), Frequency (purchases), Monetary (total spend) | Predicted segment (High-Value / Regular / Occasional / At-Risk) with a suggested marketing action |

Features: searchable product dropdown, card-style results, cached model loading (`st.cache_resource`) for fast responses, and friendly error messages if model files are missing.

> 💡 Add screenshots here, e.g. `![Product Recommendations](screenshots/recommendations.png)` and `![Customer Segmentation](screenshots/segmentation.png)`.

## 📁 Project Structure

```
Shopper_Spectrum_project/
├── data/
│   └── online_retail.csv            # UCI Online Retail dataset (download separately)
├── notebooks/
│   └── shopper_spectrum.ipynb       # Cleaning, EDA, RFM, clustering, recommender
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── cluster_label_map.pkl
│   ├── product_similarity.pkl
│   └── product_list.pkl
├── app.py                           # Streamlit application
├── requirements.txt
└── README.md
```

> Adjust the layout above to match your actual repository.

## ⚙️ Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/oviya-482/Shopper_Spectrum_project.git
cd Shopper_Spectrum_project

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from UCI and place it at data/online_retail.csv

# 5. Run the notebook to generate the model (.pkl) files

# 6. Launch the app
streamlit run app.py
```

**Suggested `requirements.txt`:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
streamlit
jupyter
```

## 💼 Business Recommendations

- **High-Value:** loyalty rewards, VIP perks, early access, personal service – retention is the top priority.
- **Regular:** bundles, cross-selling and recommendations to push them toward High-Value.
- **Occasional:** seasonal offers and email reminders to increase purchase frequency.
- **At-Risk:** win-back discounts, reminders of past purchases and feedback surveys.
- Plan stock, staffing and delivery capacity before September for the seasonal peak.
- Use the recommender for "Customers also bought" on product pages and in emails.
- Grow international markets gradually through targeted promotions.
- Refresh RFM data and retrain models regularly.

## ⚠️ Limitations

- Based on a single retailer's dataset; findings may not generalise.
- Only purchase behaviour is used (no demographics, browsing, reviews or returns).
- The recommender uses past purchase quantities only – no price, season or category.
- Segment sizes are very uneven because extreme customers form their own clusters.
- Models were evaluated with internal measures (silhouette score, sanity checks), not in a live campaign.

## 🚀 Future Scope

- Add behavioural features (average basket value, product variety, time between purchases).
- Compare KMeans with hierarchical clustering or Gaussian Mixture Models.
- Combine product similarity with category, price and season for richer recommendations.
- Connect the app to a live database for automatic updates.
- Run A/B experiments to measure real sales impact; evaluate the recommender with precision on held-out purchases.
- Split the large Occasional segment further and treat extreme outliers separately before clustering.

## 📚 References

- [Labmentix](https://www.labmentix.in/about)
- [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail)
- [Streamlit docs](https://docs.streamlit.io/) · [scikit-learn KMeans](https://scikit-learn.org/stable/modules/clustering.html#k-means) · [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity) · [pandas](https://pandas.pydata.org/docs/) · [Seaborn](https://seaborn.pydata.org/) · [NumPy](https://numpy.org/doc/stable/) · [Matplotlib](https://matplotlib.org/stable/)
- Sarwar, B., Karypis, G., Konstan, J. and Riedl, J. (2001). *Item-based collaborative filtering recommendation algorithms.* Proceedings of the 10th International Conference on World Wide Web.

## 👩‍💻 Author

**Oviya S**
MBA Business Analytics School of Management Studies – PG, RVS College of Arts and Science (Autonomous), Coimbatore
Business Analytics Intern, Labmentix

- Internship guide: Dr. S. Yamini, Associate Professor & Director (Academic), Dept. of Computer Science
- Industry mentor: Ms. Namrata, Labmentix

GitHub: [oviya-482](https://github.com/oviya-482)
App link:https://shopperspectrumproject-ovs.streamlit.app/
