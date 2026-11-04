"""
🛒 Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce
Complete Python Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from mpl_toolkits.mplot3d import Axes3D
import pickle
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams.update({"figure.dpi": 120})

# ============================================================
# STEP 1: Dataset Collection and Understanding
# ============================================================
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv('online_retail.csv', encoding='latin1')
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nUnique customers: {df['CustomerID'].nunique()}")
print(f"Unique products:  {df['Description'].nunique()}")
print(f"Unique countries: {df['Country'].nunique()}")


# ============================================================
# STEP 2: Data Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Data Preprocessing")
print("=" * 60)

df['InvoiceNo']  = df['InvoiceNo'].astype(str)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

before = len(df)

# Remove cancelled invoices (InvoiceNo starting with 'C')
df_clean = df[~df['InvoiceNo'].str.startswith('C')].copy()
print(f"Cancelled invoices removed : {before - len(df_clean)}")

# Remove missing CustomerID
df_clean = df_clean[df_clean['CustomerID'].notna()]
print(f"Missing CustomerID removed : {before - len(df_clean)} total rows dropped")

# Remove negative/zero quantities and prices
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
print(f"Clean dataset shape        : {df_clean.shape}")

df_clean['CustomerID']  = df_clean['CustomerID'].astype(int)
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
df_clean['Month']       = df_clean['InvoiceDate'].dt.to_period('M').astype(str)


# ============================================================
# STEP 3: Exploratory Data Analysis (EDA)
# All 8 EDA items from the project specification are here
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Exploratory Data Analysis (EDA)")
print("=" * 60)

# ── 3.1 Transaction Volume by Country ───────────────────────
print("\n[EDA 3.1] Transaction volume by country")
top_countries = df_clean['Country'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top_countries.index[::-1], top_countries.values[::-1], color='steelblue', alpha=0.85)
ax.set_title('Transaction Volume by Country (Top 10)', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Transactions')
for bar, val in zip(bars, top_countries.values[::-1]):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('eda_01_country_volume.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_01_country_volume.png")

# ── 3.2 Top-Selling Products ────────────────────────────────
print("\n[EDA 3.2] Top-selling products")
top_products = df_clean['Description'].value_counts().head(10)

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(top_products.index[::-1], top_products.values[::-1], color='coral', alpha=0.85)
ax.set_title('Top 10 Best-Selling Products', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Orders')
for bar, val in zip(bars, top_products.values[::-1]):
    ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('eda_02_top_products.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_02_top_products.png")

# ── 3.3 Purchase Trends Over Time ───────────────────────────
print("\n[EDA 3.3] Purchase trends over time")
monthly_txn   = df_clean.groupby('Month')['InvoiceNo'].nunique()
monthly_rev   = df_clean.groupby('Month')['TotalAmount'].sum()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax1.plot(monthly_txn.index, monthly_txn.values, marker='o', color='teal', linewidth=2)
ax1.fill_between(monthly_txn.index, monthly_txn.values, alpha=0.15, color='teal')
ax1.set_title('Monthly Purchase Trends', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Invoices')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

ax2.plot(monthly_rev.index, monthly_rev.values, marker='s', color='darkorange', linewidth=2)
ax2.fill_between(monthly_rev.index, monthly_rev.values, alpha=0.15, color='darkorange')
ax2.set_ylabel('Revenue (£)')
ax2.set_xlabel('Month')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('eda_03_purchase_trends.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_03_purchase_trends.png")

# ── 3.4 Monetary Distribution per Transaction AND per Customer
print("\n[EDA 3.4] Monetary distribution per transaction and per customer")
customer_spend = df_clean.groupby('CustomerID')['TotalAmount'].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Per transaction
df_clean['TotalAmount'].clip(upper=df_clean['TotalAmount'].quantile(0.99)).hist(
    bins=60, ax=ax1, color='purple', alpha=0.75)
ax1.set_title('Monetary Distribution — Per Transaction\n(capped at 99th percentile)', fontweight='bold')
ax1.set_xlabel('Transaction Amount (£)')
ax1.set_ylabel('Frequency')

# Per customer
customer_spend.clip(upper=customer_spend.quantile(0.99)).hist(
    bins=60, ax=ax2, color='mediumseagreen', alpha=0.75)
ax2.set_title('Monetary Distribution — Per Customer\n(capped at 99th percentile)', fontweight='bold')
ax2.set_xlabel('Total Customer Spend (£)')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('eda_04_monetary_distributions.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_04_monetary_distributions.png")

# ── 3.5 RFM Distributions ───────────────────────────────────
print("\n[EDA 3.5] RFM distributions")

# Build RFM (needed for EDA plots — also reused in Step 4)
snapshot_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df_clean.groupby('CustomerID').agg(
    Recency  = ('InvoiceDate',  lambda x: (snapshot_date - x.max()).days),
    Frequency= ('InvoiceNo',    'nunique'),
    Monetary = ('TotalAmount',  'sum')
).reset_index()

print(f"  RFM table shape: {rfm.shape}")
print(rfm.describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
colors = ['skyblue', 'orange', 'mediumseagreen']
for ax, col, color in zip(axes, ['Recency', 'Frequency', 'Monetary'], colors):
    data = rfm[col].clip(upper=rfm[col].quantile(0.99))
    ax.hist(data, bins=40, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(rfm[col].median(), color='red', linestyle='--', linewidth=1.5, label=f'Median: {rfm[col].median():.0f}')
    ax.set_title(f'{col} Distribution', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Number of Customers')
    ax.legend(fontsize=9)
plt.suptitle('RFM Feature Distributions (capped at 99th percentile)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('eda_05_rfm_distributions.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_05_rfm_distributions.png")

# ── 3.6 Elbow Curve for Cluster Selection ───────────────────
print("\n[EDA 3.6] Elbow curve for cluster selection")

scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

K_range    = range(2, 9)
inertias, sil_scores = [], []
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_tmp.fit(rfm_scaled)
    inertias.append(km_tmp.inertia_)
    sil_scores.append(silhouette_score(rfm_scaled, km_tmp.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=4, color='red', linestyle='--', linewidth=1.5, label='Chosen k = 4')
ax1.set_title('Elbow Curve (Inertia vs k)', fontweight='bold')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.legend()

ax2.plot(list(K_range), sil_scores, 'gs-', linewidth=2, markersize=8)
ax2.axvline(x=4, color='red', linestyle='--', linewidth=1.5, label='Chosen k = 4')
ax2.set_title('Silhouette Score vs k', fontweight='bold')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.legend()

plt.tight_layout()
plt.savefig('eda_06_elbow_silhouette.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_06_elbow_silhouette.png")
print(f"  Silhouette scores: { {k: round(s,3) for k, s in zip(K_range, sil_scores)} }")

# ── 3.7 Customer Cluster Profiles ───────────────────────────
print("\n[EDA 3.7] Customer cluster profiles")

# Final KMeans with k=4
km_final      = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = km_final.fit_predict(rfm_scaled)

cs = rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean()
print("  Cluster centroids (original scale):\n", cs.round(2))

# Label clusters
label_map = {}
label_map[cs['Monetary'].idxmax()] = 'High-Value'
remaining = [c for c in range(4) if c not in label_map]
label_map[cs.loc[remaining, 'Recency'].idxmax()] = 'At-Risk'
remaining = [c for c in range(4) if c not in label_map]
label_map[cs.loc[remaining, 'Frequency'].idxmax()] = 'Regular'
remaining = [c for c in range(4) if c not in label_map]
label_map[remaining[0]] = 'Occasional'

rfm['Segment'] = rfm['Cluster'].map(label_map)
print("\n  Segment distribution:\n", rfm['Segment'].value_counts())

COLORS = {'High-Value':'#2ecc71','Regular':'#3498db','Occasional':'#f39c12','At-Risk':'#e74c3c'}
SEG_ORDER = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
seg_summary = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(2)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, metric in zip(axes, ['Recency','Frequency','Monetary']):
    vals  = [seg_summary.loc[s, metric] if s in seg_summary.index else 0 for s in SEG_ORDER]
    bars  = ax.bar(SEG_ORDER, vals, color=[COLORS[s] for s in SEG_ORDER], alpha=0.87, width=0.55)
    ax.set_title(f'Average {metric} by Segment', fontweight='bold')
    ax.set_ylabel(metric)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                f'{val:,.1f}', ha='center', va='bottom', fontsize=9)
plt.suptitle('Customer Cluster Profiles (RFM Averages)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('eda_07_cluster_profiles.png', bbox_inches='tight')
plt.close()

# 3D scatter of clusters
fig = plt.figure(figsize=(10, 7))
ax  = fig.add_subplot(111, projection='3d')
for seg, grp in rfm.groupby('Segment'):
    ax.scatter(np.log1p(grp['Recency']), np.log1p(grp['Frequency']), np.log1p(grp['Monetary']),
               label=seg, alpha=0.45, s=18, color=COLORS[seg])
ax.set_xlabel('log(Recency)');  ax.set_ylabel('log(Frequency)');  ax.set_zlabel('log(Monetary)')
ax.set_title('3D Customer Segments (RFM — log scale)', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('eda_07b_cluster_3d.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_07_cluster_profiles.png & eda_07b_cluster_3d.png")

# ── 3.8 Product Recommendation Heatmap / Similarity Matrix ──
print("\n[EDA 3.8] Product recommendation heatmap / similarity matrix")

top_prod_list = df_clean['Description'].value_counts().head(500).index
df_top        = df_clean[df_clean['Description'].isin(top_prod_list)]
pivot         = df_top.pivot_table(index='CustomerID', columns='Description',
                                   values='Quantity', fill_value=0)
print(f"  Customer–Product matrix shape: {pivot.shape}")

sim_matrix = cosine_similarity(pivot.T)
sim_df     = pd.DataFrame(sim_matrix, index=pivot.columns, columns=pivot.columns)

# Heatmap — top 15 products
top15      = df_clean['Description'].value_counts().head(15).index
heat_data  = sim_df.loc[top15, top15]

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(heat_data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.4, annot_kws={'size': 7}, vmin=0, vmax=1)
ax.set_title('Product Similarity Matrix — Top 15 Products\n(Item-Based Collaborative Filtering)',
             fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0,  fontsize=8)
plt.tight_layout()
plt.savefig('eda_08_similarity_heatmap.png', bbox_inches='tight')
plt.close()
print("  → Saved: eda_08_similarity_heatmap.png")

# Demo recommendation
def recommend_products(product_name, sim_df, n=5):
    if product_name not in sim_df.index:
        matches = [p for p in sim_df.index if product_name.upper() in p.upper()]
        if not matches:
            return [f"Product '{product_name}' not found."]
        product_name = matches[0]
    return sim_df[product_name].sort_values(ascending=False).drop(product_name).head(n).index.tolist()

test_product = 'WHITE HANGING HEART T-LIGHT HOLDER'
recs = recommend_products(test_product, sim_df)
print(f"\n  Demo — Products similar to '{test_product}':")
for i, r in enumerate(recs, 1):
    print(f"    {i}. {r}")


# ============================================================
# STEP 4: Clustering Methodology (models already built above)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Clustering Summary")
print("=" * 60)
print(f"  Algorithm   : KMeans (k=4)")
print(f"  Scaler      : StandardScaler")
print(f"  Best k      : 4 (Elbow + Silhouette)")
print(f"  Inertia     : {km_final.inertia_:.2f}")
print(f"  Silhouette  : {silhouette_score(rfm_scaled, km_final.labels_):.4f}")
print(f"\n  Segment profiles:\n{seg_summary}")


# ============================================================
# STEP 5: Save Models
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Saving Models")
print("=" * 60)

import os
os.makedirs('models', exist_ok=True)
with open('models/kmeans.pkl',   'wb') as f: pickle.dump(km_final,  f)
with open('models/scaler.pkl',   'wb') as f: pickle.dump(scaler,    f)
with open('models/label_map.pkl','wb') as f: pickle.dump(label_map, f)
sim_df.to_pickle('models/similarity_matrix.pkl')

print("  ✅ models/kmeans.pkl")
print("  ✅ models/scaler.pkl")
print("  ✅ models/label_map.pkl")
print("  ✅ models/similarity_matrix.pkl")
print("\n🎉 All done! Launch the app with: streamlit run streamlit_app.py")
