import pyreadr
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.utils import to_time_series_dataset
import seaborn as sns
import matplotlib.cm as cm
# Load the RData file
result = pyreadr.read_r('./data/portarthur_sd_df_2019.rdata')

# Extract the dataframe (usually only one object in the file)
df = list(result.values())[0]

# Show basic info
print(df.columns)
print(df.head())
# Filter rows where source and destination counties are the same
bp_df = df[df['from_cnt'] == df['to_cnt']].copy()

# Confirm structure
print(bp_df[['origin_census_block_group', 'destination_cbg', 'device_count', 
             'destination_device_count', 'year', 'uid']].head())
# Convert UID (day of year) to datetime
bp_df['date'] = pd.to_datetime(bp_df['uid'], format='%j', errors='coerce')
bp_df['month'] = bp_df['date'].dt.month
# Group by destination CBG and month, sum the incoming device counts (visits)
monthly_centrality = (
    bp_df.groupby(['destination_cbg', 'month'])['destination_device_count']
    .sum()
    .reset_index()
    .rename(columns={'destination_cbg': 'cbg', 'destination_device_count': 'degree_centrality'})
)

# Sort for visibility
monthly_centrality = monthly_centrality.sort_values(by=['cbg', 'month'])

# Save to CSV
monthly_centrality.to_csv('./outputs/monthly_centrality.csv', index=False)
print(monthly_centrality.head())
bp_df['destination_device_count'] = pd.to_numeric(bp_df['destination_device_count'], errors='coerce')
bp_df = bp_df.dropna(subset=['destination_device_count'])

# Create a directed graph
G = nx.DiGraph()

# Iterate over each row and add edge with weight
for _, row in bp_df.iterrows():
    origin = row['origin_census_block_group']
    destination = row['destination_cbg']
    weight = row['destination_device_count']

    if G.has_edge(origin, destination):
        G[origin][destination]['weight'] += weight
    else:
        G.add_edge(origin, destination, weight=weight)
# You can add attributes like total devices at origin if needed
device_counts = bp_df.groupby('origin_census_block_group')['device_count'].first()

# Add device count as a node attribute
for node, count in device_counts.items():
    if G.has_node(node):
        G.nodes[node]['device_count'] = count

# Select a small subgraph
subgraph = G.subgraph(list(G.nodes)[:30])
pos = nx.spring_layout(subgraph, seed=42)

# Extract weights
weights = [subgraph[u][v]['weight'] for u, v in subgraph.edges()]

# Normalize weights between 0 and 1 for edge_color
min_w = min(weights)
max_w = max(weights)
norm_weights = [(w - min_w) / (max_w - min_w + 1e-5) for w in weights]

# Draw with normalized colors
nx.draw(
    subgraph,
    pos,
    with_labels=True,
    node_size=500,
    font_size=6,
    edge_color=norm_weights,
    edge_cmap=plt.cm.Blues
)

plt.title("Sample Mobility Subgraph (CBG-to-CBG)")
# Save to file before showing
plt.savefig('./outputs/sample_mobility_subgraph.png', dpi=300, bbox_inches='tight')
plt.show()
# Group by destination CBG and month, summing device visits (in-degree centrality)
monthly_indegree = (
    bp_df.groupby(['destination_cbg', 'month'])['destination_device_count']
    .sum()
    .reset_index()
    .rename(columns={
        'destination_cbg': 'cbg',
        'destination_device_count': 'in_degree_centrality'
    })
)
# Normalize in-degree centrality per CBG
normalized_df = (
    monthly_indegree.groupby('cbg')
    .apply(lambda group: group.assign(
        norm_in_degree_centrality=MinMaxScaler().fit_transform(group[['in_degree_centrality']])
    ))
    .reset_index(drop=True)
)

# Select a few example CBGs 

# Select a few example CBGs (you can change these to wildfire-affected areas)

sample_cbgs = normalized_df['cbg'].value_counts().index[:4]

for cbg in sample_cbgs:
    subset = normalized_df[normalized_df['cbg'] == cbg]
    plt.plot(subset['month'], subset['norm_in_degree_centrality'], label=f'CBG {cbg}')

plt.xlabel("Month")
plt.ylabel("Normalized In-Degree Centrality")
plt.title("Mobility Recovery Pattern (In-Degree Centrality)")
plt.legend()
plt.grid(True)

# 💾 Save the figure
plt.savefig('./outputs/mobility_recovery_pattern.png', dpi=300, bbox_inches='tight')

plt.show()
normalized_df = normalized_df.rename(columns={'norm_in_degree_centrality': 'centrality'})
three_months_df = normalized_df[normalized_df['month'].isin([8, 9, 10])].copy()
def compute_resilience_triangle_window(df):
    try:
        # Extract degree centrality for each time point
        t0 = 8  # August (pre-disaster)
        tD = 9  # September (during disaster)
        t1 = 10 # October (post-disaster)

        deg_t0 = df[df['month'] == t0]['centrality'].values[0]
        deg_tD = df[df['month'] == tD]['centrality'].values[0]
        deg_t1 = df[df['month'] == t1]['centrality'].values[0]

        # Compute slopes per the paper's definitions
        vulnerability = (deg_tD - deg_t0) / (tD - t0) if (tD - t0) != 0 else float('nan')
        robustness = (deg_t1 - deg_tD) / (t1 - tD) if (t1 - tD) != 0 else float('nan')

        # Additional optional metrics
        drop_depth = deg_t0 - deg_tD
        recovery_rate = deg_t1 - deg_tD
        area_loss = 0.5 * drop_depth * (t1 - t0)

        return {
            'cbg': df['cbg'].iloc[0],
            'deg_t0': deg_t0,
            'deg_tD': deg_tD,
            'deg_t1': deg_t1,
            'vulnerability': vulnerability,
            'robustness': robustness,
            'drop_depth': drop_depth,
            'recovery_rate': recovery_rate,
            'area_loss': area_loss
        }

    except Exception as e:
        print(f"Error processing CBG {df['cbg'].iloc[0]}: {e}")
        return None

# Run the triangle computation for each CBG
results = []
for cbg, group in three_months_df.groupby('cbg'):
    result = compute_resilience_triangle_window(group)
    if result:
        results.append(result)

resilience_df = pd.DataFrame(results)
resilience_df.to_csv('./outputs/resilience_triangle_metrics.csv', index=False)
resilience_df.head(50)


# Pick a single CBG to visualize
example_cbg = resilience_df['cbg'].iloc[37]
cbg_df = three_months_df[three_months_df['cbg'] == example_cbg].sort_values('month')

# Extract values
t0, tD, t1 = 8, 9, 10
months = cbg_df['month'].values
centrality = cbg_df['centrality'].values

# Extract resilience metrics for the triangle
res_row = resilience_df[resilience_df['cbg'] == example_cbg].iloc[0]
deg_t0, deg_tD, deg_t1 = res_row['deg_t0'], res_row['deg_tD'], res_row['deg_t1']

# Triangle coordinates
triangle_x = [t0, tD, t1]
triangle_y = [deg_t0, deg_tD, deg_t1]

# Plotting
plt.figure(figsize=(8, 6))

# Plot actual mobility recovery line (blue)
plt.plot(months, centrality, '-o', label='Actual Centrality (norm)', color='blue')

# Plot resilience triangle (red)
plt.plot(triangle_x, triangle_y, '-o', color='red', label='Resilience Triangle')

# Dashed green baseline
plt.axhline(y=deg_t0, linestyle='--', color='green', label='Baseline (Pre-disaster)')

# Label disaster time points
plt.xticks([t0, tD, t1], ['t0\n(Aug)', 'tD\n(Sep)', 't1\n(Oct)'])
plt.xlabel('Month')
plt.ylabel('Normalized In-Degree Centrality')
plt.title(f'Resilience Triangle for CBG {example_cbg}')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('./outputs/resilience_triangle_plot_cbg_48245001011.png', dpi=300)

plt.show()


# Assume normalized_df has columns: 'cbg', 'month', 'centrality'
full_series_df = normalized_df.pivot(index='cbg', columns='month', values='centrality')
full_series_df = full_series_df.dropna()

cbg_ids = full_series_df.index.tolist()
X = to_time_series_dataset(full_series_df.values)
scaler = TimeSeriesScalerMinMax()
X_scaled = scaler.fit_transform(X)

n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=0)
labels = model.fit_predict(X_scaled)

plot_df = full_series_df.copy()
plot_df['cluster'] = labels
plot_df['cbg'] = plot_df.index  # ensure 'cbg' is column not index

# Melt to long format
melted_df = pd.melt(plot_df, id_vars=['cbg', 'cluster'], var_name='month', value_name='centrality')
melted_df['month'] = melted_df['month'].astype(int)

# Optional: month name mapping (won't be used for plotting now)
melted_df['month_name'] = melted_df['month'].map({
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
})

# This is your custom mapping if defined
try:
    melted_df['month'] = melted_df['month'].map(month_map)
except:
    pass  # ignore if month_map isn't defined

# Compute average per month per cluster
cluster_avg_df = melted_df.groupby(['cluster', 'month'])['centrality'].mean().reset_index()

# Plot: Only 1 line per cluster
plt.figure(figsize=(14, 7))
sns.set(style="whitegrid")

palette = cm.get_cmap("tab10", n_clusters)

for cluster_id in range(n_clusters):
    cluster_data = cluster_avg_df[cluster_avg_df['cluster'] == cluster_id]
    plt.plot(
        cluster_data['month'],
        cluster_data['centrality'],
        color=palette(cluster_id),
        linewidth=3,
        label=f"Cluster {cluster_id}"
    )

plt.title("DTW Clustering of CBGs Over Full Year Time Series", fontsize=16)
plt.xlabel("Month")
plt.ylabel("Normalized In-Degree Centrality")
plt.legend(title="Cluster", loc='upper right')
plt.xticks(ticks=range(1, 13), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
plt.tight_layout()
plt.grid(True)
# Save the figure
plt.savefig('./outputs/dtw_clustering_cbg_timeseries.png', dpi=300)

plt.show()

# Sample up to 15 CBGs per cluster to avoid overplotting
sampled_cbgs = (
    melted_df.groupby('cluster')['cbg']
    .apply(lambda x: x.drop_duplicates().sample(n=min(15, len(x.unique())), random_state=42))
    .reset_index(drop=True)
)

# Filter melted_df to only include sampled CBGs
filtered_melted_df = melted_df[melted_df['cbg'].isin(sampled_cbgs)]

# Initialize plot
plt.figure(figsize=(12, 7))
sns.set(style="whitegrid")

# Use consistent colormap
palette = cm.get_cmap("tab10", n_clusters)

# Plot individual CBG lines (faded)
for cluster_id in range(n_clusters):
    cluster_cbgs = filtered_melted_df[melted_df['cluster'] == cluster_id]
    for cbg_id in cluster_cbgs['cbg'].unique():
        cbg_data = cluster_cbgs[cluster_cbgs['cbg'] == cbg_id]
        plt.plot(
            cbg_data['month'],
            cbg_data['centrality'],
            color=palette(cluster_id),
            linewidth=1,
            alpha=0.25
        )

# Overlay bold average line for each cluster
for cluster_id in range(n_clusters):
    cluster_avg = cluster_avg_df[cluster_avg_df['cluster'] == cluster_id]
    plt.plot(
        cluster_avg['month'],
        cluster_avg['centrality'],
        color=palette(cluster_id),
        linewidth=3,
        label=f'Cluster {cluster_id }'
    )

# Axis formatting
plt.xlabel("Month", fontsize=12)
plt.ylabel("Normalized In-Degree Centrality", fontsize=12)
plt.title("Clustered Mobility Recovery Patterns (Paper 2 Style)", fontsize=14)
plt.xticks(
    ticks=range(1, 13),
    labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    rotation=45
)
plt.legend(title="Cluster", loc='upper right')
plt.grid(True)
plt.tight_layout()


# Save the plot
plt.savefig('./outputs/clustered_mobility_recovery.png', dpi=300)

plt.show()
# Add cluster info to resilience_df
plot_df = plot_df.reset_index(drop=True)  # removes 'cbg' from index, makes it just a column

resilience_df = resilience_df.merge(plot_df[['cbg', 'cluster']], on='cbg', how='left')

# Compute total baseline area and resilience per CBG
resilience_df['total_area'] = (10 - 8) * resilience_df['deg_t0']  # (t1 - t0) * deg_t0
resilience_df['resilience'] = np.where(
    resilience_df['total_area'] != 0,
    1 - (resilience_df['area_loss'] / resilience_df['total_area']),
    np.nan  # or 0 if you prefer to assume no resilience
)
# Compute cluster-level stats
cluster_summary = resilience_df.groupby('cluster').agg(
    loss_of_resilience=('area_loss', 'mean'),
    resilience_ratio=('resilience', lambda x: np.mean(x) * 100)
).reset_index()

# Format for display like paper
cluster_summary['resilience_ratio'] = cluster_summary['resilience_ratio'].map("{:.2f}%".format)
cluster_summary['loss_of_resilience'] = cluster_summary['loss_of_resilience'].round(4)
cluster_summary.to_csv('./outputs/cluster_resilience_summary.csv', index=False)

cluster_summary.head()
def run():
    print("Running Paper 2 resilience triangle")

cluster_summary.head(50)
def run():
    print("Running Paper 1 RCN analysis...")

    # Your actual logic goes here
    # Example:
    # load data, compute RCN, save plots, etc.
