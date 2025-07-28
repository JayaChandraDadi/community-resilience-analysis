import pyreadr
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
# Load the RData file
result = pyreadr.read_r('./data/portarthur_sd_df_2019.rdata')

# Extract the dataframe (assuming there's only one object inside)
df = list(result.values())[0]

# View the structure
print(df.columns)
print(df.head())
# Step 2a: Filter data where people stay in their own CBG
df_residence = df[df['origin_census_block_group'] == df['destination_cbg']].copy()

# Step 2b: Convert uid to integer (to sort and handle dates if needed)
df_residence['uid'] = df_residence['uid'].astype(int)

# Step 2c: Group by origin_cbg and uid (date), and sum the devices that stayed within the home CBG
daily_residence_activity = df_residence.groupby(
    ['origin_census_block_group', 'uid']
)['destination_device_count'].sum().reset_index()
# Rename for clarity
daily_residence_activity.columns = ['cbg', 'day', 'residential_device_count']
# View result
print(daily_residence_activity.head())
# Define baseline period
baseline_start = 260
baseline_end = 290

# Filter baseline window
baseline_df = daily_residence_activity[
    (daily_residence_activity['day'] >= baseline_start) &
    (daily_residence_activity['day'] <= baseline_end)
]

# Convert to numeric type to avoid mean error
baseline_df['residential_device_count'] = baseline_df['residential_device_count'].astype(int)

# Compute average residential device count per CBG during baseline
baseline_avg = baseline_df.groupby('cbg')['residential_device_count'].mean().reset_index()
baseline_avg.columns = ['cbg', 'baseline_residential_device_count']

# Show result
print(baseline_avg.head())
# Step 1: Merge daily activity with baseline values
resilience_df = daily_residence_activity.merge(
    baseline_avg,
    on='cbg',
    how='inner'
)

# Step 2: Calculate normalized activity A(t)
resilience_df['A_t'] = resilience_df['residential_device_count'].astype(float) / resilience_df['baseline_residential_device_count']

# Step 3: View result
print(resilience_df.head())
# Step 1: Define disaster and recovery period
disaster_start = 260
recovery_end = 290

# Step 2: Filter for recovery window
recovery_df = resilience_df[
    (resilience_df['day'] >= disaster_start) & 
    (resilience_df['day'] <= recovery_end)
].copy()

# Step 3: Calculate RCN = ∫ (1 - A(t)) dt → approximated using sum over days
recovery_df['activity_gap'] = 1 - recovery_df['A_t']  # A_equilibrium = 1

# Step 4: Integrate per CBG
rcn_df = recovery_df.groupby('cbg')['activity_gap'].sum().reset_index()
rcn_df.columns = ['cbg', 'RCN']
rcn_df.to_csv('./outputs/rcn_scores.csv', index=False)
# View RCN scores
print(rcn_df.head())
# Scale RCN values
scaler = StandardScaler()
rcn_scaled = scaler.fit_transform(rcn_df[['RCN']])

# Set number of clusters (e.g., 3)
n_clusters = 3

# Fit clustering
agg_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
rcn_df['cluster'] = agg_cluster.fit_predict(rcn_scaled)
# Merge cluster labels into daily data
merged = daily_residence_activity.merge(rcn_df, on='cbg', how='left')
# Convert to numeric (fixes your error)
merged['residential_device_count'] = pd.to_numeric(merged['residential_device_count'], errors='coerce')

# Now group and compute average safely
plot_df = merged.groupby(['cluster', 'day'])['residential_device_count'].mean().reset_index()


# Plot
plt.figure(figsize=(10,6))
sns.lineplot(data=plot_df, x='day', y='residential_device_count', hue='cluster', palette='Set2')
plt.axvspan(260, 290, color='gray', alpha=0.3, label='Disaster Window')  # Shade disaster period
plt.xlabel('Day of Year (UID)')
plt.ylabel('Own Device Count')
plt.title('Residential Activity by Cluster (Port Arthur)')
plt.legend()
plt.tight_layout()
plt.savefig('./outputs/clustered_mobility_plot.png', dpi=300)
plt.show()

# Group by cluster and compute average RCN
rcn_per_cluster = rcn_df.groupby('cluster')['RCN'].mean().reset_index()
rcn_per_cluster.to_csv('./outputs/rcn_per_cluster.csv', index=False)

print(rcn_per_cluster)

plt.figure(figsize=(8, 6))
sns.barplot(x='cluster', y='RCN', data=rcn_per_cluster, palette='viridis')
plt.title('Average RCN per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average RCN (Resilience Capacity Number)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('./outputs/average_rcn_per_cluster.png')
plt.show()
def run():
    print("Running Paper 1 RCN analysis...")