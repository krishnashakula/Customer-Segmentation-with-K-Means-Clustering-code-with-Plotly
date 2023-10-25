import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# Load the dataset from Kaggle
data = pd.read_csv('Mall_Customers.csv')

# Select the features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Determine the optimal number of clusters using the elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Create an interactive elbow plot using Plotly
elbow_fig = px.line(x=range(1, 11), y=inertias, title='Elbow Method for Optimal k')
elbow_fig.update_traces(mode='lines+markers')
elbow_fig.update_xaxes(title='Number of Clusters (k)')
elbow_fig.update_yaxes(title='Inertia')
elbow_fig.show()

# Perform K-means clustering with the chosen k
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(X)

# Add the cluster labels to the original dataset
data['Cluster'] = labels

# Create an interactive scatterplot using Plotly
scatter_fig = px.scatter(data, x='Annual Income (k$)', y='Spending Score (1-100)', color='Cluster',
                         title='Customer Segmentation with K-Means Clustering', template='plotly')
scatter_fig.update_xaxes(title='Annual Income (k$)')
scatter_fig.update_yaxes(title='Spending Score (1-100)')
scatter_fig.show()


# Explore data types and descriptive statistics
print(data.info())
print(data.describe())

# Visualize distributions of variables
plt.hist(data['Annual Income (k$)'], bins=10)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.title('Distribution of Annual Income')
plt.show()

plt.hist(data['Spending Score (1-100)'], bins=10)
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.title('Distribution of Spending Score')
plt.show()

# Analyze correlation between variables
correlation = data['Annual Income (k$)'].corr(data['Spending Score (1-100)'])
print('Correlation between Annual Income and Spending Score:', correlation)

# Perform K-means clustering to identify customer segments
kmeans = KMeans(n_clusters=5, random_state=0)
labels = kmeans.fit_predict(data[['Annual Income (k$)', 'Spending Score (1-100)']])
data['Cluster'] = labels
