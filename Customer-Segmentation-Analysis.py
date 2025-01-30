
# Project Repository: https://github.com/zelihaguven/Customer-Segmentation-Analysis

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
# Load the customer data
data = pd.read_csv('data/Mall_Customers.csv')

# Step 2: Data Exploration
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Data Cleaning (if necessary)
# Check for missing values
print("Missing values in the dataset:\n", data.isnull().sum())

# Step 4: Feature Selection for Clustering
# Selecting relevant features for segmentation
features = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 5: Data Standardization
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 6: K-Means Clustering
# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# From the elbow graph, choose the optimal number of clusters (e.g., 5)
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Step 7: Add cluster labels to the original data
data['Cluster'] = clusters

# Step 8: Visualize the clusters
plt.figure(figsize=(10, 5))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster')
plt.show()

# Step 9: Save the results
data.to_csv('data/Customer_Segmentation_Results.csv', index=False)
print("Segmentation results saved to 'data/Customer_Segmentation_Results.csv'")