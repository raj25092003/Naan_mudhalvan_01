import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('customer_data.csv')

# Display basic information
print(df.head())

features = ['Age', 'AnnualIncome', 'SpendingScore']  
X = df[features]

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Display the scaled features
print(pd.DataFrame(X_scaled, columns=features).head())

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method For Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
sns.scatterplot(data=df, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='Set1')
plt.title('Customer Segmentation')
plt.show()

# Show cluster centers
print("Cluster Centers:")
print(scaler.inverse_transform(kmeans.cluster_centers_))


df['PurchaseHistory'] = df['PurchaseHistory'].apply(lambda x: x.split(','))  # Converting to list of product IDs

# Create a dummy purchase matrix (binary: 1 if purchased, 0 if not)
product_ids = list(set([item for sublist in df['PurchaseHistory'] for item in sublist]))  # Get all unique product IDs
purchase_matrix = pd.DataFrame(0, index=df['CustomerID'], columns=product_ids)

for idx, row in df.iterrows():
    for product in row['PurchaseHistory']:
        purchase_matrix.loc[row['CustomerID'], product] = 1

# Compute cosine similarity between customers
cosine_sim = cosine_similarity(purchase_matrix)

# Convert cosine similarity matrix into a DataFrame
cosine_sim_df = pd.DataFrame(cosine_sim, index=df['CustomerID'], columns=df['CustomerID'])

# Function to get personalized recommendations for a customer
def get_recommendations(customer_id, top_n=5):
    similar_customers = cosine_sim_df[customer_id].sort_values(ascending=False).iloc[1:top_n+1]
    recommendations = []
    
    # Get the products purchased by similar customers
    for similar_customer in similar_customers.index:
        similar_customer_products = df.loc[df['CustomerID'] == similar_customer, 'PurchaseHistory'].values[0]
        for product in similar_customer_products:
            if product not in df.loc[df['CustomerID'] == customer_id, 'PurchaseHistory'].values[0]:
                recommendations.append(product)
    
    return recommendations[:top_n]

customer_id = 1
print(f"Recommendations for Customer {customer_id}: {get_recommendations(customer_id)}")

print("Targeted Campaign for Cluster 1 (High Income, High Spending):")
print(targeted_campaign[['CustomerID', 'Age', 'AnnualIncome', 'SpendingScore']])





