import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             silhouette_score)
from matplotlib import pyplot

from sklearn.cluster import KMeans, AgglomerativeClustering # clustering algorithms
from sklearn.decomposition import PCA # dimensionality reduction
from sklearn.metrics import silhouette_score # used as a metric to evaluate the cohesion in a cluster
from sklearn.neighbors import NearestNeighbors # for selecting the optimal eps value when using DBSCAN
import numpy as np
from sklearn.cluster import DBSCAN

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import SilhouetteVisualizer

from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances

# Import data
df = pd.read_excel("D:\MMA\Fall 2022\Data_mining\Individual_project\Kickstarter.xlsx")

# Pre-Processing
df = df.dropna()

df.drop(['id','name','deadline','state_changed_at', 'created_at', 'launched_at', 'static_usd_rate',
              'disable_communication','pledged','created_at_weekday','created_at_month','created_at_day',
              'created_at_yr', 'created_at_hr', 'create_to_launch_days','deadline_yr','state_changed_at_yr',
              'currency','name_len','blurb_len','state_changed_at_weekday',
              'state_changed_at_day','state_changed_at_yr','state_changed_at_hr', 'state_changed_at_month',
                'launched_at_day','deadline_day','launch_to_state_change_days','launched_at_yr','country','spotlight'], axis=1, inplace=True )

df.drop(df[df.state == 'canceled'].index, inplace=True)
df.drop(df[df.state == 'suspended'].index, inplace=True)

df.describe()

print(df.columns)

#Eliminate the 90% for backers_count
bc_90_quantile = df["backers_count"].quantile(0.9)
goal_95_quantile = df["goal"].quantile(0.95)
goal_5_quantile = df["goal"].quantile(0.05)

indexdrop = df[(df["backers_count"] >= bc_90_quantile) | (df['goal'] >= goal_95_quantile)| (df['goal'] <= goal_5_quantile)].index
df.drop(indexdrop, inplace=True)

#Dummify

X = pd.get_dummies(df)
X = pd.get_dummies(X, columns=['staff_pick'])
#X['spotlight'] = X['spotlight'].map({True:'True', False:'False'})
#X = pd.get_dummies(X, columns=['spotlight'])

# Standardize predictors
from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#X_std = scaler.fit_transform(X)
#X_std=pd.DataFrame(X_std, columns=X.columns)

##Create function to plot Silhouette score for different clusters
def silhouettePlot(range_, data):
    '''
    we will use this function to plot a silhouette plot that helps us to evaluate the cohesion in clusters (k-means only)
    '''
    half_length = int(len(range_)/2)
    range_list = list(range_)
    fig, ax = plt.subplots(half_length, 2, figsize=(15,8))
    for _ in range_:
        kmeans = KMeans(n_clusters=_, random_state=42)
        q, mod = divmod(_ - range_list[0], 2)
        sv = SilhouetteVisualizer(kmeans, colors="yellowbrick", ax=ax[q][mod])
        ax[q][mod].set_title("Silhouette Plot with n={} Cluster".format(_))
        sv.fit(data)
    fig.tight_layout()
    fig.show()
    fig.savefig('elbow_plot.png')

##Create elbow Plots score for different clusters
def elbowPlot(range_, data, figsize=(10,10)):
    '''
    the elbow plot function helps to figure out the right amount of clusters for a dataset
    '''
    inertia_list = []
    for n in range_:
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(data)
        inertia_list.append(kmeans.inertia_)
        
    # plotting
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    sns.lineplot(y=inertia_list, x=range_, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Inertia")
    ax.set_xticks(list(range_))
    fig.show()
    fig.savefig('elbow_plot.png')
#Create PFA function to calculate which features to select using PCA and KMeans Clustering

'''
Compute the sample covariance matrix or correlation matrix,
Compute the Principal components and eigenvalues of the Covariance or Correlation matrix A.
Choose the subspace dimension n, we get new matrix A_n, the vectors Vi are the rows of A_n.
Cluster the vectors |Vi|, using K-Means
For each cluster, find the corresponding vector Vi which is closest to the mean of the cluster. '''

class PFA(object):
    def __init__(self, n_features, q=None):
        self.q = q
        self.n_features = n_features
    
    def fit(self, X):
        if not self.q:
           self.q = X.shape[1]
        sc = StandardScaler()
        X = sc.fit_transform(X)
    
        pca = PCA(n_components=self.q).fit(X) # calculation Covmatrix is embeded in PCA
        A_q = pca.components_.T
    
        kmeans = KMeans(n_clusters=self.n_features, random_state = 42).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_
    
        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([A_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))
    
        self.indices_ = [sorted(f, key=lambda x: x[1])[0][0] for f in dists.values()]
        self.features_ = X[:, self.indices_]

#Identify the number of features to keep
pfa = PFA(n_features=15,q=None)
pfa.fit(X)
x = pfa.features_
print(x)
column_indices = pfa.indices_
X_std_sliced=X.iloc[:,column_indices]


#Elbow plot
elbowPlot(range(1,11), X_std_sliced)

#Silhouette plot
silhouettePlot(range(3,9), X_std_sliced)

#KMeans clustering with 4 clusters

kmeans = KMeans(n_clusters=4, random_state = 42)
cluster_labels = kmeans.fit_predict(X_std_sliced)
X_std_sliced["clusters"] = cluster_labels
X["clusters"] = cluster_labels
# using PCA to reduce the dimensionality
pca = PCA(n_components=2, whiten=False, random_state=42)
X_std_pca = pca.fit_transform(X_std_sliced)
X_std_sliced_pca = pd.DataFrame(data=X_std_pca, columns=["pc_1", "pc_2"])
X_std_sliced_pca["clusters"] = cluster_labels

# plotting the clusters with seaborn
fig = px.scatter(X_std_sliced_pca, x='pc_1', y='pc_2', color='clusters')
fig.update_traces(marker={'size': 4})
fig.show(renderer='browser')

X[X['state_successful'] == 1].groupby('clusters').count()
X[X['state_successful'] == 0].groupby('clusters').count()

