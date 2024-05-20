# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:33:00 2024

@author: EMRE KARATAS

recommender system based on user_id, movie_id !!!
test verileri farklı olması gerekiyor, aynı user_id ve movie_id sisteme gelmemeli test kısmında !!!
"""


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Check for missing values
print(movies.isnull().sum())
print(ratings.isnull().sum())

# Convert genres to a list of genres
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Calculate average rating for each movie
movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
movie_ratings.columns = ['movieId', 'avg_rating']

# Merge with the movies dataframe
movies = movies.merge(movie_ratings, on='movieId', how='left')

# Split the ratings data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Update the training data for the models
movie_ratings_train = train_data.groupby('movieId')['rating'].mean().reset_index()
movie_ratings_train.columns = ['movieId', 'avg_rating']
movies_train = movies.merge(movie_ratings_train, on='movieId', how='left')

user_movie_matrix_train = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
scaler = StandardScaler()
user_movie_matrix_train_scaled = scaler.fit_transform(user_movie_matrix_train)

# Function to recommend based on genre with user-based collaborative filtering
def recommend_by_genre(user_id, movie_id, top_n=5):
    movie_genres = movies_train[movies_train['movieId'] == movie_id]['genres'].values[0]
    similar_movies = movies_train[movies_train['genres'].apply(lambda x: any(genre in x for genre in movie_genres))]
    user_ratings = train_data[train_data['userId'] == user_id]
    rated_movie_ids = user_ratings['movieId'].tolist()
    similar_movies = similar_movies[~similar_movies['movieId'].isin(rated_movie_ids)]
    similar_movies = similar_movies.merge(movie_ratings_train, on='movieId', how='left')
    recommendations = similar_movies.sort_values(by='avg_rating', ascending=False).head(top_n)
    return recommendations[['movieId', 'title', 'avg_rating']]

# Apply K-means clustering
def apply_kmeans(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    user_clusters = kmeans.fit_predict(user_movie_matrix_train_scaled)
    user_cluster_df = pd.DataFrame({'userId': user_movie_matrix_train.index, 'cluster': user_clusters})
    return user_cluster_df

# Function to recommend based on clustering with tuned number of clusters
def recommend_by_cluster(user_id, user_cluster_df, top_n=5):
    user_cluster = user_cluster_df[user_cluster_df['userId'] == user_id]['cluster'].values[0]
    similar_users = user_cluster_df[user_cluster_df['cluster'] == user_cluster]['userId'].values
    similar_users_ratings = train_data[train_data['userId'].isin(similar_users)]
    top_rated_movies = similar_users_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_n).index
    recommendations = movies_train[movies_train['movieId'].isin(top_rated_movies)]
    return recommendations[['movieId', 'title', 'avg_rating']]

# Evaluate rule-based recommender
def evaluate_rule_based_model(test_data):
    mse_list = []
    for _, row in test_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        recommended_movies = recommend_by_genre(user_id, movie_id)
        if not recommended_movies.empty:
            predicted_rating = recommended_movies['avg_rating'].values[0]
            mse_list.append(mean_squared_error([actual_rating], [predicted_rating]))
    return np.mean(mse_list)

# Evaluate clustering-based recommender
def evaluate_clustering_based_model(test_data, n_clusters):
    user_cluster_df = apply_kmeans(n_clusters)
    mse_list = []
    for user_id, group in test_data.groupby('userId'):
        actual_ratings = group['rating'].values
        predicted_ratings = recommend_by_cluster(user_id, user_cluster_df)['avg_rating'].values[:len(actual_ratings)]
        if len(predicted_ratings) < len(actual_ratings):
            predicted_ratings = np.pad(predicted_ratings, (0, len(actual_ratings) - len(predicted_ratings)), 'constant', constant_values=(0, 0))
        mse_list.append(mean_squared_error(actual_ratings, predicted_ratings))
    return np.mean(mse_list)

# Calculate MSE for both models
rule_based_mse = evaluate_rule_based_model(test_data)
clustering_based_mse = evaluate_clustering_based_model(test_data, n_clusters=10)

print(f"Rule-Based Recommender MSE: {rule_based_mse}")
print(f"Clustering-Based Recommender MSE: {clustering_based_mse}")