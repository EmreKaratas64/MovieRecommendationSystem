# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:23:58 2024

@author: EMRE KARATAS
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

print("\n")

# Display the first few rows of each dataframe
print(movies.head())
print("\n")
print(ratings.head())

print("\n")

# Check for missing values
print(movies.isnull().sum())
print(ratings.isnull().sum())

# Convert genres to a list of genres
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

print("\n")

# Display the modified dataframe
print(movies.head())
print("\n")


# Calculate average rating for each movie
movie_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
movie_ratings.columns = ['movieId', 'avg_rating']

# Merge with the movies dataframe
movies = movies.merge(movie_ratings, on='movieId', how='left')


# Split the ratings data into training and testing sets
train_data, test_data = train_test_split(ratings, test_size=0.1, random_state=42)

# Update the training data for the models
movie_ratings_train = train_data.groupby('movieId')['rating'].mean().reset_index()
movie_ratings_train.columns = ['movieId', 'avg_rating']
movies_train = movies.merge(movie_ratings_train, on='movieId', how='left')

# Drop the redundant column
movies_train = movies_train.drop(columns=['avg_rating_x'])
movies_train = movies_train.rename(columns={'avg_rating_y': 'avg_rating'})

# if we dont use above dropping this data will have avg_rating_x and avg_rating_y and this will result
# a key error for this column when tried to use it in recommend_by_genre function
#print("Movies Train\n")
#print(movies_train.head()) 

user_movie_matrix_train = train_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
scaler = StandardScaler()
user_movie_matrix_train_scaled = scaler.fit_transform(user_movie_matrix_train)

# Apply K-means clustering
kmeans = KMeans(n_clusters=10, random_state=0)
user_clusters = kmeans.fit_predict(user_movie_matrix_train_scaled)

# Add cluster labels to users
user_cluster_df = pd.DataFrame({'userId': user_movie_matrix_train.index, 'cluster': user_clusters})

# Function to recommend based on genre
def recommend_by_genre(user_id, movie_id, top_n=5):
    movie_genres = movies_train[movies_train['movieId'] == movie_id]['genres'].values[0]
    similar_movies = movies_train[movies_train['genres'].apply(lambda x: any(genre in x for genre in movie_genres))]
    recommendations = similar_movies.sort_values(by='avg_rating', ascending=False).head(top_n)
    return recommendations[['movieId', 'title', 'avg_rating']]

# Function to recommend based on clustering with tuned number of clusters
def recommend_by_cluster(user_id, top_n=5):
    user_cluster = user_cluster_df[user_cluster_df['userId'] == user_id]['cluster'].values[0]
    similar_users = user_cluster_df[user_cluster_df['cluster'] == user_cluster]['userId'].values
    similar_users_ratings = ratings[ratings['userId'].isin(similar_users)]
    top_rated_movies = similar_users_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(top_n).index
    recommendations = movies[movies['movieId'].isin(top_rated_movies)]
    return recommendations[['movieId', 'title', 'avg_rating']]


print("\n")

# Example usage
print(recommend_by_genre(1, 1))

print("\n")
# Example usage
print(recommend_by_cluster(1))

print("\n")


# Function to calculate MSE for rule-based recommender
def evaluate_rule_based(user_id, movie_id, actual_rating):
    predicted_rating = recommend_by_genre(user_id, movie_id)['avg_rating'].values[0]
    mse = mean_squared_error([actual_rating], [predicted_rating])
    return mse

# Function to calculate MSE for clustering-based recommender
def evaluate_clustering_based(user_id, actual_ratings):
    predicted_ratings = recommend_by_cluster(user_id)['avg_rating'].values[:len(actual_ratings)]
    mse = mean_squared_error(actual_ratings, predicted_ratings)
    return mse

print("\n")
# Example evaluation
print(evaluate_rule_based(72, 489, 4.0))
print(evaluate_clustering_based(21, [4.0, 5.0, 3.0]))

