# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:36:48 2024

@author: EMRE KARATAS
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



def main():
    # Function to calculate the rule-based recommendation
    def rule_based_recommender(user_id, movie_id):
        return movie_mean_rating.get(movie_id, 0)


    # Function to calculate the clustering-based recommendation
    def clustering_based_recommender(user_id, movie_id):
        cluster = user_cluster_map.get(user_id)
        cluster_users = [user for user, cluster_id in user_cluster_map.items() if cluster_id == cluster]
        cluster_ratings = train[train['userId'].isin(cluster_users)]
        cluster_movie_mean_rating = cluster_ratings.groupby('movieId')['rating'].mean().to_dict()
        return cluster_movie_mean_rating.get(movie_id, 0)

    # Combine both recommenders (simple average of both)
    def combined_recommender(user_id, movie_id):
        rule_based_rating = rule_based_recommender(user_id, movie_id)
        clustering_based_rating = clustering_based_recommender(user_id, movie_id)
        return (rule_based_rating + clustering_based_rating) / 2
    
    # Load dataset
    ratings = pd.read_csv('ratings.csv')
    #movies = pd.read_csv('movies.csv')

    original_min = ratings['rating'].min()
    original_max = ratings['rating'].max()    

    # Normalize the ratings
    scaler = MinMaxScaler()
    ratings['rating'] = scaler.fit_transform(ratings[['rating']])

    # Split into training and test set
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)

    # Rule-Based Recommender: Simple rule based on mean rating per movie
    movie_mean_rating = train.groupby('movieId')['rating'].mean().to_dict()


    # Clustering-Based Recommender
    # Pivot table for users x movies
    user_movie_matrix = train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Apply KMeans clustering
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    user_clusters = kmeans.fit_predict(user_movie_matrix)

    # Add cluster info to users
    user_cluster_map = {user: cluster for user, cluster in zip(user_movie_matrix.index, user_clusters)}


    # Evaluate the system
    predictions = []
    actuals = []

    for _, row in test.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        actual_rating = row['rating']
        
        predicted_rating = combined_recommender(user_id, movie_id)
        
        predictions.append(predicted_rating)
        actuals.append(actual_rating)

    # Calculate MSE
    mse = mean_squared_error(actuals, predictions)

    print(f"Mean Squared Error: {mse}")

    # Denormalize MSE for interpretation
    denormalized_mse = mse * (original_max - original_min) ** 2

    print(f"Denormalized Mean Squared Error: {denormalized_mse}")



if __name__ == '__main__':
    main()






