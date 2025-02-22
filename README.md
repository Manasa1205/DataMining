# DataMining
This repository contains multiple data science projects focusing on classification, clustering, and natural language processing using various machine learning techniques.

## NBA Player Position Classification with Neural Networks
Description:
This project implements a neural network model to classify NBA players into different positions based on their performance statistics. The dataset includes features like points scored, assists, rebounds, and more. The model is trained using an 80% training and 20% validation split, with hyperparameter tuning applied using GridSearchCV. It also evaluates the model on a dummy test set and through 10-fold stratified cross-validation.

Key Techniques Used:
Neural Networks (MLPClassifier)
Hyperparameter tuning (GridSearchCV)
Accuracy and confusion matrix evaluation
Stratified K-Fold Cross Validation

## NLP-Based Text Retrieval and Analysis of U.S. Presidential Inaugural Addresses
Description:
This project focuses on text retrieval and analysis of U.S. presidential inaugural addresses using NLP techniques. The dataset consists of inaugural speeches, and the task involves extracting meaningful insights using various natural language processing methods like sentiment analysis, topic modeling, and more.

Key Techniques Used:
NLP Text Analysis
Sentiment Analysis
Topic Modeling
Text Retrieval and Feature Engineering

## Wine Dataset Clustering - K-means and Hierarchical Analysis
Description:
This project applies K-means clustering and hierarchical clustering techniques on the Wine Dataset, which contains chemical analysis results of wines from three different cultivars. The task involves applying K-means clustering to determine the optimal number of clusters (using the Elbow Method), as well as performing agglomerative hierarchical clustering using both single-link and complete-link methods.

Key Techniques Used:
K-means Clustering
Agglomerative Hierarchical Clustering (Single-Link and Complete-Link)
Silhouette Coefficient Calculation
Dendrogram Visualization

## Requirements: 
Python 3.x, pandas, numpy, scikit-learn, matplotlib,seaborn
