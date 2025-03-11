
# -*- coding: utf-8 -*-


#Using classification method (Neural Networks) on the dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Set random state
random_state = 0

# Load the data
data = pd.read_csv('nba_stats.csv')

# Separate features and target
X = data.drop('Pos', axis=1)
y = data['Pos']

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(include=[np.number]).columns
categorical_columns = X.select_dtypes(exclude=[np.number]).columns

# Encode categorical variables
label_encoders = {}
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    X[col] = label_encoders[col].fit_transform(X[col].astype(str))

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Task 1: Neural Network with 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state, stratify=y)

# Define parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(100, 50), (150, 100), (200, 100), (100, 100)],
    'alpha': [0.001, 0.01, 0.1],  # L2 regularization
    'max_iter': [1000, 2000, 2500],
    'learning_rate_init': [0.0001, 0.001, 0.01]  # Learning rate
}

# Create the model for GridSearchCV
nn_model = MLPClassifier(random_state=random_state, early_stopping=True, n_iter_no_change=10, warm_start=True)

# Perform GridSearchCV to tune the hyperparameters with verbose=0 to suppress fitting progress
grid_search = GridSearchCV(nn_model, param_grid, cv=5, n_jobs=-1, verbose=0)  # Set verbose=0 to suppress output
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_nn_model = grid_search.best_estimator_

# Predict and calculate accuracy for training set
y_train_pred = best_nn_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict and calculate accuracy for validation set
y_val_pred = best_nn_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("Task 1 Results:")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("\nTraining Set Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))

print("\nValidation Set Confusion Matrix:")
print(confusion_matrix(y_val, y_val_pred))


# Task 2: Evaluate on dummy test set
dummy_test = pd.read_csv('dummy_test.csv')
X_dummy_test = dummy_test.drop('Pos', axis=1)
y_dummy_test = dummy_test['Pos']

# Apply label encoding and scaling to dummy test data
for col in categorical_columns:
    if col in X_dummy_test.columns:
        X_dummy_test[col] = label_encoders[col].transform(X_dummy_test[col].astype(str))

X_dummy_test_scaled = scaler.transform(X_dummy_test)

# Predict on dummy test set
y_dummy_pred = best_nn_model.predict(X_dummy_test_scaled)
dummy_test_accuracy = accuracy_score(y_dummy_test, y_dummy_pred)

print("\nTask 2 Results:")
print(f"Dummy Test Set Accuracy: {dummy_test_accuracy:.4f}")
print("\nDummy Test Set Confusion Matrix:")
print(confusion_matrix(y_dummy_test, y_dummy_pred))


#Using the same model with the same parameters you have chosen in classification above. 
#However, instead of using 80%/20% train/test split, apply 10-fold stratified cross-validation. 
#Print out the accuracy of each fold. Print out the average accuracy across all the folds.


# Task 3: 10-fold stratified cross-validation with regularized model
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
fold_accuracies = []

print("\nTask 3 Results:")
for fold, (train_index, val_index) in enumerate(skf.split(X_scaled, y), 1):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Train using the best model from grid search
    best_nn_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = best_nn_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    fold_accuracies.append(val_accuracy)
    print(f"Fold {fold} Accuracy: {val_accuracy:.4f}")

# Average accuracy across all folds
average_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Accuracy across all folds: {average_accuracy:.4f}")
