#!/usr/bin/env python3
"""
Train ML models to predict Kubernetes failures from collected metrics.

This script:
1. Loads the metrics dataset created by collect_metrics.py
2. Preprocesses the data 
3. Trains multiple ML models (Random Forest, XGBoost, and Neural Network)
4. Evaluates and compares the models
5. Saves the best model for deployment
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Train ML models on metrics data')
    parser.add_argument('--input', default='metrics_data.csv', help='Input metrics CSV file')
    parser.add_argument('--output-dir', default='models', help='Directory to save trained models')
    parser.add_argument('--test-size', default=0.2, type=float, help='Test set size')
    parser.add_argument('--feature-plot', default=True, type=bool, help='Generate feature importance plots')
    return parser.parse_args()

def preprocess_data(df):
    """Preprocess the metrics dataset for training"""
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Convert timestamp to datetime and extract features
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Drop non-numeric columns that aren't useful for training
    drop_columns = ['timestamp', 'scenario']
    X = df.drop(drop_columns + ['is_failure'], axis=1)
    y = df['is_failure']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the column names and scaler
    feature_names = X.columns
    
    return X_scaled, y, feature_names, scaler

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    """Train a Random Forest model"""
    print("\nTraining Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Random Forest - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return rf, feature_importance, {'accuracy': accuracy, 'f1': f1, 'auc': auc}

def train_xgboost(X_train, y_train, X_test, y_test, feature_names):
    """Train an XGBoost model"""
    print("\nTraining XGBoost model...")
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"XGBoost - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return clf, feature_importance, {'accuracy': accuracy, 'f1': f1, 'auc': auc}

def train_neural_network(X_train, y_train, X_test, y_test):
    """Train a Neural Network model"""
    print("\nTraining Neural Network model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Make predictions
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Neural Network - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, None, {'accuracy': accuracy, 'f1': f1, 'auc': auc}

def plot_feature_importance(feature_importance_rf, feature_importance_xgb, output_dir):
    """Plot feature importance for Random Forest and XGBoost models"""
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    top_features_rf = feature_importance_rf.head(10)
    sns.barplot(x='importance', y='feature', data=top_features_rf)
    plt.title('Random Forest Top 10 Feature Importance')
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    top_features_xgb = feature_importance_xgb.head(10)
    sns.barplot(x='importance', y='feature', data=top_features_xgb)
    plt.title('XGBoost Top 10 Feature Importance')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()

def save_model(model, model_type, metrics, scaler, feature_names, output_dir):
    """Save the trained model and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'rf' or model_type == 'xgb':
        # Save sklearn/xgboost model
        joblib.dump(model, os.path.join(output_dir, f'{model_type}_model.joblib'))
    elif model_type == 'nn':
        # Save neural network model
        model.save(os.path.join(output_dir, 'nn_model'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save metrics
    with open(os.path.join(output_dir, f'{model_type}_metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

def main(args):
    # Load the dataset
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess the data
    X, y, feature_names, scaler = preprocess_data(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train and evaluate Random Forest
    rf_model, rf_importance, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    
    # Train and evaluate XGBoost
    xgb_model, xgb_importance, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, feature_names)
    
    # Train and evaluate Neural Network
    nn_model, _, nn_metrics = train_neural_network(X_train, y_train, X_test, y_test)
    
    # Plot feature importance
    if args.feature_plot and rf_importance is not None and xgb_importance is not None:
        plot_feature_importance(rf_importance, xgb_importance, args.output_dir)
    
    # Save the models
    save_model(rf_model, 'rf', rf_metrics, scaler, feature_names, args.output_dir)
    save_model(xgb_model, 'xgb', xgb_metrics, scaler, feature_names, args.output_dir)
    save_model(nn_model, 'nn', nn_metrics, scaler, feature_names, args.output_dir)
    
    # Determine the best model
    best_model = max([('rf', rf_metrics['f1']), ('xgb', xgb_metrics['f1']), ('nn', nn_metrics['f1'])], key=lambda x: x[1])
    
    print(f"\nBest model: {best_model[0]} with F1 score: {best_model[1]:.4f}")
    print(f"All models saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args) 