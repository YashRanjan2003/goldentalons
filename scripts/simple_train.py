#!/usr/bin/env python3
"""
Simplified Training Script for Kubernetes Failure Prediction System

This script:
1. Loads the metrics dataset created by simple_collect.py
2. Preprocesses the data
3. Trains a Random Forest model
4. Evaluates the model and reports performance
5. Saves the model for later use
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import joblib
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Train model on metrics data')
    parser.add_argument('--input', default='metrics_data.csv', help='Input metrics CSV file')
    parser.add_argument('--output-dir', default='models', help='Directory to save trained model')
    parser.add_argument('--test-size', default=0.2, type=float, help='Test set size')
    return parser.parse_args()

def preprocess_data(df):
    """Preprocess the metrics dataset for training"""
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Convert timestamp to numeric if it's not
    if 'timestamp' in df.columns and not pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
    
    # Keep only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Make sure 'is_failure' is in the dataset
    if 'is_failure' not in numeric_cols:
        print("Error: 'is_failure' column not found in the dataset")
        return None, None, None, None
    
    # Exclude non-feature columns
    exclude_cols = ['is_failure', 'scenario']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Split into features and target
    X = df[feature_cols]
    y = df['is_failure']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the column names
    feature_names = X.columns
    
    return X_scaled, y, feature_names, scaler

def train_model(X_train, y_train, X_test, y_test, feature_names):
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
    
    print("\nTop 10 Most Important Features:")
    for i, (feature, importance) in enumerate(zip(feature_importance['feature'][:10], 
                                                feature_importance['importance'][:10])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    return rf, {'accuracy': accuracy, 'f1': f1, 'auc': auc}

def save_model(model, metrics, scaler, feature_names, output_dir):
    """Save the trained model and related artifacts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, 'rf_model.joblib'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    print(f"\nModel and artifacts saved to {output_dir}")

def main(args):
    # Load the dataset
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
        print(f"Dataset shape: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Preprocess the data
    X, y, feature_names, scaler = preprocess_data(df)
    if X is None:
        return
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train the model
    model, metrics = train_model(X_train, y_train, X_test, y_test, feature_names)
    
    # Save the model and artifacts
    save_model(model, metrics, scaler, feature_names, args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    main(args) 