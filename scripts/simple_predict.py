#!/usr/bin/env python3
"""
Simplified Prediction Service for Kubernetes Failure Prediction System

This script:
1. Loads the trained model
2. Continuously collects metrics from the Flask app
3. Makes real-time predictions for potential failures
4. Provides a simple command-line interface for viewing predictions
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import requests
import joblib
import pickle
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Kubernetes Failure Prediction Service')
    parser.add_argument('--model-dir', default='models', help='Directory containing trained model')
    parser.add_argument('--app-host', default='localhost', help='Host of the Flask app')
    parser.add_argument('--app-port', default=5000, type=int, help='Port of the Flask app')
    parser.add_argument('--interval', default=5, type=int, help='Interval between predictions in seconds')
    parser.add_argument('--threshold', default=0.7, type=float, help='Prediction threshold')
    return parser.parse_args()

def load_model(model_dir):
    """Load the trained model and related artifacts"""
    print(f"Loading model from {model_dir}...")
    
    model_path = os.path.join(model_dir, 'rf_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    # Load model, scaler, and feature names
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    print("Model loaded successfully")
    return model, scaler, feature_names

def collect_metrics(app_url):
    """Collect metrics from the app"""
    metrics = {}
    
    try:
        # Get main metrics
        response = requests.get(app_url)
        if response.status_code == 200:
            data = response.json()
            metrics['memory_usage_mb'] = data.get('memory_usage_mb', 0)
            metrics['cpu_usage_percent'] = data.get('cpu_usage_percent', 0)
            metrics['timestamp'] = data.get('time', time.time())
        else:
            print(f"Error fetching metrics: Status code {response.status_code}")
            return None
            
        # Check health status
        health_response = requests.get(f"{app_url}/health")
        metrics['is_healthy'] = 1 if health_response.status_code == 200 else 0
        
        # Generate some load and measure success/error
        success_count = 0
        error_count = 0
        
        for _ in range(5):
            try:
                resp = requests.get(f"{app_url}/api/data")
                if resp.status_code == 200:
                    success_count += 1
                else:
                    error_count += 1
            except Exception:
                error_count += 1
        
        metrics['success_requests'] = success_count
        metrics['error_requests'] = error_count
        metrics['error_rate'] = error_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
        
        return metrics
        
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        return None

def preprocess_metrics(metrics, scaler, feature_names):
    """Preprocess metrics for prediction"""
    # Create a DataFrame with the features
    df = pd.DataFrame([metrics])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Keep only the features used by the model, in the correct order
    df = df[feature_names]
    
    # Scale the features
    X_scaled = scaler.transform(df)
    
    return X_scaled

def analyze_prediction(probability, metrics, threshold):
    """Analyze the prediction and determine the likely failure type"""
    is_failure = probability >= threshold
    
    failure_type = None
    if is_failure:
        # Determine the most likely type of failure based on the metrics
        if metrics['memory_usage_mb'] > 300:
            failure_type = "Memory leak"
        elif metrics['cpu_usage_percent'] > 80:
            failure_type = "CPU overload"
        elif not metrics['is_healthy']:
            failure_type = "Service down"
        elif metrics['error_rate'] > 0.2:
            failure_type = "High error rate"
        else:
            failure_type = "Unknown failure type"
    
    return is_failure, failure_type

def main(args):
    # Load the model
    model, scaler, feature_names = load_model(args.model_dir)
    
    app_url = f"http://{args.app_host}:{args.app_port}"
    
    print(f"\nStarting prediction service for app at {app_url}")
    print(f"Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Collect metrics
            metrics = collect_metrics(app_url)
            
            if metrics:
                # Preprocess metrics
                X = preprocess_metrics(metrics, scaler, feature_names)
                
                # Make prediction
                probability = model.predict_proba(X)[0, 1]
                
                # Analyze prediction
                is_failure, failure_type = analyze_prediction(probability, metrics, args.threshold)
                
                # Print prediction
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Prediction: {probability:.4f}")
                print(f"  Memory: {metrics['memory_usage_mb']:.1f} MB, CPU: {metrics['cpu_usage_percent']:.1f}%")
                
                if metrics['error_rate'] > 0:
                    print(f"  Error Rate: {metrics['error_rate']:.2%}")
                    
                if not metrics['is_healthy']:
                    print(f"  ⚠️ Service Health Check: FAILED")
                
                if is_failure:
                    print(f"  ⚠️ FAILURE PREDICTED: {failure_type} (probability: {probability:.2%})")
                print()
            
            # Wait for next prediction
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nPrediction service stopped")

if __name__ == "__main__":
    args = parse_args()
    main(args) 