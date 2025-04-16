#!/usr/bin/env python3
"""
Kubernetes Failure Prediction Service

This script:
1. Loads the trained model
2. Continuously collects metrics from Prometheus
3. Makes predictions on potential failures in real-time
4. Provides a simple API endpoint to get current failure predictions
5. Sends alerts when a failure is predicted
"""

import os
import time
import json
import argparse
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import joblib
import pickle
from flask import Flask, jsonify
import threading
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app for the API
app = Flask(__name__)

# Global prediction state
current_predictions = {
    'timestamp': None,
    'probability': 0.0,
    'is_failure_predicted': False,
    'metrics': {},
    'failure_type': None
}

def parse_args():
    parser = argparse.ArgumentParser(description='K8s Failure Prediction Service')
    parser.add_argument('--model-dir', default='models', help='Directory containing trained models')
    parser.add_argument('--model-type', default='rf', choices=['rf', 'xgb', 'nn'], help='Type of model to use')
    parser.add_argument('--prom-host', default='localhost', help='Host of Prometheus')
    parser.add_argument('--prom-port', default=9090, type=int, help='Port of Prometheus')
    parser.add_argument('--threshold', default=0.7, type=float, help='Prediction threshold')
    parser.add_argument('--interval', default=10, type=int, help='Prediction interval in seconds')
    parser.add_argument('--api-port', default=8080, type=int, help='Port for prediction API')
    parser.add_argument('--alert-webhook', default=None, help='Webhook URL for alerts')
    return parser.parse_args()

def load_model(model_dir, model_type):
    """Load the trained ML model and related artifacts"""
    logger.info(f"Loading {model_type} model from {model_dir}")
    
    # Load the scaler
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    
    # Load feature names
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
    else:
        raise FileNotFoundError(f"Feature names not found at {feature_names_path}")
    
    # Load the model
    if model_type in ['rf', 'xgb']:
        model_path = os.path.join(model_dir, f'{model_type}_model.joblib')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    elif model_type == 'nn':
        from tensorflow.keras.models import load_model as load_keras_model
        model_path = os.path.join(model_dir, 'nn_model')
        if os.path.exists(model_path):
            model = load_keras_model(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Model loaded successfully")
    return model, scaler, feature_names

def query_prometheus(prom_url, query):
    """Query Prometheus for metrics"""
    try:
        response = requests.get(
            f"{prom_url}/api/v1/query",
            params={'query': query}
        )
        return response.json()
    except Exception as e:
        logger.error(f"Error querying Prometheus: {e}")
        return None

def collect_current_metrics(prom_url):
    """Collect current metrics from Prometheus"""
    metrics = {}
    
    # Define metrics to collect
    metric_queries = {
        'memory_usage': 'app_memory_usage_bytes{job="k8s-failure-demo"}',
        'cpu_load': 'app_cpu_load{job="k8s-failure-demo"}',
        'request_count': 'sum(app_requests_total{job="k8s-failure-demo"}) by (status)',
        'request_latency': 'histogram_quantile(0.95, sum(rate(app_request_latency_seconds_bucket{job="k8s-failure-demo"}[1m])) by (le))',
        'pod_restarts': 'kube_pod_container_status_restarts_total{pod=~"k8s-failure-demo-.*"}',
        'memory_limit': 'container_spec_memory_limit_bytes{container="k8s-failure-demo"}',
        'cpu_limit': 'container_spec_cpu_quota{container="k8s-failure-demo"}'
    }
    
    for metric_name, query in metric_queries.items():
        result = query_prometheus(prom_url, query)
        if result and result.get('status') == 'success':
            # Extract values from Prometheus response
            for item in result.get('data', {}).get('result', []):
                if 'status' in item.get('metric', {}):
                    status = item['metric']['status']
                    metrics[f"{metric_name}_{status}"] = float(item['value'][1])
                else:
                    metrics[metric_name] = float(item['value'][1]) if len(item.get('value', [])) > 1 else 0.0
    
    # Add timestamp features
    now = datetime.now()
    metrics['hour'] = now.hour
    metrics['minute'] = now.minute
    metrics['dayofweek'] = now.weekday()
    
    return metrics

def preprocess_metrics(metrics, scaler, feature_names):
    """Preprocess the metrics for prediction"""
    # Create a DataFrame with the correct features
    df = pd.DataFrame([metrics])
    
    # Handle missing features
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0.0
    
    # Ensure correct order of features
    df = df[feature_names]
    
    # Apply scaling
    X_scaled = scaler.transform(df)
    
    return X_scaled

def analyze_prediction(probability, metrics, threshold):
    """Analyze the prediction to determine if a failure is likely and what type"""
    is_failure = probability >= threshold
    
    failure_type = None
    if is_failure:
        # Identify likely failure type based on metrics
        if metrics.get('memory_usage', 0) > 300 * 1024 * 1024:  # More than 300MB
            failure_type = "Memory leak"
        elif metrics.get('cpu_load', 0) > 80:
            failure_type = "CPU overload"
        elif metrics.get('request_latency', 0) > 1.0:
            failure_type = "High latency"
        elif metrics.get('request_count_500', 0) > 0:
            failure_type = "High error rate"
        else:
            failure_type = "Unknown failure"
    
    return is_failure, failure_type

def send_alert(webhook_url, prediction_data):
    """Send an alert notification via webhook"""
    if not webhook_url:
        return
    
    try:
        data = {
            "text": f"⚠️ ALERT: Kubernetes failure predicted!",
            "attachments": [
                {
                    "title": f"Failure Type: {prediction_data['failure_type']}",
                    "color": "danger",
                    "fields": [
                        {
                            "title": "Probability",
                            "value": f"{prediction_data['probability']:.2%}",
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": prediction_data['timestamp'],
                            "short": True
                        }
                    ],
                    "text": f"Key metrics: CPU: {prediction_data['metrics'].get('cpu_load', 'N/A')}, Memory: {prediction_data['metrics'].get('memory_usage', 'N/A') / (1024 * 1024):.1f} MB"
                }
            ]
        }
        
        response = requests.post(
            webhook_url,
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to send alert: {response.text}")
    except Exception as e:
        logger.error(f"Error sending alert: {e}")

def prediction_loop(args, model, scaler, feature_names):
    """Main loop for collecting metrics and making predictions"""
    global current_predictions
    
    prom_url = f"http://{args.prom_host}:{args.prom_port}"
    last_alert_time = 0
    alert_cooldown = 300  # 5 minutes between alerts
    
    while True:
        try:
            # Collect metrics
            metrics = collect_current_metrics(prom_url)
            if not metrics:
                logger.warning("Failed to collect metrics, retrying...")
                time.sleep(5)
                continue
            
            # Preprocess metrics
            X = preprocess_metrics(metrics, scaler, feature_names)
            
            # Make prediction
            if args.model_type in ['rf', 'xgb']:
                probability = model.predict_proba(X)[0, 1]
            else:  # Neural network
                probability = model.predict(X)[0, 0]
            
            # Analyze prediction
            is_failure, failure_type = analyze_prediction(probability, metrics, args.threshold)
            
            # Update global state
            current_predictions = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'probability': float(probability),
                'is_failure_predicted': bool(is_failure),
                'metrics': {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                'failure_type': failure_type
            }
            
            # Log prediction
            logger.info(f"Prediction: {probability:.4f}, Threshold: {args.threshold}, Failure: {is_failure}")
            
            # Send alert if failure predicted and we're not in cooldown
            now = time.time()
            if is_failure and (now - last_alert_time) > alert_cooldown and args.alert_webhook:
                send_alert(args.alert_webhook, current_predictions)
                last_alert_time = now
                logger.info("Alert sent")
            
            # Wait for next prediction cycle
            time.sleep(args.interval)
            
        except Exception as e:
            logger.error(f"Error in prediction loop: {e}")
            time.sleep(5)

@app.route('/prediction', methods=['GET'])
def get_prediction():
    """API endpoint to get the current prediction"""
    return jsonify(current_predictions)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

def main(args):
    # Load the model and related artifacts
    model, scaler, feature_names = load_model(args.model_dir, args.model_type)
    
    # Start the prediction thread
    prediction_thread = threading.Thread(
        target=prediction_loop,
        args=(args, model, scaler, feature_names),
        daemon=True
    )
    prediction_thread.start()
    
    # Start the Flask API
    logger.info(f"Starting prediction API on port {args.api_port}")
    app.run(host='0.0.0.0', port=args.api_port)

if __name__ == "__main__":
    args = parse_args()
    main(args) 