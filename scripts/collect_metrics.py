#!/usr/bin/env python3
"""
Metrics Collection Script for Kubernetes Failure Prediction System

This script:
1. Simulates different failure conditions in the demo app
2. Collects metrics from Prometheus
3. Labels and saves the data for training ML models
"""

import os
import time
import random
import json
import argparse
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='Collect metrics and simulate failures')
    parser.add_argument('--app-host', default='localhost', help='Host of the demo app')
    parser.add_argument('--app-port', default=80, type=int, help='Port of the demo app')
    parser.add_argument('--prom-host', default='localhost', help='Host of Prometheus')
    parser.add_argument('--prom-port', default=9090, type=int, help='Port of Prometheus')
    parser.add_argument('--duration', default=120, type=int, help='Duration of each simulation in seconds')
    parser.add_argument('--output', default='metrics_data.csv', help='Output file for collected metrics')
    return parser.parse_args()

def simulate_failure(app_url, failure_type, enable=True):
    """Enable or disable a specific failure simulation"""
    try:
        response = requests.post(
            f"{app_url}/failures",
            json={failure_type: enable},
            headers={'Content-Type': 'application/json'}
        )
        return response.json()
    except Exception as e:
        print(f"Error setting failure {failure_type}: {e}")
        return None

def reset_all_failures(app_url):
    """Reset all failure simulations to disabled"""
    try:
        response = requests.post(
            f"{app_url}/failures",
            json={
                "high_latency": False,
                "memory_leak": False,
                "cpu_spike": False,
                "error_rate": False,
                "service_down": False
            },
            headers={'Content-Type': 'application/json'}
        )
        return response.json()
    except Exception as e:
        print(f"Error resetting failures: {e}")
        return None

def query_prometheus(prom_url, query):
    """Query Prometheus for metrics"""
    try:
        response = requests.get(
            f"{prom_url}/api/v1/query",
            params={'query': query}
        )
        return response.json()
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
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
                    metrics[metric_name] = float(item['value'][1]) if len(item.get('value', [])) > 1 else None
    
    # Add timestamp
    metrics['timestamp'] = time.time()
    
    return metrics

def generate_load(app_url, num_requests=20, interval=0.1):
    """Generate some load on the app"""
    for _ in range(num_requests):
        try:
            requests.get(f"{app_url}/api/data")
            time.sleep(interval)
        except:
            pass

def run_failure_simulations(args):
    """Run through different failure simulations and collect metrics"""
    app_url = f"http://{args.app_host}:{args.app_port}"
    prom_url = f"http://{args.prom_host}:{args.prom_port}"
    
    # Define failure scenarios
    failure_scenarios = [
        {"name": "normal", "failures": {}},
        {"name": "high_latency", "failures": {"high_latency": True}},
        {"name": "memory_leak", "failures": {"memory_leak": True}},
        {"name": "cpu_spike", "failures": {"cpu_spike": True}},
        {"name": "error_rate", "failures": {"error_rate": True}},
        {"name": "service_down", "failures": {"service_down": True}},
        {"name": "combined_resource", "failures": {"memory_leak": True, "cpu_spike": True}},
        {"name": "combined_error", "failures": {"high_latency": True, "error_rate": True}}
    ]
    
    all_metrics = []
    
    for scenario in failure_scenarios:
        print(f"\nStarting {scenario['name']} scenario")
        
        # Reset all failures
        reset_all_failures(app_url)
        time.sleep(5)  # Give time for app to stabilize
        
        # Set the failures for this scenario
        for failure_type, value in scenario['failures'].items():
            simulate_failure(app_url, failure_type, value)
        
        # Collect metrics for the specified duration
        start_time = time.time()
        end_time = start_time + args.duration
        
        while time.time() < end_time:
            # Generate some load
            generate_load(app_url)
            
            # Collect metrics
            metrics = collect_current_metrics(prom_url)
            
            # Add scenario info
            metrics['scenario'] = scenario['name']
            metrics['is_failure'] = 1 if scenario['name'] != "normal" else 0
            
            all_metrics.append(metrics)
            
            # Wait a bit before next collection
            time.sleep(5)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_metrics)
    df.to_csv(args.output, index=False)
    print(f"\nMetrics collected and saved to {args.output}")
    
    # Display some stats
    print("\nDataset statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Failure samples: {df['is_failure'].sum()} ({df['is_failure'].mean()*100:.1f}%)")
    print(f"Metrics columns: {', '.join(df.columns)}")

if __name__ == "__main__":
    args = parse_args()
    run_failure_simulations(args) 