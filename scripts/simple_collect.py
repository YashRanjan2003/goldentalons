#!/usr/bin/env python3
"""
Simplified Metrics Collection Script for Kubernetes Failure Prediction System

This script:
1. Simulates different failure conditions in the demo app
2. Collects metrics directly from the app (not using Prometheus)
3. Labels and saves the data for training ML models
"""

import time
import json
import csv
import argparse
import requests
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Collect metrics and simulate failures')
    parser.add_argument('--app-host', default='localhost', help='Host of the demo app')
    parser.add_argument('--app-port', default=5000, type=int, help='Port of the demo app')
    parser.add_argument('--duration', default=60, type=int, help='Duration of each simulation in seconds')
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

def collect_metrics(app_url):
    """Collect metrics directly from the app"""
    try:
        response = requests.get(app_url)
        if response.status_code == 200:
            data = response.json()
            return {
                'memory_usage_mb': data.get('memory_usage_mb', 0),
                'cpu_usage_percent': data.get('cpu_usage_percent', 0),
                'timestamp': data.get('time', time.time())
            }
        else:
            print(f"Error fetching metrics: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return None

def check_health(app_url):
    """Check the health status of the app"""
    try:
        response = requests.get(f"{app_url}/health")
        if response.status_code == 200:
            return True
        return False
    except Exception:
        return False

def generate_load(app_url, num_requests=10, interval=0.2):
    """Generate some load on the app"""
    success_count = 0
    error_count = 0
    
    for _ in range(num_requests):
        try:
            response = requests.get(f"{app_url}/api/data")
            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1
            time.sleep(interval)
        except Exception:
            error_count += 1
    
    return success_count, error_count

def run_failure_simulations(args):
    """Run through different failure simulations and collect metrics"""
    app_url = f"http://{args.app_host}:{args.app_port}"
    
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
        time.sleep(3)  # Give time for app to stabilize
        
        # Set the failures for this scenario
        for failure_type, value in scenario['failures'].items():
            simulate_failure(app_url, failure_type, value)
        
        # Collect metrics for the specified duration
        start_time = time.time()
        end_time = start_time + args.duration
        
        while time.time() < end_time:
            # Generate some load
            success_count, error_count = generate_load(app_url)
            
            # Collect metrics
            metrics = collect_metrics(app_url)
            
            if metrics:
                # Add additional metrics
                metrics['is_healthy'] = 1 if check_health(app_url) else 0
                metrics['success_requests'] = success_count
                metrics['error_requests'] = error_count
                metrics['error_rate'] = error_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
                
                # Add scenario info
                metrics['scenario'] = scenario['name']
                metrics['is_failure'] = 1 if scenario['name'] != "normal" else 0
                
                all_metrics.append(metrics)
            
            # Wait a bit before next collection
            time.sleep(3)
    
    # Write metrics to CSV
    if all_metrics:
        with open(args.output, 'w', newline='') as csvfile:
            # Get fieldnames from first metrics entry
            fieldnames = list(all_metrics[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for metrics in all_metrics:
                writer.writerow(metrics)
        
        print(f"\nMetrics collected and saved to {args.output}")
        print(f"Total samples: {len(all_metrics)}")
        
        # Count failure samples
        failure_count = sum(1 for m in all_metrics if m.get('is_failure', 0) == 1)
        failure_percentage = (failure_count / len(all_metrics)) * 100 if all_metrics else 0
        print(f"Failure samples: {failure_count} ({failure_percentage:.1f}%)")
    else:
        print("No metrics were collected!")

if __name__ == "__main__":
    args = parse_args()
    run_failure_simulations(args) 