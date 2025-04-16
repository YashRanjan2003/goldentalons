#!/usr/bin/env python3
"""
Generate a larger synthetic dataset for Kubernetes Failure Prediction System

This script:
1. Collects a few metrics from the Flask app
2. Generates a larger synthetic dataset based on these metrics
3. Saves the dataset to a CSV file for training
"""

import os
import time
import random
import json
import csv
import argparse
import requests
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic dataset for failure prediction')
    parser.add_argument('--app-host', default='localhost', help='Host of the demo app')
    parser.add_argument('--app-port', default=5000, type=int, help='Port of the demo app')
    parser.add_argument('--num-samples', default=500, type=int, help='Number of samples to generate')
    parser.add_argument('--output', default='synthetic_data.csv', help='Output file for metrics data')
    return parser.parse_args()

def collect_baseline_metrics(app_url):
    """Collect baseline metrics from the app"""
    try:
        # Reset all failures
        requests.post(
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
        
        time.sleep(3)  # Wait for app to stabilize
        
        # Get baseline metrics
        response = requests.get(app_url)
        if response.status_code == 200:
            data = response.json()
            baseline = {
                'memory_usage_mb': data.get('memory_usage_mb', 100),
                'cpu_usage_percent': data.get('cpu_usage_percent', 20)
            }
            return baseline
        else:
            print(f"Error fetching metrics: Status code {response.status_code}")
            return {'memory_usage_mb': 100, 'cpu_usage_percent': 20}
    except Exception as e:
        print(f"Error collecting baseline metrics: {e}")
        return {'memory_usage_mb': 100, 'cpu_usage_percent': 20}

def generate_synthetic_data(baseline, num_samples):
    """Generate synthetic data based on baseline metrics"""
    data = []
    
    # Define failure scenarios and their effects
    failure_scenarios = [
        {
            'name': 'normal',
            'memory_range': (baseline['memory_usage_mb'] * 0.9, baseline['memory_usage_mb'] * 1.1),
            'cpu_range': (baseline['cpu_usage_percent'] * 0.8, baseline['cpu_usage_percent'] * 1.2),
            'error_rate_range': (0, 0.05),
            'is_healthy_prob': 0.99,
            'is_failure': 0
        },
        {
            'name': 'high_latency',
            'memory_range': (baseline['memory_usage_mb'] * 0.9, baseline['memory_usage_mb'] * 1.2),
            'cpu_range': (baseline['cpu_usage_percent'] * 1.1, baseline['cpu_usage_percent'] * 1.5),
            'error_rate_range': (0, 0.1),
            'is_healthy_prob': 0.95,
            'is_failure': 1
        },
        {
            'name': 'memory_leak',
            'memory_range': (baseline['memory_usage_mb'] * 1.5, baseline['memory_usage_mb'] * 3.0),
            'cpu_range': (baseline['cpu_usage_percent'] * 0.9, baseline['cpu_usage_percent'] * 1.3),
            'error_rate_range': (0, 0.1),
            'is_healthy_prob': 0.9,
            'is_failure': 1
        },
        {
            'name': 'cpu_spike',
            'memory_range': (baseline['memory_usage_mb'] * 0.9, baseline['memory_usage_mb'] * 1.3),
            'cpu_range': (baseline['cpu_usage_percent'] * 1.8, baseline['cpu_usage_percent'] * 3.0),
            'error_rate_range': (0.05, 0.2),
            'is_healthy_prob': 0.8,
            'is_failure': 1
        },
        {
            'name': 'error_rate',
            'memory_range': (baseline['memory_usage_mb'] * 0.9, baseline['memory_usage_mb'] * 1.2),
            'cpu_range': (baseline['cpu_usage_percent'] * 0.9, baseline['cpu_usage_percent'] * 1.4),
            'error_rate_range': (0.2, 0.5),
            'is_healthy_prob': 0.7,
            'is_failure': 1
        },
        {
            'name': 'service_down',
            'memory_range': (baseline['memory_usage_mb'] * 0.8, baseline['memory_usage_mb'] * 1.5),
            'cpu_range': (baseline['cpu_usage_percent'] * 0.7, baseline['cpu_usage_percent'] * 1.6),
            'error_rate_range': (0.4, 0.9),
            'is_healthy_prob': 0.1,
            'is_failure': 1
        }
    ]
    
    # Generate samples for each scenario
    for scenario in failure_scenarios:
        samples_per_scenario = num_samples // len(failure_scenarios)
        print(f"Generating {samples_per_scenario} samples for scenario: {scenario['name']}")
        
        for _ in range(samples_per_scenario):
            memory = random.uniform(*scenario['memory_range'])
            cpu = random.uniform(*scenario['cpu_range'])
            error_rate = random.uniform(*scenario['error_rate_range'])
            is_healthy = 1 if random.random() < scenario['is_healthy_prob'] else 0
            
            # Introduce some correlation between metrics
            success_requests = int(random.uniform(5, 20) * (1 - error_rate))
            error_requests = int(success_requests * error_rate / (1 - error_rate) if error_rate < 1 else random.uniform(1, 10))
            
            sample = {
                'memory_usage_mb': memory,
                'cpu_usage_percent': cpu,
                'error_rate': error_rate,
                'is_healthy': is_healthy,
                'success_requests': success_requests,
                'error_requests': error_requests,
                'timestamp': time.time(),
                'scenario': scenario['name'],
                'is_failure': scenario['is_failure']
            }
            data.append(sample)
    
    # Shuffle the data
    random.shuffle(data)
    
    return data

def save_data(data, output_file):
    """Save the generated data to a CSV file"""
    if data:
        # Get fieldnames from first data entry
        fieldnames = list(data[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for item in data:
                writer.writerow(item)
        
        print(f"Generated dataset saved to {output_file}")
        print(f"Total samples: {len(data)}")
        
        # Count normal vs failure samples
        normal_count = sum(1 for item in data if item['is_failure'] == 0)
        failure_count = sum(1 for item in data if item['is_failure'] == 1)
        
        print(f"Normal samples: {normal_count} ({normal_count/len(data)*100:.1f}%)")
        print(f"Failure samples: {failure_count} ({failure_count/len(data)*100:.1f}%)")
    else:
        print("No data to save!")

def main(args):
    app_url = f"http://{args.app_host}:{args.app_port}"
    
    # Collect baseline metrics
    print("Collecting baseline metrics...")
    baseline = collect_baseline_metrics(app_url)
    print(f"Baseline metrics: Memory: {baseline['memory_usage_mb']:.1f} MB, CPU: {baseline['cpu_usage_percent']:.1f}%")
    
    # Generate synthetic data
    print(f"Generating {args.num_samples} synthetic data points...")
    data = generate_synthetic_data(baseline, args.num_samples)
    
    # Save the data
    save_data(data, args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args) 