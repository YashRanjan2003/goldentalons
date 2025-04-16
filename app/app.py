import os
import time
import random
import logging
import threading
from flask import Flask, jsonify, request, render_template
from prometheus_client import Counter, Histogram, start_http_server, Gauge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total app HTTP requests count', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])
MEMORY_USAGE = Gauge('app_memory_usage_bytes', 'Memory usage in bytes')
CPU_LOAD = Gauge('app_cpu_load', 'CPU load percentage')

# Simulated resource metrics
memory_usage = 100  # MB
cpu_usage = 20  # %

# Failure flags
SIMULATED_FAILURES = {
    "high_latency": False,
    "memory_leak": False,
    "cpu_spike": False,
    "error_rate": False,
    "service_down": False
}

# Start Prometheus metrics endpoint on a separate port
start_http_server(8000)

def simulate_resource_metrics():
    """Background thread to simulate changing resource metrics."""
    global memory_usage, cpu_usage
    
    while True:
        # Normal fluctuations
        memory_usage += random.uniform(-5, 5)
        if memory_usage < 100:
            memory_usage = 100
            
        cpu_usage += random.uniform(-3, 3)
        if cpu_usage < 10:
            cpu_usage = 10
        if cpu_usage > 90:
            cpu_usage = 90
            
        # Apply failure simulations
        if SIMULATED_FAILURES["memory_leak"]:
            memory_usage += 10
            
        if SIMULATED_FAILURES["cpu_spike"]:
            cpu_usage = min(cpu_usage + 30, 100)
        
        # Update Prometheus metrics
        MEMORY_USAGE.set(memory_usage * 1024 * 1024)  # Convert to bytes
        CPU_LOAD.set(cpu_usage)
        
        time.sleep(1)

# Start the background thread
metrics_thread = threading.Thread(target=simulate_resource_metrics, daemon=True)
metrics_thread.start()

@app.route('/', methods=['GET'])
def home():
    start_time = time.time()
    
    # Determine if this is a web UI request or API request
    if request.headers.get('Accept', '').find('text/html') != -1:
        return render_template('index.html')
    
    # Simulate high latency if enabled
    if SIMULATED_FAILURES["high_latency"]:
        time.sleep(random.uniform(0.5, 2))
    
    # Simulate errors if enabled
    if SIMULATED_FAILURES["error_rate"] and random.random() < 0.3:
        REQUEST_COUNT.labels(method='GET', endpoint='/', status='500').inc()
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(method='GET', endpoint='/').observe(request_time)
        return jsonify({"error": "Internal Server Error"}), 500
    
    # Normal response
    response = {
        "status": "ok",
        "message": "Welcome to the Kubernetes Failure Prediction System demo app",
        "time": time.time(),
        "memory_usage_mb": round(memory_usage, 2),
        "cpu_usage_percent": round(cpu_usage, 2)
    }
    
    request_time = time.time() - start_time
    REQUEST_LATENCY.labels(method='GET', endpoint='/').observe(request_time)
    REQUEST_COUNT.labels(method='GET', endpoint='/', status='200').inc()
    
    return jsonify(response)

@app.route('/api/data')
def get_data():
    start_time = time.time()
    
    # Simulate high latency if enabled
    if SIMULATED_FAILURES["high_latency"]:
        time.sleep(random.uniform(0.5, 3))
    
    # Simulate errors if enabled
    if SIMULATED_FAILURES["error_rate"] and random.random() < 0.4:
        REQUEST_COUNT.labels(method='GET', endpoint='/api/data', status='500').inc()
        request_time = time.time() - start_time
        REQUEST_LATENCY.labels(method='GET', endpoint='/api/data').observe(request_time)
        return jsonify({"error": "Data retrieval failed"}), 500
    
    # Generate some random data
    data_points = [random.random() * 100 for _ in range(10)]
    response = {
        "data": data_points,
        "count": len(data_points),
        "timestamp": time.time()
    }
    
    request_time = time.time() - start_time
    REQUEST_LATENCY.labels(method='GET', endpoint='/api/data').observe(request_time)
    REQUEST_COUNT.labels(method='GET', endpoint='/api/data', status='200').inc()
    
    return jsonify(response)

@app.route('/health')
def health():
    if SIMULATED_FAILURES["service_down"]:
        REQUEST_COUNT.labels(method='GET', endpoint='/health', status='503').inc()
        return jsonify({"status": "unhealthy"}), 503
        
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    return jsonify({"status": "healthy"})

@app.route('/failures', methods=['POST'])
def control_failures():
    """Endpoint to control failure simulations."""
    data = request.json
    response = {"status": "updated", "failures": {}}
    
    for failure_type, value in data.items():
        if failure_type in SIMULATED_FAILURES:
            SIMULATED_FAILURES[failure_type] = bool(value)
            response["failures"][failure_type] = SIMULATED_FAILURES[failure_type]
    
    return jsonify(response)

@app.route('/failures', methods=['GET'])
def get_failures():
    """Get current failure simulation status."""
    return jsonify({"failures": SIMULATED_FAILURES})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 