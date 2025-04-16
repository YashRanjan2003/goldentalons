# Kubernetes Failure Prediction System

This project implements a machine learning-based system for predicting potential failures in Kubernetes clusters by analyzing various system metrics. Instead of relying on synthetic or pre-existing datasets, this project creates a real-world web application that can be deployed on Kubernetes, simulate various failure scenarios, and generate real metrics data to train machine learning models.

## Project Components

1. **Demo Web Application**: A Flask web application with a dashboard for visualizing metrics and simulating failures
2. **Kubernetes Deployment**: Configuration files for deploying the application and monitoring tools on Kubernetes
3. **Metrics Collection**: Scripts to collect system metrics during normal operation and simulated failures
4. **Machine Learning Models**: Implementation of multiple models (Random Forest, XGBoost, Neural Network) to predict failures
5. **Prediction Service**: A service that continuously monitors cluster metrics and predicts potential failures in real-time

## Architecture

![System Architecture](https://example.com/architecture.png)

The system is structured as follows:

- **Web Application**: A Flask application with Prometheus metrics and failure simulation endpoints
- **Prometheus**: Collects and stores metrics from the application and Kubernetes cluster
- **Metrics Collector**: Script that simulates various failure scenarios and collects metrics
- **ML Training Pipeline**: Trains and evaluates different ML models using the collected metrics
- **Prediction Service**: Continuously monitors metrics and predicts potential failures

## Getting Started

### Prerequisites

- Docker
- Kubernetes cluster (can be Minikube or kind for local development)
- Python 3.8+
- kubectl

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/k8s-failure-prediction.git
cd k8s-failure-prediction
```

2. Build the Docker image:
```bash
docker build -t k8s-failure-demo:latest .
```

3. Deploy the application to Kubernetes:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

4. Deploy Prometheus for metrics collection:
```bash
kubectl apply -f k8s/prometheus-config.yaml
kubectl apply -f k8s/prometheus.yaml
```

5. Install required Python packages for the scripts:
```bash
pip install -r scripts/requirements.txt
```

## Usage

### Accessing the Application

Once deployed, you can access the web dashboard at:
```
http://<kubernetes-service-ip>
```

### Simulating Failures

The web dashboard provides toggles to simulate various failure scenarios:

1. **High Latency**: Introduces artificial delays in request processing
2. **Memory Leak**: Simulates gradually increasing memory usage
3. **CPU Spike**: Causes high CPU utilization
4. **Error Rate**: Introduces random errors in API responses
5. **Service Down**: Makes the health endpoint return unhealthy status

You can also use the API directly:
```bash
# Enable memory leak simulation
curl -X POST http://<service-ip>/failures -H "Content-Type: application/json" -d '{"memory_leak": true}'

# Enable multiple failures
curl -X POST http://<service-ip>/failures -H "Content-Type: application/json" -d '{"high_latency": true, "error_rate": true}'

# Check current failure configuration
curl http://<service-ip>/failures
```

### Collecting Metrics Data

To collect metrics during normal operation and simulated failures:

```bash
python scripts/collect_metrics.py --app-host <service-ip> --prom-host <prometheus-ip> --duration 300 --output metrics_data.csv
```

This will:
1. Run through different failure scenarios (normal, memory leak, CPU spike, etc.)
2. Collect metrics from Prometheus during each scenario
3. Save the labeled dataset to a CSV file for training

### Training Models

To train machine learning models on the collected data:

```bash
python scripts/train_model.py --input metrics_data.csv --output-dir models
```

This script will:
1. Preprocess the metrics data
2. Train multiple models (Random Forest, XGBoost, Neural Network)
3. Evaluate the models and select the best performer
4. Save the models and related artifacts to the specified directory

### Running the Prediction Service

To run the prediction service that continuously monitors metrics and predicts failures:

```bash
python scripts/prediction_service.py --prom-host <prometheus-ip> --model-dir models --model-type rf
```

This service will:
1. Load the trained model
2. Continuously collect metrics from Prometheus
3. Make real-time predictions about potential failures
4. Provide an API endpoint for querying current predictions

You can access the prediction API at:
```
http://<prediction-service-ip>:8080/prediction
```

## Development

### Project Structure

```
├── app/                        # Demo web application
│   ├── app.py                  # Flask application with metrics and failure simulation
│   └── templates/              # HTML templates for the web dashboard
├── k8s/                        # Kubernetes configuration files
│   ├── deployment.yaml         # Deployment for the demo app
│   ├── service.yaml            # Service for the demo app
│   ├── prometheus-config.yaml  # Prometheus configuration
│   └── prometheus.yaml         # Prometheus deployment
├── scripts/                    # Utility scripts
│   ├── collect_metrics.py      # Script to collect metrics and simulate failures
│   ├── train_model.py          # Script to train and evaluate ML models
│   └── prediction_service.py   # Service for real-time failure prediction
├── Dockerfile                  # Dockerfile for the demo app
├── requirements.txt            # Python dependencies for the demo app
└── README.md                   # This README file
```

### Extending the Project

To add new failure scenarios:
1. Add a new failure type to the `SIMULATED_FAILURES` dictionary in `app/app.py`
2. Implement the simulation logic in the relevant endpoint handlers
3. Add a new toggle in the web dashboard (`app/templates/index.html`)
4. Update the failure scenarios in `scripts/collect_metrics.py`

To add new metrics:
1. Add new Prometheus metrics in `app/app.py`
2. Update the metric queries in the collection scripts

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Prometheus and Kubernetes communities
- scikit-learn, XGBoost, and TensorFlow for machine learning tools 