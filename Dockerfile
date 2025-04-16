FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ .

# Expose the app port and metrics port
EXPOSE 5000 8000

# Run the application
CMD ["python", "app.py"] 