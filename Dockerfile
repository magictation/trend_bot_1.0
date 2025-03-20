FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Ensure data directory exists
RUN mkdir -p data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "run.py", "trade"]