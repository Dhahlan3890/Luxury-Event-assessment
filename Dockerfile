FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Model will be downloaded from S3 at container startup
RUN mkdir -p models

ENV MODEL_DIR=models
ENV MODEL_PATH=models/final_model.pkl
ENV METRICS_PATH=models/metrics.json
ENV SCALER_PATH=models/scaler.pkl
ENV LOG_LEVEL=INFO

EXPOSE 8000

# Download model from S3 then start API
CMD python3 -c "\
import boto3, os; \
s3 = boto3.client('s3'); \
bucket = os.environ['AWS_S3_BUCKET']; \
[s3.download_file(bucket, f'models/{f}', f'models/{f}') \
 for f in ['final_model.pkl','logreg_model.pkl','scaler.pkl','metrics.json']]; \
print('Models downloaded from S3')" \
&& uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 2