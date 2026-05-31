# CRAG Pipeline Deployment Guide

## Local Testing

### Build the Docker image locally:
```bash
docker build -t crag-pipeline:latest .
```

### Run locally:
```bash
docker run -p 5000:5000 crag-pipeline:latest
```

### Test the API:
```bash
# Health check
curl http://localhost:5000/health

# Retrieve with CRAG
curl -X POST http://localhost:5000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of vitamin D deficiency?",
    "documents": [
      {"id": "doc1", "text": "Vitamin D deficiency can cause fatigue, bone pain, and muscle weakness."},
      {"id": "doc2", "text": "Vitamins are essential nutrients required in small amounts."},
      {"id": "doc3", "text": "The Amazon rainforest produces 20% of the world oxygen."}
    ],
    "top_k": 3
  }'

# View benchmarks
curl http://localhost:5000/benchmark
```

## Google Cloud Run Deployment

### Prerequisites:
```bash
# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable services
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### Deploy to Cloud Run:
```bash
# Deploy from source (recommended)
gcloud run deploy crag-pipeline \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 3600 \
  --allow-unauthenticated

# Or build and push image manually:
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/crag-pipeline
gcloud run deploy crag-pipeline \
  --image gcr.io/YOUR_PROJECT_ID/crag-pipeline \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated
```

### Get the service URL:
```bash
gcloud run services describe crag-pipeline --platform managed --region us-central1 --format='value(status.url)'
```

### Test deployed service:
```bash
SERVICE_URL=$(gcloud run services describe crag-pipeline \
  --platform managed --region us-central1 \
  --format='value(status.url)')

curl $SERVICE_URL/health

curl -X POST $SERVICE_URL/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "...", "documents": [...], "top_k": 3}'
```

## AWS Fargate Deployment (Alternative)

### Push to ECR:
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

docker tag crag-pipeline:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/crag-pipeline:latest

docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/crag-pipeline:latest
```

### Deploy with ECS:
Use AWS console or:
```bash
aws ecs create-service \
  --cluster production \
  --service-name crag-pipeline \
  --task-definition crag-pipeline:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}
```

## Performance Characteristics

**Latency profile (first request with model downloads):**
- Cold start: ~60-90 seconds (downloading sentence-transformers models)
- Warm start: ~50-100ms

**Latency profile (steady state, documents in matrix):**
- Matrix retrieval: 0.08ms per query
- Cross-encoder evaluation: 7-28ms per chunk (batched)
- Total pipeline: ~30-50ms

**Memory:**
- Base: ~2GB (Flask + models loaded)
- Per document in matrix: ~6KB (384 dims, float32)
- 3633 documents = ~22MB additional

**Recommended Cloud Run settings:**
- Memory: 4GB (allows larger matrices, faster inference)
- CPU: 2 (enables parallel evaluation)
- Timeout: 3600 seconds (1 hour, for large ingestions)

## Environment Variables

If using Bedrock (requires AWS credentials):
```bash
gcloud run deploy crag-pipeline \
  --set-env-vars="AWS_REGION=us-east-1" \
  --update \
  # Then set secrets:
gcloud secrets create aws-access-key --data-file=<(echo $AWS_ACCESS_KEY_ID)
gcloud secrets create aws-secret-key --data-file=<(echo $AWS_SECRET_ACCESS_KEY)
gcloud run deploy crag-pipeline \
  --update-secrets=AWS_ACCESS_KEY_ID=aws-access-key:latest,AWS_SECRET_ACCESS_KEY=aws-secret-key:latest
```

## Monitoring

### View logs:
```bash
gcloud run logs read crag-pipeline --limit 50
```

### Set up Cloud Monitoring alerts:
```bash
# Create alert policy in Cloud Console:
# Monitoring → Alerting → Create Policy
# Condition: Cloud Run → Error rate > 5%
```

## Cost Estimation (Google Cloud Run)

- 2 vCPU, 4GB memory: $0.0002400 per vCPU-second
- 1000 requests per day × 0.05s average = 50 vCPU-seconds/day
- Monthly: ~1500 vCPU-seconds ≈ **$0.36/month**
- Plus storage: negligible for metadata
- Plus data egress: ~$0.12 per GB (shared ingress is free)

**Extremely cost-effective for production RAG workloads.**