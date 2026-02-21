#!/usr/bin/env bash
# Deploy web dashboard to Google Cloud Run
# Usage: ./scripts/deploy-cloudrun.sh [PROJECT_ID] [REGION]
#
# Prerequisites:
#   gcloud CLI installed and authenticated (gcloud auth login)
#   Docker installed (or use --source flag for Cloud Build)

set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${2:-us-central1}"
SERVICE_NAME="tropicalcycloneml-web"
AR_REPO="tropicalcycloneml"

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No project ID. Pass as argument or set via: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "=== Deploying TropicalCycloneML Web Dashboard ==="
echo "  Project:  $PROJECT_ID"
echo "  Region:   $REGION"
echo "  Service:  $SERVICE_NAME"
echo ""

# Ensure Artifact Registry repo exists
echo "--- Ensuring Artifact Registry repo exists..."
gcloud artifacts repositories describe "$AR_REPO" \
    --location="$REGION" \
    --project="$PROJECT_ID" 2>/dev/null || \
gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --project="$PROJECT_ID"

# Configure docker auth
echo "--- Configuring Docker authentication..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:latest"

# Build
echo "--- Building container image..."
docker build -t "$IMAGE" ./web

# Push
echo "--- Pushing to Artifact Registry..."
docker push "$IMAGE"

# Deploy
echo "--- Deploying to Cloud Run..."
gcloud run deploy "$SERVICE_NAME" \
    --image="$IMAGE" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --memory=256Mi \
    --cpu=1 \
    --min-instances=0 \
    --max-instances=3

# Get URL
URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format='value(status.url)')

echo ""
echo "=== Deployed successfully ==="
echo "  Live at: $URL"
