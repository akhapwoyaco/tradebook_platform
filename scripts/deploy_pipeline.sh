#!/bin/bash

# deploy_synthetic_data_api.sh
# This script orchestrates the deployment of the Synthetic Data API Server component
# of the Tradebook Pipeline to a Kubernetes cluster.
# It requires Docker and kubectl to be installed and configured.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration with defaults and argument parsing ---
DOCKER_IMAGE_NAME="tradebook_pipeline_gpu"
DOCKER_TAG="latest" # Default tag. Use -t option to override (e.g., -t "v1.0.0" or -t "$(git rev-parse --short HEAD)")
K8S_MANIFEST="synthetic_data/k8s-deployment.yaml"
DOCKER_REGISTRY_PREFIX="" # Optional: e.g., "your_registry_username/" or "gcr.io/your-project/"

# Parse command line arguments
while getopts "t:m:r:" opt; do
  case $opt in
    t) DOCKER_TAG="$OPTARG" ;;
    m) K8S_MANIFEST="$OPTARG" ;;
    r) # Ensure registry prefix ends with a slash if provided
       if [[ "$OPTARG" =~ /$ ]]; then
         DOCKER_REGISTRY_PREFIX="$OPTARG"
       else
         DOCKER_REGISTRY_PREFIX="$OPTARG/"
       fi
       ;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1 ;;
  esac
done

# Construct the full Docker image name including registry prefix and tag
FULL_DOCKER_IMAGE_NAME="${DOCKER_REGISTRY_PREFIX}${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"

echo "--- Starting Tradebook Pipeline (Synthetic Data API) deployment ---"
echo "Full Docker Image: ${FULL_DOCKER_IMAGE_NAME}"
echo "Kubernetes Manifest: ${K8S_MANIFEST}"

# --- Pre-deployment Checks ---
echo -e "\n--- Pre-deployment Checks ---"
command -v docker >/dev/null 2>&1 || { echo >&2 "ERROR: Docker is not installed or not in PATH. Aborting."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo >&2 "ERROR: kubectl is not installed or not in PATH. Aborting."; exit 1; }

# Optional: Check kubectl context (uncomment for stricter environments)
# CURRENT_K8S_CONTEXT=$(kubectl config current-context 2>/dev/null)
# if [ -z "$CURRENT_K8S_CONTEXT" ]; then
#   echo "WARNING: No Kubernetes context is currently set. Please ensure kubectl is configured to the correct cluster."
# else
#   echo "Kubernetes context: ${CURRENT_K8S_CONTEXT}"
# fi
echo "Pre-checks complete."

# --- Step 1: Build Docker Image for Synthetic Data API Server ---
echo -e "\n--- Step 1: Building Docker Image ---"
echo "Building Docker image: ${FULL_DOCKER_IMAGE_NAME} from ${PWD}/synthetic_data/Dockerfile.gpu"
docker build -t "${FULL_DOCKER_IMAGE_NAME}" -f synthetic_data/Dockerfile.gpu .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker image build failed! Please check your Dockerfile and context."
    exit 1
fi
echo "Docker image built successfully."

# --- Step 2: (Optional) Push Docker Image to a Container Registry ---
# This step is executed only if a DOCKER_REGISTRY_PREFIX is provided.
if [ -n "${DOCKER_REGISTRY_PREFIX}" ]; then
  echo -e "\n--- Step 2: Pushing Docker Image to Registry ---"
  echo "Ensure you are logged into your Docker registry (e.g., 'docker login ${DOCKER_REGISTRY_PREFIX%/*}')"
  echo "Pushing Docker image: ${FULL_DOCKER_IMAGE_NAME}"
  docker push "${FULL_DOCKER_IMAGE_NAME}"
  if [ $? -ne 0 ]; then
      echo "ERROR: Docker image push failed! Please check your registry login and permissions."
      exit 1
  fi
  echo "Docker image pushed to registry successfully."
else
  echo -e "\n--- Step 2: Skipping Docker Image Push ---"
  echo "No Docker registry prefix provided. If deploying to a remote cluster, provide a registry using -r option."
fi

# --- Step 3: Deploy to Kubernetes ---
echo -e "\n--- Step 3: Applying Kubernetes Manifests ---"
echo "Applying Kubernetes manifests from ${K8S_MANIFEST}..."
# Ensure kubectl is configured to the correct cluster context
kubectl apply -f "${K8S_MANIFEST}"

if [ $? -ne 0 ]; then
    echo "ERROR: Kubernetes deployment failed! Check your K8s manifest and cluster configuration."
    exit 1
fi
echo "Kubernetes deployment initiated. Use \`kubectl get pods\` to monitor status."

# --- Step 4: Verify Deployment Rollout ---
echo -e "\n--- Step 4: Verifying Deployment Rollout Status ---"
# The deployment name 'synthetic-data-api-deployment' must match the name in your K8s manifest.
echo "Waiting for 'synthetic-data-api-deployment' to be ready (timeout: 300s)..."
kubectl rollout status deployment/synthetic-data-api-deployment --timeout=300s

if [ $? -ne 0 ]; then
    echo "ERROR: Synthetic Data API server deployment did not become ready within 300 seconds!"
    echo "For troubleshooting, check pod logs and events:"
    echo "  \`kubectl get pods\`"
    echo "  \`kubectl describe pod <synthetic-data-api-deployment-pod-name>\`"
    echo "  \`kubectl logs <synthetic-data-api-deployment-pod-name>\`"
    exit 1
fi
echo "Synthetic Data API server deployment is running successfully."

# --- Step 5: Post-Deployment Information ---
echo -e "\n--- Step 5: Deployment Complete ---"
echo "Deployment of Synthetic Data API Server finished successfully."
echo "You may need to configure Ingress or port-forwarding to access the API server externally."
echo "To port-forward the API server to localhost:5000 (assuming Kubernetes service name 'synthetic-data-api-service'):"
echo "  kubectl port-forward svc/synthetic-data-api-service 5000:5000"
echo "To check service external IP (if using LoadBalancer service type):"
echo "  kubectl get svc synthetic-data-api-service"

echo -e "\n--- Deployment pipeline finished successfully ---"