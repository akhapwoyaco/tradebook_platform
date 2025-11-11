#!/bin/bash

# deploy_pipeline_data.sh
# This script orchestrates the deployment of the Pipeline Data component
# of the Tradebook Pipeline to a Kubernetes cluster.
# It requires Docker and kubectl to be installed and configured.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration with defaults and argument parsing ---
DOCKER_IMAGE_NAME="tradebook_pipeline_data"
DOCKER_TAG="latest" # Default tag. Use -t option to override (e.g., -t "v1.0.0" or -t "$(git rev-parse --short HEAD)")
K8S_MANIFEST="pipeline_data/k8s-deployment.yaml"
DOCKER_REGISTRY_PREFIX="" # Optional: e.g., "your_registry_username/" or "gcr.io/your-project/"
DEPLOYMENT_NAME="pipeline-data-deployment"
SERVICE_NAME="pipeline-data-service"
DOCKERFILE_PATH="pipeline_data/Dockerfile"

# Function to show usage
usage() {
    echo "Usage: $0 [-t TAG] [-m MANIFEST] [-r REGISTRY_PREFIX] [-d DOCKERFILE] [-n DEPLOYMENT_NAME] [-s SERVICE_NAME] [-h]"
    echo "  -t TAG              Docker tag (default: latest)"
    echo "  -m MANIFEST         Kubernetes manifest file (default: pipeline_data/k8s-deployment.yaml)"
    echo "  -r REGISTRY_PREFIX  Docker registry prefix (e.g., gcr.io/project/)"
    echo "  -d DOCKERFILE       Path to Dockerfile (default: pipeline_data/Dockerfile)"
    echo "  -n DEPLOYMENT_NAME  Kubernetes deployment name (default: pipeline-data-deployment)"
    echo "  -s SERVICE_NAME     Kubernetes service name (default: pipeline-data-service)"
    echo "  -h                  Show this help message"
    exit 1
}

# Parse command line arguments
while getopts "t:m:r:d:n:s:h" opt; do
  case $opt in
    t) DOCKER_TAG="$OPTARG" ;;
    m) K8S_MANIFEST="$OPTARG" ;;
    d) DOCKERFILE_PATH="$OPTARG" ;;
    n) DEPLOYMENT_NAME="$OPTARG" ;;
    s) SERVICE_NAME="$OPTARG" ;;
    r) # Ensure registry prefix ends with a slash if provided
       if [[ "$OPTARG" =~ /$ ]]; then
         DOCKER_REGISTRY_PREFIX="$OPTARG"
       else
         DOCKER_REGISTRY_PREFIX="$OPTARG/"
       fi
       ;;
    h) usage ;;
    \?) echo "Invalid option -$OPTARG" >&2; usage ;;
  esac
done

# Construct the full Docker image name including registry prefix and tag
FULL_DOCKER_IMAGE_NAME="${DOCKER_REGISTRY_PREFIX}${DOCKER_IMAGE_NAME}:${DOCKER_TAG}"

echo "--- Starting Tradebook Pipeline (Pipeline Data) deployment ---"
echo "Full Docker Image: ${FULL_DOCKER_IMAGE_NAME}"
echo "Dockerfile Path: ${DOCKERFILE_PATH}"
echo "Kubernetes Manifest: ${K8S_MANIFEST}"
echo "Deployment Name: ${DEPLOYMENT_NAME}"
echo "Service Name: ${SERVICE_NAME}"

# --- Pre-deployment Checks ---
echo -e "\n--- Pre-deployment Checks ---"
command -v docker >/dev/null 2>&1 || { echo >&2 "ERROR: Docker is not installed or not in PATH. Aborting."; exit 1; }
command -v kubectl >/dev/null 2>&1 || { echo >&2 "ERROR: kubectl is not installed or not in PATH. Aborting."; exit 1; }

# Check if required files exist
if [ ! -f "$DOCKERFILE_PATH" ]; then
    echo "ERROR: Dockerfile not found at ${DOCKERFILE_PATH}"
    echo "Please ensure the Dockerfile exists or specify the correct path with -d option."
    exit 1
fi

if [ ! -f "$K8S_MANIFEST" ]; then
    echo "ERROR: Kubernetes manifest not found at ${K8S_MANIFEST}"
    echo "Please ensure the manifest file exists or specify the correct path with -m option."
    exit 1
fi

# Check kubectl context
CURRENT_K8S_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "")
if [ -z "$CURRENT_K8S_CONTEXT" ]; then
  echo "WARNING: No Kubernetes context is currently set. Please ensure kubectl is configured to the correct cluster."
  read -p "Continue anyway? (y/N): " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled by user."
    exit 1
  fi
else
  echo "Kubernetes context: ${CURRENT_K8S_CONTEXT}"
fi

# Test kubectl connectivity
if ! kubectl cluster-info >/dev/null 2>&1; then
    echo "WARNING: Unable to connect to Kubernetes cluster. Please check your kubectl configuration."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Deployment cancelled by user."
        exit 1
    fi
fi

echo "Pre-checks complete."

# --- Step 1: Build Docker Image for Pipeline Data ---
echo -e "\n--- Step 1: Building Docker Image ---"
echo "Building Docker image: ${FULL_DOCKER_IMAGE_NAME} from ${PWD}/${DOCKERFILE_PATH}"

if ! docker build -t "${FULL_DOCKER_IMAGE_NAME}" -f "${DOCKERFILE_PATH}" .; then
    echo "ERROR: Docker image build failed! Please check your Dockerfile and context."
    echo "Dockerfile path: ${DOCKERFILE_PATH}"
    echo "Build context: ${PWD}"
    exit 1
fi

echo "Docker image built successfully."

# --- Step 2: (Optional) Push Docker Image to a Container Registry ---
# This step is executed only if a DOCKER_REGISTRY_PREFIX is provided.
if [ -n "${DOCKER_REGISTRY_PREFIX}" ]; then
  echo -e "\n--- Step 2: Pushing Docker Image to Registry ---"
  echo "Ensure you are logged into your Docker registry (e.g., 'docker login ${DOCKER_REGISTRY_PREFIX%/*}')"
  echo "Pushing Docker image: ${FULL_DOCKER_IMAGE_NAME}"
  
  if ! docker push "${FULL_DOCKER_IMAGE_NAME}"; then
      echo "ERROR: Docker image push failed! Please check your registry login and permissions."
      echo "Registry: ${DOCKER_REGISTRY_PREFIX%/*}"
      echo "Image: ${FULL_DOCKER_IMAGE_NAME}"
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

if ! kubectl apply -f "${K8S_MANIFEST}"; then
    echo "ERROR: Kubernetes deployment failed! Check your K8s manifest and cluster configuration."
    echo "Manifest file: ${K8S_MANIFEST}"
    echo "Current context: ${CURRENT_K8S_CONTEXT}"
    exit 1
fi

echo "Kubernetes deployment initiated. Use 'kubectl get pods' to monitor status."

# --- Step 4: Verify Deployment Rollout ---
echo -e "\n--- Step 4: Verifying Deployment Rollout Status ---"
echo "Waiting for '${DEPLOYMENT_NAME}' to be ready (timeout: 300s)..."

if ! kubectl rollout status deployment/${DEPLOYMENT_NAME} --timeout=300s; then
    echo "ERROR: Pipeline Data deployment did not become ready within 300 seconds!"
    echo "For troubleshooting, check pod logs and events:"
    echo "  kubectl get pods -l app=${DEPLOYMENT_NAME}"
    echo "  kubectl describe deployment ${DEPLOYMENT_NAME}"
    echo "  kubectl logs -l app=${DEPLOYMENT_NAME} --tail=50"
    echo "  kubectl get events --sort-by=.metadata.creationTimestamp --field-selector involvedObject.name=${DEPLOYMENT_NAME}"
    
    # Show current pod status
    echo -e "\nCurrent pod status:"
    kubectl get pods -l app=${DEPLOYMENT_NAME} 2>/dev/null || echo "No pods found with label app=${DEPLOYMENT_NAME}"
    exit 1
fi

echo "Pipeline Data deployment is running successfully."

# --- Step 5: Post-Deployment Information ---
echo -e "\n--- Step 5: Deployment Complete ---"
echo "Deployment of Pipeline Data component finished successfully."

# Check if service exists and provide connection info
if kubectl get svc "${SERVICE_NAME}" >/dev/null 2>&1; then
    echo -e "\nService Information:"
    kubectl get svc "${SERVICE_NAME}"
    
    SERVICE_TYPE=$(kubectl get svc "${SERVICE_NAME}" -o jsonpath='{.spec.type}' 2>/dev/null || echo "unknown")
    case $SERVICE_TYPE in
        "LoadBalancer")
            echo -e "\nTo get external IP (may take a few minutes):"
            echo "  kubectl get svc ${SERVICE_NAME} -w"
            echo "  # Wait for EXTERNAL-IP to be assigned"
            ;;
        "NodePort")
            NODE_PORT=$(kubectl get svc "${SERVICE_NAME}" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo "unknown")
            echo -e "\nTo access via NodePort:"
            echo "  Node Port: ${NODE_PORT}"
            echo "  kubectl get nodes -o wide  # Get node IPs"
            echo "  Access via <node-ip>:${NODE_PORT}"
            ;;
        "ClusterIP"|*)
            SERVICE_PORT=$(kubectl get svc "${SERVICE_NAME}" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "5000")
            echo -e "\nTo port-forward the service to localhost:"
            echo "  kubectl port-forward svc/${SERVICE_NAME} ${SERVICE_PORT}:${SERVICE_PORT}"
            ;;
    esac
else
    echo "WARNING: Service '${SERVICE_NAME}' not found. You may need to configure access manually."
    echo "Available services:"
    kubectl get svc 2>/dev/null || echo "Unable to list services"
fi

echo -e "\nUseful monitoring commands:"
echo "  kubectl get pods -l app=${DEPLOYMENT_NAME}"
echo "  kubectl logs -l app=${DEPLOYMENT_NAME} -f"
echo "  kubectl describe deployment ${DEPLOYMENT_NAME}"
echo "  kubectl top pods -l app=${DEPLOYMENT_NAME}  # Resource usage (if metrics-server is installed)"

# Show deployment summary
echo -e "\n--- Deployment Summary ---"
echo "✓ Docker image built: ${FULL_DOCKER_IMAGE_NAME}"
[ -n "${DOCKER_REGISTRY_PREFIX}" ] && echo "✓ Docker image pushed to registry"
echo "✓ Kubernetes manifests applied: ${K8S_MANIFEST}"
echo "✓ Deployment verified: ${DEPLOYMENT_NAME}"
echo "✓ Service configured: ${SERVICE_NAME}"

echo -e "\n--- Deployment pipeline finished successfully ---"