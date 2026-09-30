#!/usr/bin/env bash
# Build the image, load it into Minikube and deploy the manifests in k8s/.
set -euo pipefail

minikube status >/dev/null 2>&1 || minikube start --driver=docker --cpus=2 --memory=3072
# Use an explicit version tag: `minikube image load` does not overwrite an existing tag.
IMAGE_TAG="${IMAGE_TAG:-1.2.0}"
docker build -t "heart-disease-api:${IMAGE_TAG}" -t heart-disease-api:latest .
minikube image load "heart-disease-api:${IMAGE_TAG}"
kubectl apply -f k8s/
kubectl rollout status deployment/heart-disease-api --timeout=180s
kubectl get pods,svc
echo "API URL: $(minikube service heart-disease-api --url | head -1)"
