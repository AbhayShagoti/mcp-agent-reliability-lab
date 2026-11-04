#!/usr/bin/env bash
# Load the local image into kind and wait until /health responds.
set -euo pipefail

CLUSTER="${CLUSTER:-arl}"
kubectl apply -k deploy/k8s/overlays/kind
kubectl rollout status deployment/arl-api --timeout=180s
kubectl port-forward svc/arl-api 8080:8000 >/tmp/arl-port-forward.log 2>&1 &
PF=$!
trap 'kill ${PF} >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8080/health; then
    echo
    curl -sf http://127.0.0.1:8080/metrics | head -n 5
    exit 0
  fi
  sleep 2
done
echo "kind smoke test did not become healthy" >&2
exit 1
