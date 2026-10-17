#!/usr/bin/env bash
# Build the XPU image, push it, replace the hal Job, and stream its log.
set -euo pipefail

if [[ -z "${REGISTRY:-}" ]]; then
  echo "REGISTRY is required and has no default host" >&2
  exit 1
fi

IMAGE="${IMAGE:-$REGISTRY/sgl-kernel-xpu-gptq}"
TAG="${TAG:-dev}"
MODE="${MODE:-smoke}"
STAGE="${STAGE:-baseline}"
BASELINE_TPS="${BASELINE_TPS:-}"
export IMAGE TAG MODE STAGE BASELINE_TPS

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

docker build -f Dockerfile.xpu -t "$IMAGE:$TAG" .
docker push "$IMAGE:$TAG"

kubectl delete job sgl-gptq-xpu-test --ignore-not-found

rendered="$(mktemp)"
python3 - "$rendered" <<'PY'
import os
import sys
from pathlib import Path

text = Path("deploy/k8s/test-job.yaml").read_text()
for key in ("IMAGE", "TAG", "MODE", "STAGE", "BASELINE_TPS", "SGL_GPTQ_PROFILE", "MAX_NEW_TOKENS", "SGL_GPTQ_STEP"):
    text = text.replace("${" + key + "}", os.environ.get(key, ""))
Path(sys.argv[1]).write_text(text)
PY
kubectl apply -f "$rendered"
rm -f "$rendered"

for _ in $(seq 1 120); do
  if kubectl logs job/sgl-gptq-xpu-test >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

kubectl logs -f job/sgl-gptq-xpu-test

for _ in $(seq 1 30); do
  ok="$(kubectl get job sgl-gptq-xpu-test -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
  bad="$(kubectl get job sgl-gptq-xpu-test -o jsonpath='{.status.failed}' 2>/dev/null || true)"
  if [[ "${ok}" == "1" ]]; then
    exit 0
  fi
  if [[ -n "${bad}" && "${bad}" != "0" ]]; then
    echo "job sgl-gptq-xpu-test failed" >&2
    exit 1
  fi
  sleep 1
done

echo "job sgl-gptq-xpu-test did not succeed" >&2
exit 1
