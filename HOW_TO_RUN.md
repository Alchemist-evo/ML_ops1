# How to Run This Project - Step by Step

This guide takes you from an empty machine to the full system running: training, experiment tracking, the API in Docker, the Kubernetes deployment and the monitoring dashboard. Each step says what it does and what you should see.

**Author:** Prajwal Shetty K P (BITS ID: 2025AE05434)

## Contents

0. [What you will run](#0-what-you-will-run)
1. [Prerequisites](#1-prerequisites)
2. [Get the code and set up Python](#2-get-the-code-and-set-up-python)
3. [Get and clean the data](#3-get-and-clean-the-data)
4. [Exploratory data analysis](#4-exploratory-data-analysis)
5. [Train the models](#5-train-the-models)
6. [View experiments in MLflow](#6-view-experiments-in-mlflow)
7. [Run tests and lint](#7-run-tests-and-lint)
8. [Make a prediction from the command line](#8-make-a-prediction-from-the-command-line)
9. [Run the API locally](#9-run-the-api-locally)
10. [Run the API in Docker](#10-run-the-api-in-docker)
11. [Deploy to Kubernetes (Minikube)](#11-deploy-to-kubernetes-minikube)
12. [Turn on monitoring (Prometheus + Grafana)](#12-turn-on-monitoring-prometheus--grafana)
13. [CI/CD on GitHub Actions](#13-cicd-on-github-actions)
14. [Rebuild the report](#14-rebuild-the-report)
15. [Clean up](#15-clean-up)
16. [Troubleshooting](#16-troubleshooting)

---

## 0. What you will run

```
UCI data -> clean -> train (LogReg + RandomForest, MLflow) -> models/model.joblib
   -> FastAPI (/predict, /health, /metrics) -> Docker image -> Kubernetes (2 pods + Service)
   -> Prometheus scrapes /metrics -> Grafana dashboard
```

You can stop after any step; each one works on its own once its prerequisites are met.

| Steps | You need | Time |
|---|---|---|
| 2-9 (data, training, tests, local API) | Python only | about 5 min |
| 10 (Docker) | + Docker | about 3 min |
| 11-12 (Kubernetes, monitoring) | + Minikube, kubectl | about 10 min first time (image downloads) |

## 1. Prerequisites

| Tool | Needed for | Check with |
|---|---|---|
| Python 3.14 (the version this project was built and tested with) | steps 2-9 | `python --version` |
| Git | cloning | `git --version` |
| Docker | steps 10-12 | `docker --version` |
| Minikube | steps 11-12 | `minikube version` |
| kubectl | steps 11-12 | `kubectl version --client` |
| Chromium or Chrome | step 14 only | `chromium --version` |

**Install on Arch Linux** (the environment used for this project):

```bash
sudo pacman -S --needed docker minikube kubectl
sudo systemctl enable --now docker
sudo usermod -aG docker $USER      # then log out and back in
```

**Install on Ubuntu/Debian:** install Docker from the official Docker docs, then Minikube and kubectl from their official install pages; add yourself to the `docker` group the same way.

**Install on macOS/Windows:** use Docker Desktop (you can enable its built-in Kubernetes instead of Minikube; the manifests in `k8s/` work on either).

Verify Docker works **without sudo**:

```bash
docker run --rm hello-world
```

If this says "permission denied", see [Troubleshooting](#16-troubleshooting).

## 2. Get the code and set up Python

```bash
git clone https://github.com/Alchemist-evo/ML_ops1.git
cd ML_ops1

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Expected:** `pip` finishes without errors. All versions are pinned, so you get the exact environment the project was built and tested with.

All following commands are run from the repository root (`ML_ops1/`) with the virtual environment active.

## 3. Get and clean the data

```bash
python -m src.download_data
python -m src.data_processing
```

**What happens**

1. `download_data` fetches the UCI Cleveland heart-disease file and saves it, with column names, to `data/raw/heart_disease_raw.csv`.
2. `data_processing` replaces `?` with missing values, fills them with the column median (6 cells), converts the target to 0/1 and writes `data/processed/heart_disease_clean.csv`.

**Expected output**

```
Saved raw data to data/raw/heart_disease_raw.csv
Clean dataset: (303, 14) -> data/processed/heart_disease_clean.csv
```

The cleaned data is also committed, so you can skip this step, but running it proves the pipeline works from scratch.

## 4. Exploratory data analysis

```bash
python -m src.eda
```

Saves five plots to `reports/figures/`: `class_balance.png`, `histograms_numeric.png`, `categorical_vs_target.png`, `correlation_heatmap.png` and `boxplots_numeric.png`, and prints summary statistics plus the strongest correlations with the target.

**Expected:** class balance about 54% / 46%, with `thal`, `ca`, `exang`, `oldpeak`, `thalach` and `cp` among the top correlations.

## 5. Train the models

```bash
python -m src.train
```

**What happens**

1. Splits the data 80/20 (stratified, seed 42).
2. For Logistic Regression (8 candidates) and Random Forest (18 candidates), runs `GridSearchCV` with 5-fold stratified cross-validation on the training set.
3. Cross-validates the best of each, then scores it on the test set.
4. Logs everything to MLflow (parameters, metrics, plots, model, and one nested run per candidate).
5. Saves the model with the best CV ROC-AUC to `models/model.joblib`, with `models/model_metadata.json`.

Takes roughly 10-30 seconds.

**Expected output (abridged)**

```
=== logistic_regression ===
best params: {'clf__C': 0.1, 'clf__class_weight': 'balanced'}
CV (5-fold, train): {... 'roc_auc': '0.903±0.017'}
Hold-out test    : {'accuracy': 0.885, 'precision': 0.839, 'recall': 0.929, ... 'roc_auc': 0.965}

=== random_forest ===
...
Saved best model 'logistic_regression' to models/model.joblib
```

Re-running gives the same numbers because the random seed is fixed. Each run adds a new pair of MLflow runs.

## 6. View experiments in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open <http://127.0.0.1:5000>, choose the experiment **heart-disease-classification**, then **Training runs**.

- You will see 2 top-level runs (`logistic_regression`, `random_forest`) per training run; expand one to see its nested `candidate_N` runs.
- Tick both runs and use **Compare** to see parameters and metrics side by side.
- Open a run and go to **Artifacts** for the ROC curve, confusion matrix, feature importance, dataset and saved model.

Stop the UI with `Ctrl+C`. Run it from the repository root so it finds `mlflow.db` and `mlruns/`.

## 7. Run tests and lint

```bash
pytest -v
flake8 .
```

**Expected:** `22 passed` and no lint output. The tests use small synthetic data and do not need the network or the trained model, so they run in about 3 seconds.

To see the pipeline's fail-on-error behaviour, temporarily add `def test_x(): assert False` to any file in `tests/`; `pytest` now exits with code 1 (CI would fail). Remove it afterwards.

## 8. Make a prediction from the command line

```bash
python -m src.predict '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```

**Expected**

```json
{
  "prediction": 0,
  "probability_disease": 0.4598...,
  "confidence": 0.5401...
}
```

A second sample to try, which should return prediction `1` with about 0.93 probability:

```
{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}
```

**What the fields mean:** `sex` 1=male; `cp` chest-pain type 1-4; `fbs` fasting blood sugar above 120; `restecg` 0-2; `exang` exercise angina; `slope` 1-3; `ca` blocked vessels 0-4; `thal` 3=normal, 6=fixed defect, 7=reversible defect.

## 9. Run the API locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl localhost:8000/health
# {"status":"ok","model":"logistic_regression"}

curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,
  "thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
# {"prediction":1,"label":"disease","probability_disease":0.9286,"confidence":0.9286}

curl -i -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":67}'
# HTTP/1.1 422  (missing fields are rejected)

curl localhost:8000/metrics | head        # Prometheus metrics
```

Open <http://localhost:8000/docs> for the interactive Swagger UI (click **POST /predict -> Try it out**). The server logs one line per request. Stop with `Ctrl+C`.

## 10. Run the API in Docker

```bash
docker build -t heart-disease-api:1.2.0 .
docker run -d --name hd-api -p 8000:8000 heart-disease-api:1.2.0
```

Wait a few seconds, then repeat the `curl` commands from step 9; the results are identical.

Useful commands:

```bash
docker ps                      # STATUS shows "healthy" after ~30 s
docker logs hd-api             # request log
docker stop hd-api && docker rm hd-api
```

**Notes**

- The image contains only `src/` and `models/`, so **run step 5 first** to create `models/model.joblib` (it is also committed to the repository).
- If you retrain, rebuild the image so the container picks up the new model.
- It runs as a non-root user and installs only the slim `requirements-api.txt`.
- Port 8000 busy? Use `-p 8080:8000` and call `localhost:8080`.

## 11. Deploy to Kubernetes (Minikube)

### Option A - one command

```bash
scripts/deploy_minikube.sh
```

It starts Minikube if needed, builds the image with tag `1.2.0`, loads it into the cluster, applies `k8s/`, waits for the rollout and prints the API URL.

### Option B - manual steps

```bash
# 1. Start a local cluster
minikube start --driver=docker --cpus=2 --memory=3072

# 2. Build the image and load it into the cluster (Minikube cannot see your local Docker images otherwise)
docker build -t heart-disease-api:1.2.0 .
minikube image load heart-disease-api:1.2.0

# 3. Deploy the Deployment (2 replicas) and the LoadBalancer Service
kubectl apply -f k8s/
kubectl rollout status deployment/heart-disease-api

# 4. Check
kubectl get pods,svc
```

**Expected:** two pods `1/1 Running`, and a service `heart-disease-api` of type `LoadBalancer`.

### Call the API

```bash
minikube service heart-disease-api --url
```

This prints a URL such as `http://192.168.49.2:31706`. The port number is assigned by Kubernetes and can differ on your machine. Use it in place of `localhost:8000`:

```bash
URL=$(minikube service heart-disease-api --url | head -1)
curl $URL/health
curl -X POST $URL/predict -H 'Content-Type: application/json' -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
```

Check that requests are shared across both pods:

```bash
kubectl logs -l app=heart-disease-api --prefix --tail=5
```

**About `EXTERNAL-IP <pending>`:** on Minikube a `LoadBalancer` gets an external IP only while `minikube tunnel` is running in a separate terminal (it asks for your sudo password). Without it the service still works through the URL above. Managed clouds (GKE/EKS/AKS) allocate the address automatically.

### Updating the image later

`minikube image load` **does not overwrite an existing tag**, so pods would silently keep running the old code. Always use a new tag:

```bash
docker build -t heart-disease-api:1.3.0 .
minikube image load heart-disease-api:1.3.0
# edit the image: line in k8s/deployment.yaml to 1.3.0, then:
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
```

## 12. Turn on monitoring (Prometheus + Grafana)

```bash
kubectl apply -k k8s/monitoring
kubectl rollout status deployment/prometheus
kubectl rollout status deployment/grafana
```

The first run downloads the Prometheus and Grafana images, which can take a few minutes.

Find the cluster IP and open the tools:

```bash
minikube ip                       # e.g. 192.168.49.2
```

| Tool | URL |
|---|---|
| Prometheus | `http://<minikube-ip>:30090` (check **Status -> Targets**: both API pods `UP`) |
| Grafana | `http://<minikube-ip>:30300/d/heart-api` (dashboard "Heart Disease API"; no login needed to view) |

Grafana's rate and latency panels only show data while requests arrive, so generate some traffic:

```bash
URL=$(minikube service heart-disease-api --url | head -1)
for i in $(seq 1 300); do
  curl -s -o /dev/null -X POST $URL/predict -H 'Content-Type: application/json' \
    -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
  sleep 0.2
done
# a few invalid requests, to populate the error panel
for i in 1 2 3 4 5; do curl -s -o /dev/null -X POST $URL/predict -H 'Content-Type: application/json' -d '{"age":1}'; done
```

Set the Grafana time range to **Last 5 minutes**. You should see request rate, p50/p95 latency (tens of milliseconds), predictions by class, the error ratio, total requests, average confidence and healthy pods (2).

Try the metrics directly in Prometheus (Graph tab):

```
sum by (endpoint, status) (rate(api_requests_total[1m]))
histogram_quantile(0.95, sum by (le) (rate(api_request_latency_seconds_bucket{endpoint="/predict"}[1m])))
sum by (label) (model_predictions_total)
```

## 13. CI/CD on GitHub Actions

The workflow `.github/workflows/ci.yml` runs automatically on every push and pull request. To use it on your own copy:

1. Fork or push this repository to your GitHub account.
2. Open the **Actions** tab. Each push produces a run with two jobs:
   - **build-test-train:** install, flake8, download data, pytest, train, upload artifacts (model, test report, training log, MLflow store).
   - **docker:** build the image, run the container, check `/health`, and call `/predict`.
3. Open a run to see the logs. Download the `ml-pipeline-artifacts` zip at the bottom of the run page.

Any failing step (lint error, failing test, training exception, broken container) turns the run red.

You can also run the same checks locally before pushing: `flake8 . && pytest` (step 7).

## 14. Rebuild the report

```bash
python scripts/make_architecture_diagram.py   # reports/figures/architecture.png
python scripts/build_report.py                # reports/report.md -> reports/report.pdf
```

The PDF is produced by rendering the Markdown to HTML and printing it with headless Chromium/Chrome, so one of them must be installed.

## 15. Clean up

```bash
# Kubernetes
kubectl delete -k k8s/monitoring
kubectl delete -f k8s/
minikube stop            # pause the cluster (keeps it for later)
minikube delete          # remove the cluster completely

# Docker
docker rm -f hd-api
docker image rm heart-disease-api:1.2.0
```

The Minikube cluster uses roughly 3 GB of RAM while running, so stop it when you are done.

## 16. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `permission denied while trying to connect to the docker API` | Your user is not in the `docker` group, or the group is not active yet. Run `sudo usermod -aG docker $USER`, then **log out and back in** (or run `newgrp docker` in the current terminal). |
| `Cannot connect to the Docker daemon` | Docker is not running: `sudo systemctl start docker`. |
| `FileNotFoundError: models/model.joblib` when starting the API | The model has not been trained yet. Run `python -m src.train` first. |
| `FileNotFoundError` for `data/processed/...` in training | Run the two commands from step 3 first. |
| `mlflow ui` shows no experiments | Start it from the repository root and pass `--backend-store-uri sqlite:///mlflow.db`; run `python -m src.train` at least once. |
| `{"detail":"Not Found"}` in the browser | You opened a path with no route. Use `/docs` (Swagger UI), `/health`, or `POST /predict`. (`/` now redirects to `/docs`; an older image or server started before this change does not.) Also check the port you started uvicorn on: it is `--port 8000`, not 800. |
| `422 Unprocessable Entity` from `/predict` | Expected for missing or out-of-range fields. Send all 13 features (see step 8 for allowed values). |
| Port 8000 or 5000 already in use | Pick another port: `uvicorn ... --port 8001`, `docker run -p 8080:8000 ...`, `mlflow ui --port 5001`. |
| Kubernetes pods `ImagePullBackOff` / `ErrImageNeverPull` | The image is not inside the cluster. Run `minikube image load heart-disease-api:1.2.0` and make sure the tag matches `k8s/deployment.yaml`. |
| Pods run but `/metrics` returns 404 or code seems old | Stale image: `minikube image load` never overwrites an existing tag. Build a new tag, load it and update the manifest (see step 11, "Updating the image later"). |
| Prometheus target `DOWN` with 404 | Same stale-image cause as above. |
| Service `EXTERNAL-IP` stays `<pending>` | Normal on Minikube; use `minikube service heart-disease-api --url`, or run `minikube tunnel` in another terminal. |
| `minikube tunnel` asks for a password / fails | It needs sudo to add a route. Use the `--url` method instead. |
| Grafana panels empty | Send traffic (step 12) and set the range to Last 5 minutes; check that Prometheus targets are `UP`. |
| Grafana or Prometheus stuck in `ContainerCreating` | Images are still downloading; check with `kubectl describe pod <name>`. |
| `pip install` fails on an older Python | The pins were resolved on Python 3.14 and newer library versions may not support older Pythons. Use 3.14, or loosen the pins in `requirements.txt`. |
| CI fails at flake8 | Run `flake8 .` locally and fix what it reports; the limit is 100 characters per line (see `.flake8`). |

---

**Quick reference**

```bash
# everything, from a fresh clone
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m src.download_data && python -m src.data_processing && python -m src.train
pytest && flake8 .
scripts/deploy_minikube.sh && kubectl apply -k k8s/monitoring
```
