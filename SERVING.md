# Serving: API, Docker, k8s, monitoring, CI

Quick start
    make install && make test
    make run                      # http://localhost:8000/docs
    docker compose up -d --build  # API :8000, Prometheus :9090, Grafana :3000

Refresh artifacts after retraining:  make artifacts  (then commit artifacts/)

Endpoints: GET /v1/users/{user_id}/recommendations?n=10 | /v1/popular | /v1/books/{item_id} | /v1/model
           GET /health/live | /health/ready | /metrics
Env: BOOKREC_API_KEY (empty = no auth), BOOKREC_ARTIFACTS_DIR, BOOKREC_DEFAULT_N, BOOKREC_MAX_N, BOOKREC_LOG_LEVEL
