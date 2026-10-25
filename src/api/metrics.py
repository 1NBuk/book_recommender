from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "bookrec_http_requests_total",
    "HTTP requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "bookrec_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)
INFLIGHT = Gauge("bookrec_http_requests_in_flight", "Requests currently being processed")

RECOMMENDATIONS = Counter(
    "bookrec_recommendations_total",
    "Recommendation lists served, by source",
    ["source"],
)

ARTIFACT_USERS = Gauge("bookrec_artifact_users", "Users with precomputed recommendations")
ARTIFACT_ITEMS = Gauge("bookrec_artifact_items", "Books in the catalog")
ARTIFACTS_LOADED_AT = Gauge("bookrec_artifacts_loaded_timestamp_seconds", "Unix time when artifacts were loaded")
MODEL_CREATED_AT = Gauge("bookrec_model_created_timestamp_seconds", "Unix time when artifacts were exported")
