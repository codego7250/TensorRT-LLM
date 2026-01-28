"""Prometheus metrics for OpenAI Server."""

from prometheus_client import Counter, Gauge, Histogram, Summary

# Common labels for all metrics
COMMON_LABELS = ["model_name", "deployment", "engine_type"]

# Metric prefix for TensorRT-LLM metrics
METRIC_PREFIX = "trtllm_"

REQUEST_FAILED_TOTAL = Counter(
    "request_failed_total",
    "Total number of requests failed",
    COMMON_LABELS
)

REQUEST_SUCCESS_TOTAL = Counter(
    METRIC_PREFIX + "request_success_total",
    "Count of successfully processed requests",
    COMMON_LABELS + ["finished_reason"]
)

# Request queue state
NUM_REQUESTS_WAITING = Gauge(
    "num_requests_waiting",
    "Number of requests waiting to be processed (in queue)",
    COMMON_LABELS
)

# KV Cache usage
KV_CACHE_USAGE_PERC = Gauge(
    "kv_cache_usage_perc",
    "KV cache usage percentage (0-100)",
    COMMON_LABELS
)

# Executor metrics
NUM_REQUESTS_RUNNING = Gauge(
    "num_requests_running",
    "Number of requests currently running",
    COMMON_LABELS
)

NUM_REQUESTS_SWAPPED = Gauge(
    "num_requests_swapped",
    "Number of requests currently swapped",
    COMMON_LABELS
)

NUM_REQUESTS_CONCURRENT = Gauge(
    "num_requests_concurrent",
    "Number of concurrent requests",
    COMMON_LABELS
)

PROMPT_TOKENS_TOTAL = Counter(
    "prompt_tokens_total",
    "Total number of prompt tokens processed",
    COMMON_LABELS
)

GENERATION_TOKENS_TOTAL = Counter(
    "generation_tokens_total",
    "Total number of generation tokens produced",
    COMMON_LABELS
)

ITERATION_TOKENS_TOTAL = Summary(
    "iteration_tokens_total",
    "Total tokens processed per iteration",
    COMMON_LABELS
)

REQUEST_PROMPT_TOKENS_TOTAL = Summary(
    "request_prompt_tokens_total",
    "Prompt tokens per request",
    COMMON_LABELS
)

REQUEST_GENERATION_TOKENS_TOTAL = Summary(
    "request_generation_tokens_total",
    "Generation tokens per request",
    COMMON_LABELS
)

# Latency histogram bucket definitions
LATENCY_BUCKETS = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
    40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
]

TTFT_BUCKETS = [
    0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
    0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
    2560.0
]

TPOT_BUCKETS = [
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
    1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
]

# Request latency histograms
E2E_REQUEST_LATENCY_SECONDS = Histogram(
    METRIC_PREFIX + "e2e_request_latency_seconds",
    "Histogram of end to end request latency in seconds",
    COMMON_LABELS,
    buckets=LATENCY_BUCKETS
)

TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    METRIC_PREFIX + "time_to_first_token_seconds",
    "Histogram of time to first token in seconds",
    COMMON_LABELS,
    buckets=TTFT_BUCKETS
)

TIME_PER_OUTPUT_TOKEN_SECONDS = Histogram(
    METRIC_PREFIX + "time_per_output_token_seconds",
    "Histogram of time per output token in seconds",
    COMMON_LABELS,
    buckets=TPOT_BUCKETS
)

REQUEST_QUEUE_TIME_SECONDS = Histogram(
    METRIC_PREFIX + "request_queue_time_seconds",
    "Histogram of time spent in WAITING phase for request",
    COMMON_LABELS,
    buckets=LATENCY_BUCKETS
)

SERVER_TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    METRIC_PREFIX + "server_time_to_first_token_seconds",
    "Histogram of server-side time to first token in seconds (from HTTP request arrival to first token sent)",
    COMMON_LABELS,
    buckets=TTFT_BUCKETS
)
