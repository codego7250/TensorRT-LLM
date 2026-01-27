"""Prometheus metrics for OpenAI Server."""
from prometheus_client import Counter, Gauge, Summary

# Common labels for all metrics
COMMON_LABELS = ["model_name", "deployment", "engine_type"]

REQUEST_FAILED_TOTAL = Counter(
    "request_failed_total",
    "Total number of requests failed",
    COMMON_LABELS
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

GENERATION_TOKENS_TOTAL = Gauge(
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
