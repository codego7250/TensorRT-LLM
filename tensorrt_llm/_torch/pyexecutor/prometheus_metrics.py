"""Prometheus metrics for PyExecutor."""

from prometheus_client import Counter, Gauge, Histogram


# Gauges (current state)
NUM_REQUESTS_RUNNING = Gauge(
    'num_requests_running',
    'Number of requests currently running'
)

NUM_REQUESTS_SWAPPED = Gauge(
    'num_requests_swapped',
    'Number of requests currently swapped'
)

# Counters (cumulative)
PROMPT_TOKENS_TOTAL = Counter(
    'prompt_tokens_total',
    'Total number of prompt tokens processed'
)

GENERATION_TOKENS_TOTAL = Counter(
    'generation_tokens_total',
    'Total number of generation tokens produced'
)

ITERATION_TOKENS_TOTAL = Counter(
    'iteration_tokens_total',
    'Total number of tokens processed per iteration'
)

# Histograms (distributions)
TIME_PER_OUTPUT_TOKEN_SECONDS = Histogram(
    'time_per_output_token_seconds',
    'Time per output token in seconds',
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

REQUEST_PROMPT_TOKENS_TOTAL = Histogram(
    'request_prompt_tokens_total',
    'Prompt tokens per request',
    buckets=(1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000)
)

REQUEST_GENERATION_TOKENS_TOTAL = Histogram(
    'request_generation_tokens_total',
    'Generation tokens per request',
    buckets=(1, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000)
)
