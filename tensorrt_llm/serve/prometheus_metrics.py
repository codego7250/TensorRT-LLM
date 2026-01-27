"""Prometheus metrics for OpenAI Server."""

from prometheus_client import Counter, Gauge

# Request lifecycle counters
REQUEST_STARTED_TOTAL = Counter(
    'fw_request_started_total',
    'Total number of requests started'
)

REQUEST_COMPLETED_TOTAL = Counter(
    'fw_request_completed_total',
    'Total number of requests completed'
)

REQUEST_CANCELLED_TOTAL = Counter(
    'fw_request_cancelled_total',
    'Total number of requests cancelled'
)

REQUEST_FAILED_TOTAL = Counter(
    'fw_request_failed_total',
    'Total number of requests failed'
)

REQUEST_SUCCESS_TOTAL = Counter(
    'fw_request_success_total',
    'Total number of successful requests by finish reason',
    ['finished_reason']
)

# Request queue state
NUM_REQUESTS_WAITING = Gauge(
    'fw_num_requests_waiting',
    'Number of requests waiting to be processed (in queue)'
)
