"""Utilities for Prometheus Metrics Collection."""

import time
from typing import Dict, Optional, Union

from .enums import MetricNames


# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py#L30
class MetricsCollector:
    labelname_finish_reason = "finished_reason"

    def __init__(self, labels: Dict[str, str]) -> None:
        # Lazy import to avoid circular dependency
        # (metrics -> serve -> executor -> metrics)
        from tensorrt_llm.serve.prometheus_metrics import (
            REQUEST_SUCCESS_TOTAL,
            E2E_REQUEST_LATENCY_SECONDS,
            TIME_TO_FIRST_TOKEN_SECONDS,
            TIME_PER_OUTPUT_TOKEN_SECONDS,
            REQUEST_QUEUE_TIME_SECONDS,
            SERVER_TIME_TO_FIRST_TOKEN_SECONDS,
        )

        self.last_log_time = time.time()
        self.labels = labels

        self.finish_reason_label = {
            MetricsCollector.labelname_finish_reason: "unknown"
        }
        self.labels_with_finished_reason = {
            **self.labels,
            **self.finish_reason_label
        }

        # Use centralized metric definitions from prometheus_metrics
        self.counter_request_success = REQUEST_SUCCESS_TOTAL
        self.histogram_e2e_time_request = E2E_REQUEST_LATENCY_SECONDS
        self.histogram_time_to_first_token = TIME_TO_FIRST_TOKEN_SECONDS
        self.histogram_time_per_output_token = TIME_PER_OUTPUT_TOKEN_SECONDS
        self.histogram_queue_time_request = REQUEST_QUEUE_TIME_SECONDS
        self.histogram_server_time_to_first_token = SERVER_TIME_TO_FIRST_TOKEN_SECONDS

    def _label_merge(self, labels: Dict[str, str]) -> Dict[str, str]:
        if labels is None or len(labels) == 0:
            return self.labels
        return {**self.labels, **labels}

    def _log_counter(self, counter, labels: Dict[str, str],
                     data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self._label_merge(labels)).inc(data)

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        # Convenience function for logging to histogram.
        histogram.labels(**self.labels).observe(data)

    def log_request_success(self, data: Union[int, float],
                            labels: Dict[str, str]) -> None:
        self._log_counter(self.counter_request_success, labels, data)
        self.last_log_time = time.time()

    def log_histogram(self, data: Optional[dict[str, float]]) -> None:
        if e2e := data.get(MetricNames.E2E, 0):
            self._log_histogram(self.histogram_e2e_time_request, e2e)
        if ttft := data.get(MetricNames.TTFT, 0):
            self._log_histogram(self.histogram_time_to_first_token, ttft)
        if tpot := data.get(MetricNames.TPOT, 0):
            self._log_histogram(self.histogram_time_per_output_token, tpot)
        if request_queue_time := data.get(MetricNames.REQUEST_QUEUE_TIME, 0):
            self._log_histogram(self.histogram_queue_time_request,
                                request_queue_time)
        # Server-side TTFT (from HTTP request arrival to first token sent)
        if server_ttft := data.get("server_ttft", 0):
            self._log_histogram(self.histogram_server_time_to_first_token,
                                server_ttft)
        self.last_log_time = time.time()

    def log_metrics_dict(self, metrics_dict: dict[str, float]) -> None:
        if finish_reason := metrics_dict.get(
                MetricsCollector.labelname_finish_reason):
            self.log_request_success(
                1, {MetricsCollector.labelname_finish_reason: finish_reason})
            self.log_histogram(metrics_dict)
