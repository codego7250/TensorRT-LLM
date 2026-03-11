# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for Prometheus Metrics Collection."""

import math
import time
from typing import Dict, Union, Optional

from .enums import MetricNames


# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py#L30
class MetricsCollector:
    """
    Collects and logs metrics from TensorRT-LLM engine stats and request performance metrics to Prometheus.

    Used by OpenAIServer in tensorrt_llm/serve/openai_server.py.

    Args:
        labels: A key-value dictionary of labels to add as metadata to all created Prometheus metrics. Useful for
        distinguishing between multiple series of the same metric name. Example:
        {"model_name": "nemotron-nano-3", "engine_type": "trtllm"}

    Created Prometheus metrics:
        trtllm_request_success_total
        trtllm_e2e_request_latency_seconds
        trtllm_time_to_first_token_seconds
        trtllm_time_per_output_token_seconds
        trtllm_request_queue_time_seconds
        trtllm_kv_cache_hit_rate
        trtllm_kv_cache_utilization
    """
    labelname_finish_reason = "finished_reason"

    PPL_BUCKETS = (
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        10.0, 20.0, 50.0, float('inf'),
    )

    TOKENS_BUCKETS = (
        3, 10, 50, 75, 100, 150, 250, 500, 750, 1000,
        2000, 4000, 8000, 16000, 32000, 100000, 200000, float('inf'),
    )

    def __init__(self, labels: Dict[str, str]) -> None:
        from prometheus_client import Counter, Gauge, Histogram
        self.last_log_time = time.time()
        self.labels = labels
        self.metric_prefix = "trtllm_"

        self.finish_reason_label = {
            MetricsCollector.labelname_finish_reason: "unknown"
        }
        self.labels_with_finished_reason = {
            **self.labels,
            **self.finish_reason_label
        }

        self.counter_request_success = Counter(
            name=self.metric_prefix + "request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=self.labels_with_finished_reason.keys())

        self.histogram_e2e_time_request = Histogram(
            name=self.metric_prefix + "e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_to_first_token = Histogram(
            name=self.metric_prefix + "time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_per_output_token = Histogram(
            name=self.metric_prefix + "time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=self.labels.keys())

        self.histogram_queue_time_request = Histogram(
            name=self.metric_prefix + "request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())


        self.kv_cache_hit_rate = Gauge(name=self.metric_prefix +
                                       "kv_cache_hit_rate",
                                       documentation="KV cache hit rate",
                                       labelnames=self.labels.keys())
        self.kv_cache_utilization = Gauge(name=self.metric_prefix +
                                          "kv_cache_utilization",
                                          documentation="KV cache utilization",
                                          labelnames=self.labels.keys())

        # --- Batch occupancy gauges (updated per iteration) ---
        self.gauge_server_queue_length = Gauge(
            name="server_queue_length",
            documentation="Number of requests waiting in the server queue.",
            labelnames=self.labels.keys())

        self.gauge_prefill_num_context_requests = Gauge(
            name="prefill_num_context_requests",
            documentation="Number of context (prefill) requests in the current batch.",
            labelnames=self.labels.keys())

        self.gauge_prefill_batch_occupancy = Gauge(
            name="prefill_batch_occupancy",
            documentation="Fraction of max batch size used by prefill context requests.",
            labelnames=self.labels.keys())

        self.gauge_num_active_requests = Gauge(
            name="num_active_requests",
            documentation="Total number of active (scheduled) requests.",
            labelnames=self.labels.keys())

        # --- Prefill batch tokens histogram ---
        self.histogram_prefill_batch_tokens = Histogram(
            name="prefill_batch_tokens",
            documentation="Number of tokens in the prefill batch.",
            buckets=self.TOKENS_BUCKETS,
            labelnames=self.labels.keys())

        # --- Perplexity histograms ---
        self.histogram_prefill_perplexity = Histogram(
            name="prefill_perplexity",
            documentation="Perplexity of prefill (prompt) tokens.",
            buckets=self.PPL_BUCKETS,
            labelnames=self.labels.keys())

        self.histogram_generation_perplexity = Histogram(
            name="generation_perplexity",
            documentation="Perplexity of generated tokens.",
            buckets=self.PPL_BUCKETS,
            labelnames=self.labels.keys())

        # Initialize gauges so they appear in /metrics immediately
        self.gauge_server_queue_length.labels(**self.labels)
        self.gauge_prefill_num_context_requests.labels(**self.labels)
        self.gauge_prefill_batch_occupancy.labels(**self.labels)
        self.gauge_num_active_requests.labels(**self.labels)

        # Metrics without prefix (match Fireworks naming)
        self.counter_tokens_prompt = Counter(
            name="tokens_prompt_total",
            documentation="Total number of prompt tokens processed.",
            labelnames=self.labels.keys())

        self.counter_tokens_cached_prompt = Counter(
            name="tokens_cached_prompt_total",
            documentation="Total number of prompt tokens reused from cache.",
            labelnames=self.labels.keys())

        self.histogram_tokens_cached_prompt = Histogram(
            name="tokens_cached_prompt_per_request",
            documentation="Histogram of cached prompt tokens per request.",
            buckets=self.TOKENS_BUCKETS,
            labelnames=self.labels.keys())

        self.labelname_token_pos = "token_pos"  # nosec B105
        labels_with_token_pos = list(
            self.labels.keys()) + [self.labelname_token_pos]

        self.counter_tokens_accepted_per_position = Counter(
            name="tokens_accepted_per_position_total",
            documentation=
            "Number of tokens accepted in generation for a given position.",
            labelnames=labels_with_token_pos)

        self.counter_tokens_drafted_per_position = Counter(
            name="tokens_drafted_per_position_total",
            documentation=
            "Number of tokens drafted in generation for a given position.",
            labelnames=labels_with_token_pos)

        self.labelname_http_code = "http_code"
        labels_with_http_code = list(
            self.labels.keys()) + [self.labelname_http_code]
        self.counter_request_error = Counter(
            name="requests_error_total",
            documentation="Total number of failed requests.",
            labelnames=labels_with_http_code)

        # Initialize all counters so they appear in /metrics even before
        # the first event occurs (avoids Prometheus staleness gaps).
        self.counter_tokens_prompt.labels(**self.labels)
        self.counter_tokens_cached_prompt.labels(**self.labels)
        self.counter_request_error.labels(**self.labels,
                                          **{self.labelname_http_code: ""})


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

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def log_histogram(self, data: Optional[dict[str, float]]) -> None:
        if not data:
            return
        if e2e := data.get(MetricNames.E2E, 0):
            self._log_histogram(self.histogram_e2e_time_request, e2e)
        if ttft := data.get(MetricNames.TTFT, 0):
            self._log_histogram(self.histogram_time_to_first_token, ttft)
        if tpot := data.get(MetricNames.TPOT, 0):
            self._log_histogram(self.histogram_time_per_output_token, tpot)
        if request_queue_time := data.get(MetricNames.REQUEST_QUEUE_TIME, 0):
            self._log_histogram(self.histogram_queue_time_request,
                                request_queue_time)
        if prompt_tokens := data.get(MetricNames.PROMPT_TOKENS, 0):
            self._log_counter(self.counter_tokens_prompt, self.labels,
                              prompt_tokens)
        if cached_tokens := data.get(MetricNames.PROMPT_CACHE_CACHED_TOKENS, 0):
            self._log_counter(self.counter_tokens_cached_prompt, self.labels,
                              cached_tokens)
            self._log_histogram(self.histogram_tokens_cached_prompt,
                                cached_tokens)
        per_pos_drafted = data.get(MetricNames.SPEC_DEC_DRAFTED_PER_POS)
        per_pos_accepted = data.get(MetricNames.SPEC_DEC_ACCEPTED_PER_POS)
        if per_pos_drafted is not None and per_pos_accepted is not None:
            # Only emit positions up to the last non-zero draft
            last_nonzero = -1
            for i in range(len(per_pos_drafted) - 1, -1, -1):
                if per_pos_drafted[i] > 0:
                    last_nonzero = i
                    break
            for pos in range(last_nonzero + 1):
                labels_with_pos = {**self.labels, self.labelname_token_pos: pos}
                if per_pos_drafted[pos] > 0:
                    self.counter_tokens_drafted_per_position.labels(
                        **labels_with_pos).inc(per_pos_drafted[pos])
                if per_pos_accepted[pos] > 0:
                    self.counter_tokens_accepted_per_position.labels(
                        **labels_with_pos).inc(per_pos_accepted[pos])

        prefill_ppl = data.get(MetricNames.PREFILL_PERPLEXITY)
        if prefill_ppl is not None and math.isfinite(prefill_ppl):
            self._log_histogram(self.histogram_prefill_perplexity, prefill_ppl)
        gen_ppl = data.get(MetricNames.GENERATION_PERPLEXITY)
        if gen_ppl is not None and math.isfinite(gen_ppl):
            self._log_histogram(self.histogram_generation_perplexity, gen_ppl)
        self.last_log_time = time.time()

    def log_iteration_metrics(self, stats: dict) -> None:
        """Update gauge metrics from iteration stats polled from the executor."""
        if not isinstance(stats, dict):
            return
        num_queued = stats.get("numQueuedRequests", 0)
        num_active = stats.get("numActiveRequests", 0)
        max_active = stats.get("maxNumActiveRequests", 1)

        ifb = stats.get("inflightBatchingStats") or {}
        num_context = ifb.get("numContextRequests", 0)
        num_ctx_tokens = ifb.get("numCtxTokens", 0)

        self.gauge_server_queue_length.labels(**self.labels).set(num_queued)
        self.gauge_num_active_requests.labels(**self.labels).set(num_active)
        self.gauge_prefill_num_context_requests.labels(**self.labels).set(num_context)

        occupancy = num_context / max_active if max_active > 0 else 0.0
        self.gauge_prefill_batch_occupancy.labels(**self.labels).set(occupancy)

        if num_ctx_tokens > 0:
            self._log_histogram(self.histogram_prefill_batch_tokens, num_ctx_tokens)

    def log_request_error(self, http_code: Union[int, str] = "") -> None:
        """Increment the error counter, labeled by HTTP status code."""
        labels = {**self.labels, self.labelname_http_code: str(http_code)}
        self.counter_request_error.labels(**labels).inc(1)
        self.last_log_time = time.time()

    def log_request_metrics_dict(self, metrics_dict: dict[str, float]) -> None:
        """
        Log per-request metrics from TRTLLM engine responses.
        This method updates Prometheus metrics including:
        - counter_request_success
        - histogram_e2e_time_request
        - histogram_time_to_first_token
        - histogram_time_per_output_token
        - histogram_queue_time_request

        Args:
            metrics_dict: A dictionary containing request metrics with the following expected keys:
                - `MetricsCollector.labelname_finish_reason` (str): Finish reason string indicating
                  request completion status.
                - `MetricNames.E2E` (float): End-to-end request latency in seconds.
                - `MetricNames.TTFT` (float): Time to first token in seconds.
                - `MetricNames.TPOT` (float): Time per output token in seconds.
                - `MetricNames.REQUEST_QUEUE_TIME` (float): Request queue time in seconds.

        Returns:
            None: Metrics are logged to Prometheus; nothing is returned.

        Note:
            - Needs to include `return_perf_metrics: true` in LLM args to populate the metrics_dict field
            from the engine responses.
            - Metrics are only recorded when MetricsCollector.labelname_finish_reason is present
            in the metrics_dict, indicating the request has finished.

        """
        if finish_reason := metrics_dict.get(
                MetricsCollector.labelname_finish_reason):

            # If the request finishes, log per-request metrics
            self._log_counter(
                self.counter_request_success,
                {MetricsCollector.labelname_finish_reason: finish_reason}, 1)
            if e2e := metrics_dict.get(MetricNames.E2E, 0):
                self._log_histogram(self.histogram_e2e_time_request, e2e)
            if ttft := metrics_dict.get(MetricNames.TTFT, 0):
                self._log_histogram(self.histogram_time_to_first_token, ttft)
            if tpot := metrics_dict.get(MetricNames.TPOT, 0):
                self._log_histogram(self.histogram_time_per_output_token, tpot)
            if request_queue_time := metrics_dict.get(
                    MetricNames.REQUEST_QUEUE_TIME, 0):
                self._log_histogram(self.histogram_queue_time_request,
                                    request_queue_time)
            self.last_log_time = time.time()
            self.log_request_success(
                1, {MetricsCollector.labelname_finish_reason: finish_reason})
            self.log_histogram(metrics_dict)

    def log_iteration_stats(self, iteration_stats: dict) -> None:
        """
        Log iteration-level statistics from TRTLLM engine.

        This method updates Prometheus metrics including:
        - kv_cache_hit_rate
        - kv_cache_utilization

        Args:
            iteration_stats: A JSON dict returned from `BaseLLM.get_stats()` containing iteration-level statistics
                with the following expected structure:
                - "kvCacheStats" (dict): KV cache statistics containing:
                    - "cacheHitRate" (float): Cache hit rate (0.0 to 1.0). If present (including zero),
                      the kv_cache_hit_rate gauge is updated.
                    - "usedNumBlocks" (int): Number of KV cache blocks currently in use.
                    - "maxNumBlocks" (int): Maximum number of KV cache blocks available. Should always be
                      non-zero.

        Returns:
            None: Metrics are logged to Prometheus; nothing is returned.

        Note:
            - Needs to include `enable_iter_perf_stats: true` in LLM args to collect iteration-level stats.
            - KV cache utilization is only calculated and logged when both "usedNumBlocks" and
              "maxNumBlocks" are present in kvCacheStats and "maxNumBlocks" is non-zero.
        """
        if kv_stats := iteration_stats.get("kvCacheStats"):
            cache_hit_rate = kv_stats.get("cacheHitRate")
            if cache_hit_rate is not None:
                self._log_gauge(self.kv_cache_hit_rate, cache_hit_rate)
            if "usedNumBlocks" in kv_stats and "maxNumBlocks" in kv_stats:
                max_num_blocks = kv_stats["maxNumBlocks"]
                if max_num_blocks:
                    utilization = kv_stats["usedNumBlocks"] / max_num_blocks
                    self._log_gauge(self.kv_cache_utilization, utilization)
