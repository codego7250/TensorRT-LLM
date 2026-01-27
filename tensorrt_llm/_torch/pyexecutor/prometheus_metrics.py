import array
import json
import os
import tempfile

# File path for sharing metrics between worker and main process
PROM_METRICS_FILENAME = os.path.join(tempfile.gettempdir(), "trtllm_prom_metrics.bin")

# Global file handle for writing metrics
prom_metrics_file = None

# Metrics dictionary - accumulated by worker, read by main process
prom_metrics = {
    "num_requests_running": 0.0,
    "num_requests_swapped": 0.0,
    "iteration_tokens_total_sum": 0.0,
    "iteration_tokens_total_count": 0.0,
    "time_per_output_token_seconds_sum": 0.0,
    "time_per_output_token_seconds_count": 0.0,
    "prompt_tokens_total": 0.0,
    "request_prompt_tokens_total_sum": 0.0,
    "request_prompt_tokens_total_count": 0.0,
    "generation_tokens_total": 0.0,
    "request_generation_tokens_total_sum": 0.0,
    "request_generation_tokens_total_count": 0.0,
}


def write_metrics_to_file():
    """Write current metrics to shared file for main process to read."""
    global prom_metrics_file
    try:
        if prom_metrics_file is None:
            prom_metrics_file = os.open(PROM_METRICS_FILENAME, os.O_RDWR | os.O_CREAT)
        # Write keys as JSON followed by null byte, then binary doubles
        data = (
            json.dumps(list(prom_metrics.keys())).encode("UTF-8")
            + b"\0"
            + array.array("d", prom_metrics.values()).tobytes()
        )
        os.pwrite(prom_metrics_file, data, 0)
    except Exception:
        pass  # Silently fail if file operations fail


def read_metrics_from_file():
    """Read metrics from shared file (called by main process)."""
    try:
        if not os.path.exists(PROM_METRICS_FILENAME):
            return None
        with open(PROM_METRICS_FILENAME, "rb") as f:
            data = f.read()
        if not data:
            return None
        # Parse: JSON keys + null byte + binary doubles
        null_idx = data.index(b"\0")
        keys = json.loads(data[:null_idx].decode("UTF-8"))
        values = array.array("d")
        values.frombytes(data[null_idx + 1 :])
        return dict(zip(keys, values))
    except Exception:
        return None
