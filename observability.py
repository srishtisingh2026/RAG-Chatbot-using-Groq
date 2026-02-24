# observability.py

import json
import logging
import os
import requests
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "rag_app.log")

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)


class Telemetry:

    def __init__(self, backend_url=None):
        self.backend_url = backend_url

    def build_trace(self, data: dict, spans: list):
        trace = data.copy()
        trace["spans"] = spans
        trace["timestamp"] = datetime.utcnow().isoformat()
        return trace

    def log_trace(self, trace: dict):
        logger.info(json.dumps(trace))

        if self.backend_url:
            try:
                requests.post(
                    self.backend_url,
                    json=trace,
                    timeout=2
                )
            except Exception:
                pass