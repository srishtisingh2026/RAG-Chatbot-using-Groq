# observability.py

import json
import logging
import os
import requests
from datetime import datetime
from azure.cosmos import CosmosClient

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, "rag_app.log")

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path, mode="a")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)


class Telemetry:

    def __init__(self, backend_url=None):

        self.backend_url = backend_url

        # Cosmos configuration
        self.cosmos_conn = os.getenv("COSMOS_CONN_WRITE")
        self.db_name = os.getenv("COSMOS_DB", "llmops-data")
        self.container_name = "raw_traces"

        self.client = None
        self.container = None

        if self.cosmos_conn:

            try:

                self.client = CosmosClient.from_connection_string(
                    self.cosmos_conn
                )

                db = self.client.get_database_client(self.db_name)

                self.container = db.get_container_client(
                    self.container_name
                )

                print("Cosmos telemetry enabled")

            except Exception as e:

                print(f"Cosmos initialization failed: {e}")

    # ---------------------------------------------------------
    # Trace Logging
    # ---------------------------------------------------------

    def log_trace(self, trace: dict):

        try:

            # Ensure ID exists
            if "trace_id" in trace:
                trace["id"] = trace["trace_id"]

            # Ensure partition key
            trace["partitionKey"] = trace.get(
                "partitionKey",
                trace.get("id")
            )

            # Ensure timestamp
            trace.setdefault(
                "logged_at",
                datetime.utcnow().isoformat()
            )

            # -----------------------------------------
            # 1. Local logging
            # -----------------------------------------

            logger.info(json.dumps(trace))

            # -----------------------------------------
            # 2. External backend
            # -----------------------------------------

            if self.backend_url:

                try:

                    requests.post(
                        self.backend_url,
                        json=trace,
                        timeout=1
                    )

                except Exception:
                    pass

            # -----------------------------------------
            # 3. Cosmos DB
            # -----------------------------------------

            if self.container:

                try:

                    self.container.upsert_item(body=trace)

                except Exception as e:

                    print(f"Cosmos logging failed: {e}")

        except Exception as e:

            print(f"Telemetry error: {e}")