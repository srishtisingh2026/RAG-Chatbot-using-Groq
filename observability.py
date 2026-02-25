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

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler(log_file_path, mode="a")
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)


class Telemetry:

    def __init__(self, backend_url=None):
        self.backend_url = backend_url
        
        # Cosmos DB Configuration
        self.cosmos_conn = os.getenv("COSMOS_CONN_WRITE")
        self.db_name = os.getenv("COSMOS_DB", "llmops-data")
        self.container_name = "raw_traces"
        
        self.client = None
        self.container = None
        
        if self.cosmos_conn:
            try:
                # Parse connection string for endpoint and key
                endpoint = self.cosmos_conn.split("AccountEndpoint=")[1].split(";")[0]
                key = self.cosmos_conn.split("AccountKey=")[1].split(";")[0]
                
                self.client = CosmosClient(endpoint, key)
                db = self.client.get_database_client(self.db_name)
                self.container = db.get_container_client(self.container_name)
            except Exception as e:
                print(f"Failed to initialize Cosmos DB client: {e}")

    def log_trace(self, trace: dict):
        # Ensure 'id' exists for Cosmos DB (using trace_id)
        if "trace_id" in trace:
            trace["id"] = trace["trace_id"]
        
        # 1. Local logging
        logger.info(json.dumps(trace))

        # 2. External Backend logging
        if self.backend_url:
            try:
                requests.post(self.backend_url, json=trace, timeout=2)
            except Exception:
                pass
                
        # 3. Azure Cosmos DB logging
        if self.container:
            try:
                self.container.upsert_item(body=trace)
            except Exception as e:
                print(f"Failed to log trace to Cosmos DB: {e}")