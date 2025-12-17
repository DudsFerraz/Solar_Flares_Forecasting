import os
import json
import logging
from datetime import datetime, timezone
import numpy as np

class OutputHandler:
    def __init__(self):
        self.json_path = os.getenv("DASHBOARD_JSON_PATH", "dashboard_state.json")

    def update_dashboard_file(self, all_predictions: dict, latency_min: float):
        state = {
            "last_update_utc": datetime.now(timezone.utc).isoformat(),
            "data_latency_minutes": latency_min,
            "predictions": all_predictions
        }

        def numpy_converter(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            with open(self.json_path, 'w') as f:
                json.dump(state, f, indent=4, default=numpy_converter)
        except Exception as e:
            logging.error(f"Falha ao atualizar arquivo do dashboard: {e}")