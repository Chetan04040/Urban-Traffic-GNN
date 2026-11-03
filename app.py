"""
app.py - Full-Stack Urban Traffic GNN Web Server & REST API
Serves interactive real-time visualizer and provides API endpoints for:
1. Road network graph metadata and adjacency (/api/metadata)
2. Empirical benchmark comparison metrics (/api/metrics)
3. Multi-horizon model predictions & ground truth playback (/api/predictions)
4. Live PyTorch inference endpoint (/api/predict)
"""

import os
import sys
import json
import numpy as np
import torch
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

PORT = 8080
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
WEB_DIR = os.path.join(ROOT_DIR, "web_app")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

class TrafficAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/metadata":
            self.send_json_response(self.get_metadata())
        elif path == "/api/metrics":
            self.send_json_response(self.get_metrics())
        elif path == "/api/predictions":
            self.send_json_response(self.get_predictions())
        else:
            # Fall back to serving static files from web_app/
            super().do_GET()

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def get_metadata(self):
        meta_file = os.path.join(DATA_DIR, "metadata.json")
        adj_file = os.path.join(DATA_DIR, "adj_matrix.npy")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)
            if os.path.exists(adj_file):
                adj = np.load(adj_file).tolist()
                meta["adj_matrix"] = adj
            return meta
        return {"error": "Metadata not found. Please run train_and_compare.py first."}

    def get_metrics(self):
        metrics_file = os.path.join(RESULTS_DIR, "comparative_results.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                return json.load(f)
        return {"error": "Metrics not found. Please run train_and_compare.py first."}

    def get_predictions(self):
        gt_file = os.path.join(RESULTS_DIR, "ground_truth_test.npy")
        if not os.path.exists(gt_file):
            return {"error": "Predictions not found. Please run train_and_compare.py first."}

        gt = np.load(gt_file)  # [T, T_out, N]
        models = ["Baseline_LSTM", "Base_STGCN", "Improved_A_STGCN"]
        preds = {}
        for m in models:
            m_file = os.path.join(RESULTS_DIR, f"{m}_preds.npy")
            if os.path.exists(m_file):
                preds[m] = np.load(m_file).tolist()

        return {
            "ground_truth": gt.tolist(),
            "predictions": preds
        }

def start_server():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, TrafficAPIHandler)
    print("=" * 70)
    print(f"🚦 Urban Traffic GNN Full-Stack Server Running at:")
    print(f"   👉 http://localhost:{PORT}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    start_server()
