"""
run_pipeline.py - Master Executable Script for Urban Traffic GNN Project
Provides single-command orchestration:
- python3 run_pipeline.py --test      (Runs unit and integration test suite)
- python3 run_pipeline.py --train     (Runs end-to-end dataset generation, training, and benchmarking)
- python3 run_pipeline.py --serve     (Starts interactive web visualizer dashboard)
- python3 run_pipeline.py --all       (Executes test, train, and launches web server)
"""

import sys
import os
import argparse
import subprocess

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

def run_tests():
    print("\n" + "=" * 60)
    print(">>> RUNNING ARCHITECTURE & UNIT TEST SUITE")
    print("=" * 60)
    res = subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "test_models.py")])
    if res.returncode != 0:
        print("[Error] Tests failed. Aborting pipeline.")
        sys.exit(res.returncode)

def run_training():
    print("\n" + "=" * 60)
    print(">>> RUNNING DATA GENERATION & MODEL BENCHMARKING")
    print("=" * 60)
    res = subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "train_and_compare.py")])
    if res.returncode != 0:
        print("[Error] Training failed.")
        sys.exit(res.returncode)

def run_server():
    print("\n" + "=" * 60)
    print(">>> LAUNCHING INTERACTIVE FULL-STACK WEB DASHBOARD")
    print("=" * 60)
    try:
        subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "app.py")])
    except KeyboardInterrupt:
        print("\nWeb server stopped.")

def main():
    parser = argparse.ArgumentParser(description="Urban Traffic GNN Project Pipeline")
    parser.add_argument("--test", action="store_true", help="Run model test suite")
    parser.add_argument("--train", action="store_true", help="Train and benchmark all models")
    parser.add_argument("--serve", action="store_true", help="Launch interactive web app")
    parser.add_argument("--all", action="store_true", help="Run full pipeline: test -> train -> serve")

    args = parser.parse_args()

    if not any([args.test, args.train, args.serve, args.all]):
        # Default behavior if run with no args: launch web server (or run tests if data doesn't exist)
        data_exists = os.path.exists(os.path.join(PROJECT_DIR, "results", "comparative_results.json"))
        if not data_exists:
            print("[Auto] First time setup detected. Running test & training pipeline...")
            run_tests()
            run_training()
        run_server()
        return

    if args.test or args.all:
        run_tests()
    if args.train or args.all:
        run_training()
    if args.serve or args.all:
        run_server()

if __name__ == "__main__":
    main()
