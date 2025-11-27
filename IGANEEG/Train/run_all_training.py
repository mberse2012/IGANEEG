
"""
Batch run all 4 training scripts
Train IGANEEG models using different window configurations respectively
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import argparse


def run_training_script(script_name, config_name):
    """Run a single training script"""
    print(f"\n{'=' * 80}")
    print(f"Starting training script: {script_name}")
    print(f"Configuration: {config_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    start_time = time.time()

    try:
        # Run training script
        result = subprocess.run([sys.executable, script_name],
                                capture_output=True,
                                text=True,
                                timeout=7200)  # 2 hour timeout

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n{script_name} execution completed!")
        print(f"Execution time: {duration / 60:.2f} minutes")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if result.returncode == 0:
            print(f"✅ {script_name} executed successfully")
            print("Output:")
            print(result.stdout)
        else:
            print(f"❌ {script_name} execution failed")
            print("Error output:")
            print(result.stderr)

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} execution timeout (2 hours)")
        return False
    except Exception as e:
        print(f"❌ {script_name} execution exception: {e}")
        return False


def run_parallel_training(configs, max_workers=2):
    """Run training scripts in parallel"""
    import concurrent.futures

    print(f"Using parallel mode, maximum parallel count: {max_workers}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_training_script, script, config): config
            for script, config in configs
        }

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(future_to_config):
            config = future_to_config[future]
            try:
                success = future.result()
                if success:
                    print(f"✅ {config} training completed")
                else:
                    print(f"❌ {config} training failed")
            except Exception as e:
                print(f"❌ {config} training exception: {e}")


def run_sequential_training(configs):
    """Run training scripts sequentially"""
    print("Using sequential mode")

    success_count = 0
    total_count = len(configs)

    for script, config in configs:
        success = run_training_script(script, config)
        if success:
            success_count += 1

        # Brief rest to avoid resource competition
        time.sleep(5)

    print(f"\n{'=' * 80}")
    print(f"All training scripts execution completed!")
    print(f"Success: {success_count}/{total_count}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")


def check_prerequisites():
    """Check running prerequisites"""
    print("Checking running prerequisites...")

    # Check if data path exists
    data_path = "../Data Preprocessing/sliding_window_results"
    if not os.path.exists(data_path):
        print(f"❌ Data path does not exist: {data_path}")
        return False

    # Check necessary files
    required_files = [
        "../Model/IGANEEG.py",
        "../Model/config.py",
        "../Model/utils.py"
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Necessary file does not exist: {file_path}")
            return False

    # Check training scripts
    training_scripts = [
        "train_win64_overlap0.py",
        "train_win64_overlap32.py",
        "train_win128_overlap0.py",
        "train_win128_overlap32.py"
    ]

    for script in training_scripts:
        if not os.path.exists(script):
            print(f"❌ Training script does not exist: {script}")
            return False

    print("✅ All prerequisite checks passed")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch run IGANEEG training scripts")
    parser.add_argument("--mode", choices=["sequential", "parallel"],
                        default="sequential", help="Running mode")
    parser.add_argument("--workers", type=int, default=2,
                        help="Maximum worker processes in parallel mode")
    parser.add_argument("--configs", nargs="+",
                        choices=["win64_overlap0", "win64_overlap32",
                                 "win128_overlap0", "win128_overlap32"],
                        help="Specify configurations to run, run all configurations by default")

    args = parser.parse_args()

    print("IGANEEG Batch Training Script")
    print("=" * 80)
    print(f"Running mode: {args.mode}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not check_prerequisites():
        print("Prerequisite check failed, please resolve the above issues and try again")
        return

    # Define all configurations
    all_configs = [
        ("train_win64_overlap0.py", "win64_overlap0"),
        ("train_win64_overlap32.py", "win64_overlap32"),
        ("train_win128_overlap0.py", "win128_overlap0"),
        ("train_win128_overlap32.py", "win128_overlap32")
    ]

    # Filter configurations
    if args.configs:
        configs = [(script, config) for script, config in all_configs
                   if config in args.configs]
    else:
        configs = all_configs

    print(f"Configurations to run: {[config for _, config in configs]}")

    # Run training
    if args.mode == "parallel":
        run_parallel_training(configs, args.workers)
    else:
        run_sequential_training(configs)


if __name__ == "__main__":
    main()
