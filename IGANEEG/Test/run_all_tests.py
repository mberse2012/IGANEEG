
"""
Batch run all IGANEEG model test scripts
Support sequential running and parallel running
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
import pickle


class TestRunner:
    def __init__(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.configs = [
            "win64_overlap0",
            "win64_overlap32",
            "win128_overlap0",
            "win128_overlap32"
        ]
        self.results = {}

    def run_single_test(self, config_name):
        """Run single test configuration"""
        print(f"\n{'=' * 80}")
        print(f"Starting test: {config_name}")
        print(f"{'=' * 80}")

        start_time = time.time()

        # Build test script path
        test_script = os.path.join(self.test_dir, f"test_{config_name}.py")

        if not os.path.exists(test_script):
            print(f"Error: Test script does not exist - {test_script}")
            return None

        try:
            # Run test script
            result = subprocess.run(
                [sys.executable, test_script],
                cwd=self.test_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            end_time = time.time()
            duration = end_time - start_time

            # Record result
            test_result = {
                'config': config_name,
                'success': result.returncode == 0,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'start_time': datetime.fromtimestamp(start_time),
                'end_time': datetime.fromtimestamp(end_time)
            }

            # Try to load test result file
            results_file = os.path.join(self.test_dir, f'results_{config_name}', 'test_results.pkl')
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'rb') as f:
                        test_metrics = pickle.load(f)
                    test_result['metrics'] = test_metrics
                except Exception as e:
                    print(f"Warning: Unable to load test result file {results_file}: {e}")

            self.results[config_name] = test_result

            # Output result
            if test_result['success']:
                print(f"✅ Test {config_name} completed successfully!")
                print(f"⏱️  Time taken: {duration:.2f} seconds")

                if 'metrics' in test_result:
                    metrics = test_result['metrics']
                    print(f"📊 MMD: {metrics.get('mmd', 'N/A'):.6f}")
                    print(f"📊 Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"📊 MAE: {metrics.get('mae', 'N/A'):.4f}")
                    print(f"📊 MSE: {metrics.get('mse', 'N/A'):.4f}")
            else:
                print(f"❌ Test {config_name} failed!")
                print(f"⏱️  Time taken: {duration:.2f} seconds")
                print(f"Error message:\n{result.stderr}")

            return test_result

        except subprocess.TimeoutExpired:
            print(f"❌ Test {config_name} timed out!")
            return None
        except Exception as e:
            print(f"❌ Error occurred while running test {config_name}: {e}")
            return None

    def run_sequential(self, configs=None):
        """Run all tests sequentially"""
        if configs is None:
            configs = self.configs

        print("🚀 Starting sequential test execution...")
        print(f"Test configurations: {', '.join(configs)}")

        total_start_time = time.time()

        for config in configs:
            self.run_single_test(config)

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        print(f"\n{'=' * 80}")
        print("🎉 All tests completed!")
        print(f"⏱️  Total time: {total_duration:.2f} seconds")
        print(f"{'=' * 80}")

        self.print_summary()
        self.save_results()

    def run_parallel(self, configs=None, max_workers=2):
        """Run tests in parallel (simplified version, sequential running recommended due to resource constraints)"""
        print("⚠️  Warning: Parallel execution may consume significant resources")
        print("Sequential running mode is recommended")

        # For stability, use sequential running here
        self.run_sequential(configs)

    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 80)
        print("📋 Test Results Summary")
        print("=" * 80)

        successful_tests = []
        failed_tests = []

        for config, result in self.results.items():
            if result and result['success']:
                successful_tests.append(config)
            else:
                failed_tests.append(config)

        print(f"✅ Successful: {len(successful_tests)} tests")
        print(f"❌ Failed: {len(failed_tests)} tests")

        if successful_tests:
            print(f"\n✅ Successful tests:")
            for config in successful_tests:
                result = self.results[config]
                print(f"  - {config}: {result['duration']:.2f}s")
                if 'metrics' in result:
                    metrics = result['metrics']
                    print(f"    MMD: {metrics.get('mmd', 'N/A'):.6f}, "
                          f"Acc: {metrics.get('accuracy', 'N/A'):.4f}")

        if failed_tests:
            print(f"\n❌ Failed tests:")
            for config in failed_tests:
                print(f"  - {config}")

        # Performance comparison
        if len(successful_tests) > 1:
            print(f"\n📊 Performance comparison:")
            print(f"{'Configuration':<20} {'MMD':<12} {'Accuracy':<12} {'MAE':<12} {'MSE':<12}")
            print("-" * 80)

            for config in successful_tests:
                result = self.results[config]
                if 'metrics' in result:
                    metrics = result['metrics']
                    print(f"{config:<20} "
                          f"{metrics.get('mmd', 0):<12.6f} "
                          f"{metrics.get('accuracy', 0):<12.4f} "
                          f"{metrics.get('mae', 0):<12.4f} "
                          f"{metrics.get('mse', 0):<12.4f}")

    def save_results(self):
        """Save test results"""
        results_file = os.path.join(self.test_dir, 'all_test_results.pkl')

        try:
            with open(results_file, 'wb') as f:
                pickle.dump(self.results, f)
            print(f"\n💾 Test results saved to: {results_file}")
        except Exception as e:
            print(f"❌ Failed to save test results: {e}")

        # Generate text report
        report_file = os.path.join(self.test_dir, 'test_report.txt')
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("IGANEEG Model Test Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for config, result in self.results.items():
                    f.write(f"Configuration: {config}\n")
                    f.write("-" * 30 + "\n")
                    if result:
                        f.write(f"Status: {'Successful' if result['success'] else 'Failed'}\n")
                        f.write(f"Time taken: {result['duration']:.2f} seconds\n")
                        if 'metrics' in result:
                            metrics = result['metrics']
                            f.write(f"MMD: {metrics.get('mmd', 'N/A'):.6f}\n")
                            f.write(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}\n")
                            f.write(f"MAE: {metrics.get('mae', 'N/A'):.4f}\n")
                            f.write(f"MSE: {metrics.get('mse', 'N/A'):.4f}\n")
                    f.write("\n")

            print(f"📄 Test report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Failed to generate test report: {e}")


def main():
    parser = argparse.ArgumentParser(description='Batch run IGANEEG model tests')
    parser.add_argument('--mode', choices=['sequential', 'parallel'],
                        default='sequential', help='Running mode')
    parser.add_argument('--configs', nargs='+',
                        choices=['win64_overlap0', 'win64_overlap32', 'win128_overlap0', 'win128_overlap32'],
                        help='Specify configurations to run')
    parser.add_argument('--workers', type=int, default=2,
                        help='Maximum worker processes for parallel running')

    args = parser.parse_args()

    # Create test runner
    runner = TestRunner()

    # Determine configurations to run
    configs = args.configs if args.configs else runner.configs

    print("🧠 IGANEEG Model Batch Testing Tool")
    print("=" * 50)
    print(f"Running mode: {args.mode}")
    print(f"Test configurations: {', '.join(configs)}")
    print(f"Working directory: {runner.test_dir}")
    print("=" * 50)

    # Run tests
    if args.mode == 'sequential':
        runner.run_sequential(configs)
    else:
        runner.run_parallel(configs, args.workers)


if __name__ == "__main__":
    main()
