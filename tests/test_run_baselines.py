import tempfile
import unittest
from pathlib import Path
from dacbench.logger import load_logs, log2dataframe

from dacbench.run_baselines import run_random, DISCRETE_ACTIONS, run_static

benchmarks = [
    "SigmoidBenchmark",
    "LubyBenchmark",
    "FastDownwardBenchmark",
    "CMAESBenchmark",
    "ModeaBenchmark",
    "SGDBenchmark",
]


class TestRunBaselines(unittest.TestCase):
    def run_random_test_with_benchmark(self, benchmark):
        seeds = [42]
        fixed = 2
        num_episodes = 10

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir)

            run_random(result_path, benchmark, num_episodes, seeds, fixed)

            expected_experiment_path = (
                result_path / benchmark / f"random_fixed{fixed}_{seeds[0]}"
            )
            self.assertTrue(expected_experiment_path.exists())

            performance_tracking_log = (
                expected_experiment_path / "PerformanceTrackingWrapper.jsonl"
            )
            self.assertTrue(performance_tracking_log.exists())

            logs = log2dataframe(load_logs(performance_tracking_log))
            self.assertEqual(len(logs), num_episodes)
            self.assertTrue((logs["seed"] == seeds[0]).all())

    def test_run_random_SigmoidBenchmark(self):
        self.run_random_test_with_benchmark("SigmoidBenchmark")

    def test_run_random_LubyBenchmark(self):
        self.run_random_test_with_benchmark("LubyBenchmark")

    def test_run_random_FastDownwardBenchmark(self):
        self.run_random_test_with_benchmark("FastDownwardBenchmark")

    def test_run_random_CMAESBenchmark(self):
        self.run_random_test_with_benchmark("CMAESBenchmark")

    def test_run_random_SGDBenchmark(self):
        self.run_random_test_with_benchmark("SGDBenchmark")

    def test_run_random_ModeaBenchmark(self):
        self.run_random_test_with_benchmark("ModeaBenchmark")

    def run_static_test_with_benchmark(self, benchmark):
        seeds = [42]
        num_episodes = 10
        action = DISCRETE_ACTIONS[benchmark][0]
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir)

            run_static(result_path, benchmark, action, num_episodes, seeds)

            expected_experiment_path = (
                result_path / benchmark / f"static_{action}_{seeds[0]}"
            )
            self.assertTrue(expected_experiment_path.exists())

            performance_tracking_log = (
                expected_experiment_path / "PerformanceTrackingWrapper.jsonl"
            )
            self.assertTrue(performance_tracking_log.exists())

            logs = log2dataframe(load_logs(performance_tracking_log))
            self.assertEqual(len(logs), num_episodes)
            self.assertTrue((logs["seed"] == seeds[0]).all())

    def test_run_static_SigmoidBenchmark(self):
        self.run_static_test_with_benchmark("SigmoidBenchmark")

    def test_run_static_LubyBenchmark(self):
        self.run_static_test_with_benchmark("LubyBenchmark")

    def test_run_static_FastDownwardBenchmark(self):
        self.run_static_test_with_benchmark("FastDownwardBenchmark")

    def test_run_static_CMAESBenchmark(self):
        self.run_static_test_with_benchmark("CMAESBenchmark")

    def test_run_static_SGDBenchmark(self):
        self.run_static_test_with_benchmark("SGDBenchmark")

    def test_run_static_ModeaBenchmark(self):
        self.run_static_test_with_benchmark("ModeaBenchmark")
