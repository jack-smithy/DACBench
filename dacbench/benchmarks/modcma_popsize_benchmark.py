import csv
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from gymnasium import spaces

from dacbench.abstract_benchmark import AbstractBenchmark, objdict
from dacbench.envs import CMAESPopSizeEnv

DEFAULT_CFG_SPACE = CS.ConfigurationSpace()
POP_SIZE = CSH.UniformFloatHyperparameter(name="Pop_size", lower=4, upper=5120)
DEFAULT_CFG_SPACE.add_hyperparameter(POP_SIZE)

INFO = {
    "identifier": "CMA-ES",
    "name": "pop-size adaption in CMA-ES",
    "reward": "Negative best function value",
    "state_description": [
        "lambda_",
        "ptnorm",
        "normalisation_factor",
    ],
}

CMAES_DEFAULTS = objdict(
    {
        "action_space_class": "Box",
        "action_space_args": [np.array([0]), np.array([5120])],
        "config_space": DEFAULT_CFG_SPACE,
        "observation_space_class": "Box",
        "observation_space_type": np.float32,
        "observation_space_args": [-1 * np.inf * np.ones(3), np.inf * np.ones(3)],
        "reward_range": (-(10**9), (10**9)),
        "cutoff": 1e6,
        "seed": 0,
        "instance_set_path": "../instance_sets/cma/cma_train.csv",
        "test_set_path": "../instance_sets/cma/cma_test.csv",
        "benchmark_info": INFO,
        "budget": int(1e5)
    }
)


class CMAESPopSizeBenchmark(AbstractBenchmark):
    """
    Benchmark with default configuration & relevant functions for CMA-ES
    """

    def __init__(self, config_path=None, config=None):
        """
        Initialize CMA Benchmark

        Parameters
        -------
        config_path : str
            Path to config file (optional)
        """
        super(CMAESPopSizeBenchmark, self).__init__(config_path, config)
        if not self.config:
            self.config = objdict(CMAES_DEFAULTS.copy())

        for key in CMAES_DEFAULTS:
            if key not in self.config:
                self.config[key] = CMAES_DEFAULTS[key]
                

    def get_environment(self):
        """
        Return CMAESEnv env with current configuration

        Returns
        -------
        CMAESEnv
            CMAES environment
        """
        if "instance_set" not in self.config.keys():
            self.read_instance_set()

        # Read test set if path is specified
        if (
            "test_set" not in self.config.keys()
            and "test_set_path" in self.config.keys()
        ):
            self.read_instance_set(test=True)

        env = CMAESPopSizeEnv(self.config)
        
        for func in self.wrap_funcs:
            env = func(env)

        return env

    def read_instance_set(self, test=False):
        """
        Read path of instances from config into list
        """
        if test:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.test_set_path
            )
            keyword = "test_set"
        else:
            path = (
                os.path.dirname(os.path.abspath(__file__))
                + "/"
                + self.config.instance_set_path
            )
            keyword = "instance_set"

        self.config[keyword] = {}
        with open(path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                init_locs = [float(row[f"init_loc{i}"]) for i in range(int(row["dim"]))]
                instance = [
                    int(row["fcn_index"]),
                    int(row["dim"]),
                    float(row["init_sigma"]),
                    init_locs,
                ]
                self.config[keyword][int(row["ID"])] = instance


    # not using this
    def get_benchmark(self, seed=0):
        """
        Get benchmark from the LTO paper

        Parameters
        -------
        seed : int
            Environment seed

        Returns
        -------
        env : CMAESEnv
            CMAES environment
        """
        self.config = objdict(CMAES_DEFAULTS.copy())
        self.config.seed = seed
        self.read_instance_set()
        self.read_instance_set(test=True)
        return CMAESPopSizeEnv(self.config)
