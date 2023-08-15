import resource
import sys
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
import ioh
from modcma import ModularCMAES

from dacbench import AbstractEnv

resource.setrlimit(resource.RLIMIT_STACK, (2**35, -1))
sys.setrecursionlimit(10**9)

warnings.filterwarnings("ignore")


class CMAESArtificialPopSizeEnv(AbstractEnv):
    """
    Environment to record the population size in PSA-CMA-ES
    """

    def __init__(self, config):
        """
        Initialize CMA Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(CMAESArtificialPopSizeEnv, self).__init__(config)
        self.es = None
        
        self.budget = config.budget
        self.fid = config.fid
        self.test = config.test        

        self.get_reward = self.get_default_reward
        self.get_state = self.get_default_state
        
        if self.test:
            self.run_history = np.array([])
            self.used_budget = np.array([])
            self.lambda_history = np.array([])
            
        else:
            self.hist = np.array([])
        


    def step(self, _):
        """
        Execute environment step

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info
        """

        truncated = super(CMAESArtificialPopSizeEnv, self).step_()
        terminated = not self.es.step()

        if not (terminated or truncated):
            pass
        else:
            if not self.test:
                self.hist = np.append(self.hist, self.current_precision)
                print(self.current_precision)
                np.save(f'./logs/fid{self.fid}/training_precision', self.hist)
                
        return self.get_state(self), self.get_reward(self), terminated, truncated, {}

    def reset(self, seed=None, options={}):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        
        np.random.seed(seed)
            
        self.current_precision = np.inf

        super(CMAESArtificialPopSizeEnv, self).reset_(seed)
        
        if self.test:
            np.save(f"logs/fid{self.fid}/prec_psa", self.run_history)
            np.save(f"logs/fid{self.fid}/used_budget_psa", self.used_budget)
            np.save(f"logs/fid{self.fid}/lambda_psa", self.lambda_history)

        # self.fid = self.instance[0]
        self.dim = self.instance[1]
        self.sigma0 = self.instance[2]
        self.x0 = np.array(self.instance[3]) if len(self.instance[3]) == self.dim else None

        print(f"FID{self.fid}, dim={self.dim}")

        self.objective = ioh.get_problem(
            fid=self.fid, dimension=self.dim, instance=1, problem_class=ioh.ProblemClass.BBOB
        )

        self.es = ModularCMAES(
            self.objective,
            self.dim,
            budget=self.budget,
            pop_size_adaptation='psa',
            max_lambda_=512,
            min_lambda_=10
        )

        return self.get_state(self), {}

    def close(self):
        """
        No additional cleanup necessary

        Returns
        -------
        bool
            Cleanup flag
        """
        
        print('closed')
        
        return True

    def render(self, mode: str = "human"):
        """
        Render env in human mode

        Parameters
        ----------
        mode : str
            Execution mode
        """
        if mode != "human":
            raise NotImplementedError

        pass

    def get_default_reward(self, _):
        """
        Compute reward

        Returns
        -------
        float
            Reward

        """
        
        self.current_precision = self.objective.state.current_best.y - self.objective.optimum.y
        
        if self.test:
            self.run_history = np.append(self.run_history, self.current_precision)
            self.used_budget = np.append(self.used_budget, self.es.parameters.used_budget)
            self.lambda_history = np.append(self.lambda_history, self.es.parameters.lambda_)
        
        reward = -1 * np.log(self.current_precision)

        return min(reward, 10**12)

    def get_default_state(self, _):
        """
        Gather state description
        """

        lam = self.es.parameters.lambda_
        pt = self.es.parameters.pnorm
        # ps = np.linalg.norm(self.es.parameters.ps)
        # pc = np.linalg.norm(self.es.parameters.pc)
        scale_factor = self.es.parameters.expected_update_snorm()

        state = [
            lam,
            pt,
            scale_factor,
            # ps,
            # pc,
        ]

        return state
