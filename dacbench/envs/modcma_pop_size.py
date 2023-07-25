
import resource
import sys
import threading
import warnings
from collections import deque

import numpy as np
import ioh
from modcma import ModularCMAES, Parameters

from dacbench import AbstractEnv

resource.setrlimit(resource.RLIMIT_STACK, (2**35, -1))
sys.setrecursionlimit(10**9)

warnings.filterwarnings("ignore")


class CMAESPopSizeEnv(AbstractEnv):
    """
    Environment to control the step size of CMA-ES
    """

    def __init__(self, config):
        """
        Initialize CMA Env

        Parameters
        -------
        config : objdict
            Environment configuration
        """
        super(CMAESPopSizeEnv, self).__init__(config)
        self.es = None
        self.budget = config.budget

        self.get_reward = self.get_default_reward
        self.get_state = self.get_default_state

    def step(self, action):
        """
        Execute environment step

        Parameters
        ----------
        action : list
            action to execute

        Returns
        -------
        np.array, float, bool, dict
            state, reward, done, info 
        """
        
        tol = 1e-8
        
        truncated = super(CMAESPopSizeEnv, self).step_()
        terminated = not self.es.step()
        
        if not (terminated or truncated):
            """Moves forward in time one step"""
            self.es.parameters.update_popsize(round(min(max(action[0], 4), 512*10)))
            
            print(self.get_state(self))
            print(self.get_reward(self))
            print('')
            
            if abs(self.get_reward(self)) < tol:
                truncated = True
            
        return self.get_state(self), self.get_reward(self), terminated, truncated, {}

    def reset(self, seed=None, options={}):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        super(CMAESPopSizeEnv, self).reset_(seed)
        
        self.dim = self.instance[0]
        self.fid = self.instance[1]
        
        self.objective = ioh.get_problem(
            fid = self.fid,
            dimension=self.dim,
            instance=1,
            problem_class=ioh.ProblemClass.BBOB
        )
        
        self.es = ModularCMAES(
            self.objective,
            self.dim,
            budget = self.budget,
            pop_size_adaptation=None
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
        opt = self.objective.optimum
        dy =  self.objective.state.current_best.y - opt.y
        
        reward = min(self.reward_range[1], max(self.reward_range[0], -1 * dy))
        return reward

    def get_default_state(self, _):
        """
        Gather state description

        Returns
        -------
        dict
            Environment state

        """

        state = {
            "lambda_": self.es.parameters.lambda_,
            "ptnorm": self.es.parameters.pnorm,
            "normalisation_factor": self.es.parameters.expected_update_snorm(),
        }
        return state
