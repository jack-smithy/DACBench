
import resource
import sys
import warnings
import os

import numpy as np
import ioh
from modcma import ModularCMAES

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
        
        self.previous_obj_best = self.current_obj_best
        
        truncated = super(CMAESPopSizeEnv, self).step_()
        terminated = not self.es.step()
        
        if not (terminated or truncated):
            """Moves forward in time one step"""
            self.es.parameters.update_popsize(round(min(max(action[0], 4), 512)))
    
        self.current_obj_best = self.es.parameters.fopt

        return self.get_state(self), self.get_reward(self), terminated, truncated, {}

    def reset(self, seed=None, options={}):
        """
        Reset environment

        Returns
        -------
        np.array
            Environment state
        """
        self.previous_obj_best = 0
        self.current_obj_best = 0
        
        super(CMAESPopSizeEnv, self).reset_(seed)
        
        self.dim = self.instance[1]
        #self.fid = self.instance[0]
        self.fid = 3
        self.sigma0 = self.instance[2]
        self.x0 = self.instance[3] if len(self.instance[3])==self.dim else None
        self.lambda0 = np.random.randint(self.dim // 2, self.dim * 2)
        instance = np.random.randint(0, 20)
        
        print(f'FID{self.fid}, dim={self.dim}')
        
        self.objective = ioh.get_problem(
            fid = 3,
            dimension=self.dim,
            instance=1,
            problem_class=ioh.ProblemClass.BBOB
        )
        
        
        self.es = ModularCMAES(
            self.objective,
            self.dim,
            budget = self.budget,
            pop_size_adaptation=None,
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

        dy = self.previous_obj_best - self.current_obj_best

        return dy


    def get_default_state(self, _):
        """
        Gather state description
        """

        lam = self.es.parameters.lambda_
        pt = self.es.parameters.pnorm
        #scale_factor = self.es.parameters.expected_update_snorm()
        
        state = [
            lam,
            pt,
            #scale_factor,
            self.current_obj_best
        ]
        
        return state
