import warnings

warnings.filterwarnings("ignore")
import numpy as np
from pathlib import Path
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.td3 import TD3
from stable_baselines3.a2c import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise

import seaborn

seaborn.set_context("talk")

from dacbench.wrappers import PerformanceTrackingWrapper, RewardNoiseWrapper
from dacbench.logger import Logger, load_logs, log2dataframe

from dacbench.benchmarks import ModCMABenchmark, CMAESPopSizeBenchmark

bench = CMAESPopSizeBenchmark()
env = bench.get_environment()
print(env.instance_set)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

agent = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
#agent.learn(total_timesteps=10000, log_interval=50)
agent.save('test1')

vec_env = agent.get_env()

del agent

model = TD3.load("test1")

obs = vec_env.reset()
while True:
    print(obs)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
