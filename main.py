import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.td3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from dacbench.wrappers import PerformanceTrackingWrapper, RewardNoiseWrapper
from dacbench.logger import Logger, load_logs, log2dataframe

from dacbench.benchmarks import CMAESPopSizeBenchmark

bench = CMAESPopSizeBenchmark()
env = bench.get_environment()
env = Monitor(env, "./logs/")

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=100,
                             deterministic=True, render=False)

agent = TD3("MlpPolicy",
            env, 
            learning_rate=1e-3,
            gamma=0.98,
            buffer_size=200000,
            gradient_steps=-1,
            train_freq=(5, "step"),
            action_noise=action_noise,
            verbose=1)

agent.learn(total_timesteps=3e5, callback=eval_callback, progress_bar=True)
agent.save('./logs/test1')

vec_env = agent.get_env()

del agent

model = TD3.load("test1")

print(vec_env)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs[0])
    obs, rewards, dones, info = env.step(action)
