import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3.td3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from dacbench.benchmarks import CMAESPopSizeBenchmark

bench = CMAESPopSizeBenchmark()
env = bench.get_environment()
env = Monitor(env, "./logs/")

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

agent = TD3("MlpPolicy", env, learning_rate=1e-7, action_noise=action_noise, verbose=1)

vec_env = agent.get_env()

model = TD3.load("./logs/best_model")

obs = vec_env.reset()
print(obs)

terminated, truncated = False, False

while True:
    while not (terminated or truncated):
        action, _states = model.predict(obs)
        print(action.flatten())
        obs, rewards, terminated, truncated, info = env.step(action.flatten())
    vec_env.reset()