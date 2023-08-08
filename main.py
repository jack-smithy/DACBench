import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3.td3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from dacbench.benchmarks import CMAESPopSizeBenchmark


bench = CMAESPopSizeBenchmark()
env = bench.get_benchmark()
fid = bench.config.fid

if bench.config.test:
    print('testing mode - stop now')

env = Monitor(env, f"./logs/fid{fid}/")

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

eval_callback = EvalCallback(env,
                             best_model_save_path=f"./logs/fid{fid}/",
                             log_path=f"./logs/fid{fid}", 
                             eval_freq=100,
                             n_eval_episodes=5,
                             deterministic=True, 
                             render=False)


agent = TD3("MlpPolicy",
            env, 
            learning_rate=5e-5,
            action_noise=action_noise,
            verbose=1)


agent.learn(total_timesteps=100000, callback=eval_callback, progress_bar=True)
agent.save(f'./logs/fid{fid}/final_model')
env.close()

