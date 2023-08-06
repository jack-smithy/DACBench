import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3.td3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from dacbench.benchmarks import CMAESPopSizeBenchmark, CMAESArtificialPopSizeBenchmark

def test_agent():
    bench = CMAESPopSizeBenchmark()
    env = bench.get_environment()
    #env = Monitor(env, "./logs/baseline/")
    
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # agent = TD3("MlpPolicy", env, learning_rate=1e-7, action_noise=action_noise, verbose=1)
    
    model = TD3.load("./logs/best_model", env=env)
    vec_env = model.get_env()

    obs = vec_env.reset()
    
    tol = 100
    
    #terminated, truncated = False, False

    reps = 0
    while reps < 1:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = vec_env.step(action)
        if rewards > tol:
            vec_env.reset()
            reps += 1
            
def test_baseline():
    bench = CMAESArtificialPopSizeBenchmark()
    env = bench.get_environment()
    #env = Monitor(env, "./logs/baseline/")
    
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # agent = TD3("MlpPolicy", env, learning_rate=1e-7, action_noise=action_noise, verbose=1)
    
    model = TD3.load("./logs/best_model", env=env)
    vec_env = model.get_env()

    obs = vec_env.reset()
    
    tol = 100
    

    reps = 0
    while reps < 1:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = vec_env.step(action)
        if rewards > tol:
            vec_env.reset()
            reps += 1
    
        
if __name__=="__main__":
    test_baseline()
    test_agent()