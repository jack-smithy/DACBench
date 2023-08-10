import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3.td3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from dacbench.benchmarks import CMAESPopSizeBenchmark, CMAESArtificialPopSizeBenchmark

np.random.seed()

def test_agent():
    bench = CMAESPopSizeBenchmark()
    env = bench.get_environment()
    
    fid = bench.config.fid
    
    if not bench.config.test:
        print('Not test mode')
    
    model = TD3.load(f"./logs/fid{fid}/best_model", env=env)
    vec_env = model.get_env()

    
    tol = 100
    
    #terminated, truncated = False, False

    reps = 0
    
    obs = vec_env.reset()
    while reps < 1:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = vec_env.step(action)
        
        if terminated[0]:
            reps += 1
            vec_env.reset()
            
def test_baseline():
    bench = CMAESArtificialPopSizeBenchmark()
    env = bench.get_environment()
    
    fid = bench.config.fid
    
    if not bench.config.test:
        print('Not test mode')
    
    model = TD3.load(f"./logs/fid{fid}/best_model", env=env)
    vec_env = model.get_env()

    obs = vec_env.reset()
    
    tol = 100
    
    reps = 0
    while reps < 1:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = vec_env.step(action)
                
        if terminated[0]:
            reps += 1
            vec_env.reset()

if __name__=="__main__":

    test_baseline()
    test_agent()