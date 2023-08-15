import warnings
warnings.filterwarnings("ignore")

import numpy as np

from stable_baselines3.td3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor

from dacbench.benchmarks import CMAESPopSizeBenchmark, CMAESArtificialPopSizeBenchmark

np.random.seed(15)
            
def test(benchmark):
    env = benchmark.get_environment()
    
    fid = benchmark.config.fid
    
    if not benchmark.config.test:
        print('Not test mode')
    
    model = TD3.load(f"./logs/fid{fid}/best_model", env=env)
    vec_env = model.get_env()

    obs = vec_env.reset()
    reps = 0
    while reps < 1:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated = vec_env.step(action)
                
        if terminated[0]:
            reps += 1
            vec_env.reset()
            
def main():
    """
    Main method to evaluate trained policy against PSA-CMA-ES.
    Adjust relevant parameters in modcma_popsize_benchmark and artificial_modcma
    """
    agent = CMAESPopSizeBenchmark()
    baseline = CMAESArtificialPopSizeBenchmark()
    
    print('Testing agent performance')
    test(agent)
    
    print('Testing PSA-CMA-ES performance')
    test(baseline)
    

if __name__=="__main__":
    main()