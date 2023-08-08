from stable_baselines3.common.results_plotter import load_results, ts2xy
from matplotlib import pyplot as plt
import numpy as np


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    #y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    #plt.show()
    plt.savefig("./plots/learning_curve.pdf", format="pdf")
    
def plot_history(fid):
    precs = np.load(f"./logs/fid{fid}/training_precision.npy")
    precs = moving_average(precs, window=10)
    print(precs)
    eps = np.arange(len(precs))
    plt.plot(eps, precs)
    plt.xlabel("Episode")
    plt.ylabel("Best Precision")
    #plt.yscale("log")
    plt.savefig(f"./plots/history{fid}.pdf", format="pdf")
    
def plot_eval(fid):
    prec_psa = np.load(f"./logs/fid{fid}/prec_psa.npy")
    budget_psa = np.load(f"./logs/fid{fid}/used_budget_psa.npy")
    
    prec = np.load(f"./logs/fid{fid}/prec.npy")
    budget = np.load(f"./logs/fid{fid}/used_budget.npy")
    
    plt.plot(budget, prec)
    plt.plot(budget_psa, prec_psa)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(f"./plots/eval_psa{fid}.pdf", format="pdf")
    
def plot_lambdas(fid):
    lambda_psa = np.load(f"./logs/fid{fid}/lambda_psa.npy")
    budget_psa = np.load(f"./logs/fid{fid}/used_budget_psa.npy")
    
    lambda_ = np.load(f"./logs/fid{fid}/lambda.npy")
    budget = np.load(f"./logs/fid{fid}/used_budget.npy")
    
    plt.plot(budget, lambda_)
    plt.plot(budget_psa, lambda_psa)
    #plt.yscale("log")
    plt.xscale("log")
    plt.savefig(f"./plots/lambda_psa{fid}.pdf", format="pdf")

if __name__=="__main__":
    fid = 1    
    plot_history(fid)