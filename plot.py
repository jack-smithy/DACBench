from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib
#matplotlib.use("GTK3Agg")
from matplotlib import pyplot as plt
import numpy as np

log_dir = "./logs/"
# Helper from the library
#results_plotter.plot_results(dirs=[log_dir], num_timesteps=1e5, x_axis='timesteps', task_name="Training")


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
    
def plot_history(file):
    arr = np.load(file)
    plt.plot(arr)
    plt.yscale("log")
    plt.savefig("./plots/history.pdf", format="pdf")
    
def plot_eval(log_dir, fid):
    prec_psa = np.load(f"{log_dir}/prec_psa.npy")
    budget_psa = np.load(f"{log_dir}/used_budget_psa.npy")
    
    prec = np.load(f"{log_dir}/prec.npy")
    budget = np.load(f"{log_dir}/used_budget.npy")
    
    plt.plot(budget, prec)
    plt.plot(budget_psa, prec_psa)
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(f"./plots/eval_psa{fid}.pdf", format="pdf")
    
def plot_lambdas(log_dir, fid):
    lambda_psa = np.load(f"{log_dir}/lambda_psa.npy")
    budget_psa = np.load(f"{log_dir}/used_budget_psa.npy")
    
    lambda_ = np.load(f"{log_dir}/lambda.npy")
    budget = np.load(f"{log_dir}/used_budget.npy")
    
    plt.plot(budget, lambda_)
    plt.plot(budget_psa, lambda_psa)
    #plt.yscale("log")
    plt.xscale("log")
    plt.savefig(f"./plots/lambda_psa{fid}.pdf", format="pdf")

def read_file(dir):
    return np.load(f'{dir}/evaluations.npz')

if __name__=="__main__":
    #plot_results(log_dir)
    #plot_history('history.npy')
    
    #plot_eval("./logs/fid3", 3)
    plot_lambdas("./logs/fid2", 2)