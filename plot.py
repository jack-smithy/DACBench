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
    y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    #plt.show()
    plt.savefig("./plots/learning_curve.pdf", format="pdf")
    
#plot_results(log_dir)

def read_file(dir):
    return np.load(f'{dir}/evaluations.npz')

if __name__=="__main__":
    plot_results(log_dir)