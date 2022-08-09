import matplotlib.pyplot as plt
import numpy as np

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

class PlotStyles:
    font = {'font.family': 'Arial','font.size': 9, 'axes.titlesize': 'medium',
    'xtick.major.size': 2.5, 'xtick.major.width': 1.25,
    'ytick.major.size': 2.5, 'ytick.major.width': 1.25,
    'ytick.minor.size': 1.5, 'ytick.minor.width': 0.75,
    'axes.linewidth': 1.0}
    linewidth = 1.5
    s = 10
    alpha = 0.5

def figure_factory(figsize: tuple[int, int], use_params=False, xscale='linear', yscale='linear', xlabel='none', ylabel='none'):
    """Returns a figure with a specific style.
    
    Parameters: xscale, yscale, xlabel, ylabel."""
    plt.rcParams.update(PlotStyles.font)
    fig = plt.figure(figsize=cm2inch(figsize))

    if use_params:
        plt.xscale(xscale)
        plt.yscale(yscale)
        if xlabel != 'none':
            plt.xlabel(xlabel, labelpad=9, fontsize=9)
        if xlabel != 'none':
            plt.ylabel(ylabel, labelpad=9, fontsize=9)
    plt.tick_params(labelsize=8)
    
    return fig

def sort_kinetics(true_kinetics, predicted_kinetics):
    """Sorts predicted and true kinetics.
    
    Returns sorted true kinetics, sorted predicted kinetics"""

    sort_indices = np.argsort(true_kinetics)
    sorted_true_kinetics = np.take(true_kinetics, sort_indices)
    sorted_prediction = np.take(predicted_kinetics, sort_indices)

    return (sorted_true_kinetics, sorted_prediction)
