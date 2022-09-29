import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import from local utilities
from plotting_utilities import figure_factory, PlotStyles

# choose models to plot
model_names = ["All", "EnsembleFull"]#, "Choice1", "Choice2", "Choice3", "Choice4", "Choice5", "Choice6",
 #"Choice7", "Choice8", "Choice9", "Choice10", "Choice11"]
# set titles for each plot
titles = {"All": "All features", "EnsembleFull": "All ensemble features", "Choice1": "0, 2, 3, 4, 5",
"Choice2": "0, 2, 3, 4, 11", "Choice3": "0, 2, 3, 4, 8",
"Choice4": "0, 2, 3, 4, 5, 8", "Choice5": "0, 2, 4",
"Choice6": "0, 2", "Choice7": "2, 4", "Choice8": "0, 4",
"Choice9": "1, 2", "Choice10": "0", "Choice11": "2"}

# set parameters for plotting
no_of_weights = 3
no_of_runs = 10
weight_decay_FLAG = False

# create text for textbox from model_names and choices
plot_text = ""
for key, entry in enumerate(titles):
    plot_text += f"{chr(65+key)}: {titles[entry]}\n"
weight_decays = np.zeros(no_of_weights)
dropout_ps = np.zeros(no_of_weights)
errors = {data: np.zeros((no_of_weights,no_of_runs)) for data in ["train", "val", "test", "ood"]}

for model_index, model_name in enumerate(model_names):
    run_names = [f"{model_name}MixedR{i*10}" for i in range(1,11)]
    
    for run_index, run_name in enumerate(run_names):
        # -----------  read errors ----------- :
        with open(f"./experiments/{model_name}/{run_name}/{run_name}_errors.txt", 'r') as f:
            f.readline()
            for weight_index in range(no_of_weights):
                line = f.readline().split(',')
                weight_decays[weight_index] = float(line[0])
                dropout_ps[weight_index] = float(line[1])
                errors["train"][weight_index, run_index] = float(line[2])
                errors["val"][weight_index, run_index] = float(line[3])
                errors["test"][weight_index, run_index] = float(line[4])
                errors["ood"][weight_index, run_index] = float(line[5])

    # calculate means and standard deviations of the mean
    means = {data: np.zeros(len(weight_decays)) for data in ["train", "val", "test", "ood"]}
    stds = {data: np.zeros(len(weight_decays)) for data in ["train", "val", "test", "ood"]}

    for key in means.keys():
        means[key] = np.mean(errors[key], axis=1)
        stds[key] = np.std(errors[key],axis=1)/np.sqrt(no_of_runs)

    # ------------ create a bar plot for the errors ------------
    # create the figure
    plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "model"}
    fig = figure_factory(figsize=(6,6), **plot_params)

    # convert the model index to a letter
    model_indices = list(chr(i+65) for i in np.arange(0,len(weight_decays)))
    xvals = np.arange(0, 2.25*len(weight_decays),2.25)
    
    if weight_decay_FLAG:
        plt.xticks(xvals, weight_decays)
    else:
        plt.xticks(xvals, dropout_ps)

    # plot the val errors
    plt.bar(x=xvals-0.5, height=means["val"], edgecolor = "k", width=0.5, color='tab:red', label='validation')
    plt.errorbar(x=xvals-0.5, y=means["val"], yerr=stds["val"], fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

    # plot the test errors
    plt.bar(x=xvals, height=means["test"], edgecolor = "k", width=0.5, color='tab:olive', label='test')
    plt.errorbar(x=xvals, y=means["test"], yerr=stds["test"], fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

    # plot the ood (N50) errors
    plt.bar(x=xvals+0.5, height=means["ood"], edgecolor = "k", width=0.5, color='tab:grey', label='out-of-distribution')
    plt.errorbar(x=xvals+0.5, y=means["ood"],yerr=stds["ood"], fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

    plt.text(0.05, 0.95, plot_text, fontsize=8)
    plt.legend(loc='best', fontsize=9)
    plt.savefig(f"./plots/errors_{model_name}_nn_bar_weight_decay.svg", format='svg')
    plt.close()
