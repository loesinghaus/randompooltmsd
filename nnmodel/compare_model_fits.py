import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import from local utilities
from plotting_utilities import cm2inch, PlotStyles

# choose models to plot
#model_names = ["EnsembleFull", "Choice5", "Choice6", "Choice8", "Choice11", "Choice12"]
model_names = ["EnsembleFull", "Choice1", "Choice5", "Choice6", "Choice8", "Choice11"]
# set titles for each plot
titles = {"All": "A: All features", "EnsembleFull": "B: All ensemble features", "Choice1": "C: 0, 2, 3, 4, 5",
"Choice2": "D: 0, 2, 3, 4, 11", "Choice3": "E: 0, 2, 3, 4, 8",
"Choice4": "F: 0, 2, 3, 4, 5, 8", "Choice5": "G: 0, 2, 4",
"Choice6": "H: 0, 2", "Choice7": "I: 2, 4", "Choice8": "J: 0, 4",
"Choice9": "K: 1, 2", "Choice10": "L: 0", "Choice11": "M: 2", "Choice12": "N: 0, 1, 2, 4"}

# set plot parameters
plot_train_val_test = False
RX = "R30"
plt.rcParams.update(PlotStyles.font)
# 16, 14 for 6 plots
# 16, 10 for 4 plots
fig, axs = plt.subplots(3,2,sharex=True,sharey=True, figsize = cm2inch(16,14))
plt.yscale('log')

# loop over models for one specific random seed
for model_index, model_name in enumerate(model_names):
    run_name = f"{model_name}Mixed{RX}"

    # -----------  read data ----------- :
    sorted_pairs = pickle.load(open(f"./experiments/{model_name}/{run_name}/{run_name}_p0.25_sorted_pairs", 'rb'))
    # read errors
    with open(f"./errors/errors_{model_name}.txt", 'r') as f:
        f.readline()
        for line in f:
            line = line.split(',')
            if line[0] == run_name:
                errors = [float(entry) for entry in line[2:]]
                break

    # ----------- plotting -----------     
    ax = axs[int(model_index/2.0),model_index%2]
    size_params = {"lw": PlotStyles.linewidth, "s": PlotStyles.s, "alpha": PlotStyles.alpha}

    if plot_train_val_test:
        ax.set(ylim=(6E-9, 2E-3),
        title=titles[model_name])
        # plot training data
        sorted_y_train, sorted_prediction_train = sorted_pairs[0]
        x_vals_train = np.arange(0,1,1.0/len(sorted_y_train))
        ax.plot(x_vals_train, sorted_y_train, color='b', label=f'training, mse={errors[0]:.2f}', linewidth=size_params["lw"])
        ax.scatter(x_vals_train, sorted_prediction_train, color='b', s=size_params["s"], alpha=size_params["alpha"], linewidths=0)

        # plot validation data
        sorted_y_val, sorted_prediction_val = sorted_pairs[1]
        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
        ax.plot(x_vals_val, sorted_y_val, color='r', label=f'validation, mse={errors[1]:.2f}', linewidth=size_params["lw"])
        ax.scatter(x_vals_val, sorted_prediction_val, color='r', s=size_params["s"], alpha=size_params["alpha"], linewidths=0)

        # plot test data
        sorted_y_test, sorted_prediction_test = sorted_pairs[2]
        x_vals_test = np.arange(0,1,1.0/len(sorted_y_test))
        ax.plot(x_vals_test, sorted_y_test, color='tab:olive', label=f'test, mse={errors[2]:.2f}', linewidth=size_params["lw"])
        ax.scatter(x_vals_test, sorted_prediction_test, color='tab:olive', s=size_params["s"], alpha=size_params["alpha"], linewidths=0)
    
    else:
        # plot ood data
        ax.set(ylim=(1E-7, 2E-3),
        title=titles[model_name])

        sorted_y_ood, sorted_prediction_ood = sorted_pairs[3]
        x_vals_ood = np.arange(0,1,1.0/len(sorted_y_ood))
        ax.plot(x_vals_ood, sorted_y_ood, color='k', label=f'out-of-distribution, mse={errors[3]:.2f}', linewidth=1.5)
        ax.scatter(x_vals_ood, sorted_prediction_ood, color='k', s=10, alpha=0.5, linewidths=0)

    ax.legend(loc='lower right', fontsize=8)
    if model_index%2 == 0:
        ax.set_ylabel(r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)", fontsize=9, labelpad = 9)
    if int(model_index/2.0) == int(len(model_names)/2-1):
        ax.set_xlabel("interfering strands", fontsize=9, labelpad = 9)

plt.tight_layout()
if plot_train_val_test:
    plt.savefig(f"./plots/nn_eval_{RX}_train_val_test.svg", format='svg')
else:
    plt.savefig(f"./plots/nn_eval_{RX}_ood.svg", format='svg')
plt.close()