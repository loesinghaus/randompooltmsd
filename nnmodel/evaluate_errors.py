import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.stats
# import from local utilities
from plotting_utilities import figure_factory, PlotStyles

# choose models to plot
model_names = ["All", "EnsembleFull", "Choice1", "Choice2", "Choice3", "Choice4", "Choice5", "Choice6",
 "Choice7", "Choice8", "Choice9", "Choice10", "Choice11", "Choice12"]
model_names = ["EnsembleFull", "Choice5", "Choice6"]
# set titles for each plot
titles = {"All": "All features", "EnsembleFull": "All ensemble features", "Choice1": "0, 2, 3, 4, 5",
"Choice2": "0, 2, 3, 4, 11", "Choice3": "0, 2, 3, 4, 8",
"Choice4": "0, 2, 3, 4, 5, 8", "Choice5": "0, 2, 4",
"Choice6": "0, 2", "Choice7": "2, 4", "Choice8": "0, 4",
"Choice9": "1, 2", "Choice10": "0", "Choice11": "2", "Choice12": "0, 1, 2, 4"}

# create text for textbox from model_names and choices
plot_text = ""
for key, entry in enumerate(titles):
    plot_text += f"{chr(65+key)}: {titles[entry]}\n"

no_of_runs = 10
errors = {data: np.zeros((len(model_names),no_of_runs)) for data in ["train", "val", "test", "ood"]}
for model_index, model_name in enumerate(model_names):

    # -----------  read errors ----------- :
    with open(f"./errors/errors_{model_name}.txt", 'r') as f:
        f.readline()
        for run_index in range(no_of_runs):
            line = f.readline().split(',')
            if line[1] != 'nan':
                errors["train"][model_index, run_index] = float(line[2])
                errors["val"][model_index, run_index] = float(line[3])
                errors["test"][model_index, run_index] = float(line[4])
                errors["ood"][model_index, run_index] = float(line[5])

# calculate means and standard deviations of the mean
means = {data: np.zeros(len(model_names)) for data in ["train", "val", "test", "ood"]}
stds = {data: np.zeros(len(model_names)) for data in ["train", "val", "test", "ood"]}

for key in means.keys():
    means[key] = np.mean(errors[key], axis=1)
    stds[key] = np.std(errors[key],axis=1)/np.sqrt(no_of_runs)

# ------------ create a barplot for the errors ------------
# create the figure
plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "model"}
fig = figure_factory(figsize=(16,6), **plot_params)

# convert the model index to a letter
model_indices = list(chr(i+65) for i in np.arange(0,len(model_names)))
xvals = np.arange(0, 2.25*len(model_names),2.25)
plt.xticks(xvals, model_indices)

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
plt.ylim(0,1)
plt.savefig("./plots/errors_models_nn_bar.svg", format='svg')
plt.close()

print(means["val"])
print(means["test"])
print(means["ood"])
print(stds["val"])
print(stds["test"])
print(stds["ood"])