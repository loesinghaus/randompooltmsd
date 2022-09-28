import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import from local utilities
from plotting_utilities import figure_factory, PlotStyles

# choose models to plot
model_names = ["All", "EnsembleFull", "Choice1", "Choice2", "Choice3", "Choice4", "Choice5", "Choice6",
 "Choice7", "Choice8", "Choice9", "Choice10", "Choice11"]
# set titles for each plot
titles = {"All": "All features", "EnsembleFull": "All ensemble features", "Choice1": "0, 2, 3, 4, 5",
"Choice2": "0, 2, 3, 4, 11", "Choice3": "0, 2, 3, 4, 8",
"Choice4": "0, 2, 3, 4, 5, 8", "Choice5": "0, 2, 4",
"Choice6": "0, 2", "Choice7": "2, 4", "Choice8": "0, 4",
"Choice9": "1, 2", "Choice10": "0", "Choice11": "2"}

# create text for textbox from model_names and choices
plot_text = ""
for key, entry in enumerate(titles):
    plot_text += f"{chr(65+key)}: {titles[entry]}\n"

# we look at two possibilities:
# I) The best validation error among all alphas used
# II) The best validation error for a comparable tree complexity (alpha 15)
best_errors_test = []
alphafixed_errors_test = []

# create a figure for the performance across different tree sizes
plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "number of leaves"}
fig = figure_factory(figsize=(8,6), **plot_params)

for model_index, model_name in enumerate(model_names):
    # this should match the settings of the tree generator
    run_names = [f"{model_name}MixedR{i*11}" for i in range(1,11)]

    no_of_leaves = []
    train_errors = []
    test_errors = []
    for run_name in run_names:
        # read errors
        error_df = pd.read_excel(f"./experiments/mixed/{model_name}/{run_name}/{run_name}_errors_importance.xlsx", sheet_name=f"{run_name}")
        no_of_leaves.append(error_df["no of leaves"])
        train_errors.append(error_df["train_error"])
        test_errors.append(error_df["test_error"])

    no_of_leaves = np.array(no_of_leaves)
    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)

    # find the smallest error within each run
    best_errors_test.append(np.min(test_errors,axis=1))
    alphafixed_errors_test.append(test_errors[:,20])

    # average over ten runs
    mean_leaves = np.mean(no_of_leaves,axis=0)
    mean_train = np.mean(train_errors,axis=0)
    mean_test = np.mean(test_errors,axis=0)

    # plot performance at different tree sizes averaged over ten runs
    red_colors = plt.get_cmap('magma')
    blue_colors = plt.get_cmap('viridis')
    plt.plot(mean_leaves, mean_train, color=blue_colors(model_index/len(model_names)),ls='--', linewidth=PlotStyles.linewidth)
    plt.plot(mean_leaves, mean_test, color=blue_colors(model_index/len(model_names)), linewidth=PlotStyles.linewidth, label=f"model {chr(65+model_index)}")
# finish figure
plt.legend(loc='upper right', fontsize=8, ncol=1)
plt.xlim(0,30)
plt.tight_layout()
plt.savefig("./plots/errors_mean_mixed.svg", format='svg')
plt.close()

# ------------ create a barplot for the best errors ------------
# convert list to numpy arrays
best_errors_test = np.array(best_errors_test)
alphafixed_errors_test = np.array(alphafixed_errors_test)

# calculate means and standard deviations of the mean
mean_best_errors = np.mean(best_errors_test, axis=1)
std_best_errors = np.std(best_errors_test, axis=1)/np.sqrt(len(best_errors_test[:,0]))
mean_alphafixed_errors = np.mean(alphafixed_errors_test, axis=1)
std_alphafixed_errors = np.std(alphafixed_errors_test, axis=1)/np.sqrt(len(alphafixed_errors_test[:,0]))

# create the figure
plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "model"}
fig = figure_factory(figsize=(8,6), **plot_params)

# convert the model index to a letter
model_indices = list(chr(i+65) for i in np.arange(0,len(model_names)))

# plot the best errors
#plt.bar(x=model_indices, height=mean_best_errors, edgecolor = "k")
#plt.errorbar(x=model_indices, y=mean_best_errors, yerr=std_best_errors, fmt='none', color='k', markeredgewidth=1.5, capsize = 3)

# plot the errors at a fixed alpha index of 20
plt.bar(x=model_indices, height=mean_alphafixed_errors, edgecolor = "k")
plt.errorbar(x=model_indices, y=mean_alphafixed_errors,yerr=std_alphafixed_errors, fmt='none', color='k', markeredgewidth=1.5, capsize = 3)

plt.text(0.05, 0.95, plot_text, fontsize=8)
plt.xticks(model_indices)
plt.savefig("./plots/errors_models_mixed_alpha20.svg", format='svg')
plt.close()
