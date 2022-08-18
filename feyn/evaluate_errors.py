import feyn
import pandas as pd
from plotting_utilities import figure_factory, sort_kinetics, PlotStyles
import numpy as np
import matplotlib.pyplot as plt

# models to evaluate
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

# set random states
random_states = [10*i for i in np.arange(1,11)]
#random_states = [10,20]

train_errors = {model_name: np.zeros(len(random_states)) for model_name in model_names}
val_errors = {model_name: np.zeros(len(random_states)) for model_name in model_names}
test_errors = {model_name: np.zeros(len(random_states)) for model_name in model_names}
ood_errors = {model_name: np.zeros(len(random_states)) for model_name in model_names}
for model_name in model_names:
    for random_index, random_state in enumerate(random_states):
        with open(f"./models/with_BIC/{model_name}/R{random_state}/errors.txt") as f:
            f.readline()
            # find minimum of validation and training error (added)
            lines = f.readlines()
            minimum_error = 1e3
            min_error_line = ""
            for line in lines:
                line = line.split(',')
                if float(line[2])+float(line[1]) < minimum_error:
                    minimum_error = float(line[2])+float(line[1])
                    min_error_line = line
            #min_error_line = f.readline().split(',')
            train_errors[model_name][random_index] = min_error_line[1]
            val_errors[model_name][random_index] = min_error_line[2]
            test_errors[model_name][random_index] = min_error_line[3]
            ood_errors[model_name][random_index] = min_error_line[4]

means = {}            
means["train"] = [np.mean(train_errors[model_name]) for model_name in model_names]
means["val"] = [np.mean(val_errors[model_name]) for model_name in model_names]
means["test"] = [np.mean(test_errors[model_name]) for model_name in model_names]
means["ood"] = [np.mean(ood_errors[model_name]) for model_name in model_names]

stds = {}
stds["train"] = [np.std(train_errors[model_name])/np.sqrt(len(random_states)) for model_name in model_names]
stds["val"] = [np.std(train_errors[model_name])/np.sqrt(len(random_states)) for model_name in model_names]
stds["test"] = [np.std(test_errors[model_name])/np.sqrt(len(random_states)) for model_name in model_names]
stds["ood"] = [np.std(train_errors[model_name])/np.sqrt(len(random_states)) for model_name in model_names]

# ------------ create a barplot for the errors ------------
# create the figure
plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "model"}
fig = figure_factory(figsize=(5,6), **plot_params)

# convert the model index to a letter
model_indices = ["B", "G", "H"] #list(chr(i+65) for i in np.arange(0,len(model_names)))
xvals = np.arange(0, 2.25*len(model_names),2.25)
plt.xticks(xvals, model_indices)

# plot the train errors
#plt.bar(x=xvals, height=means["train"], edgecolor = "k", width=0.5, color='b', label='train')
#plt.errorbar(x=xvals, y=means["train"], yerr=stds["train"], fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

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
plt.savefig("./plots/errors_models_feyn_bar.svg", format='svg')
plt.close()

print(means["val"])
print(means["test"])
print(means["ood"])
print(stds["val"])
print(stds["test"])
print(stds["ood"])