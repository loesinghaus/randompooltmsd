import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# import from local utilities
from plotting_utilities import figure_factory, PlotStyles
from regression_tree_utilities import y_transform_fixed

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

errors_mean_val = np.zeros(len(model_names))
errors_std_val = np.zeros(len(model_names))
errors_mean_test = np.zeros(len(model_names))
errors_std_test = np.zeros(len(model_names))
errors_mean_ood = np.zeros(len(model_names))
errors_std_ood = np.zeros(len(model_names))
for model_index, model_name in enumerate(model_names):
    # this should match the settings of the tree generator
    run_names = [f"{model_name}SingleR{i*11}" for i in range(1,11)]

    val_errors = []
    test_errors = []
    ood_errors = []
    for run_name in run_names:
        # -----------  read best error alpha ----------- :
        with open(f"./experiments/SingleCircuit/{model_name}/{run_name}/{run_name}_best_errors.txt", 'r') as f:
            f.readline()
            best_error_index = int(f.readline().split(',')[0])

        # ----------- load tree ----------- 
        my_model, random_seed = pickle.load(open(f"./experiments/SingleCircuit/{model_name}/{run_name}/"
        f"trees_pickle/{run_name}_alpha_{best_error_index}.p", "rb"))

        # ----------- load original training data ----------- 
        training_data_dict = pickle.load(open(f"./experiments/SingleCircuit/{model_name}/{run_name}/"
        f"{run_name}_inputs.p", "rb"))

        # Define X and y
        X_train = training_data_dict["X_train"]
        X_val = training_data_dict["X_val"]
        y_train = training_data_dict["y_train"]
        y_val = training_data_dict["y_val"]
        # load inverse transform
        y_inverse_transform = training_data_dict["y_inverse_transform"]
        transform_mean = training_data_dict["transform_mean"]
        transform_std = training_data_dict["transform_std"]

        # ----------- load test data (circuit two) -----------
        test_data = pd.read_excel("../input_data/strand_features_circuit_two.xlsx", sheet_name=model_name)
        X_test = test_data.drop("kinetics", axis=1)
        y_test = test_data["kinetics"]
        y_test_inv = y_test
        y_test = y_transform_fixed(y_test, transform_mean, transform_std)

        # ----------- load ood data ----------- 
        ood_data = pd.read_excel("../input_data/strand_features_N50.xlsx", sheet_name=model_name)

        X_ood = ood_data.drop("kinetics", axis=1)
        y_ood = ood_data["kinetics"]
        y_ood_inv = y_ood
        y_ood = y_transform_fixed(y_ood, transform_mean, transform_std)

        # calculate errors
        val_error = mean_squared_error(y_val, my_model.predict(X_val))
        ood_error = mean_squared_error(y_ood, my_model.predict(X_ood))
        test_error = mean_squared_error(y_test, my_model.predict(X_test))

        val_errors.append(val_error)
        test_errors.append(test_error)
        ood_errors.append(ood_error)

        print(f"Run name: {run_name}, {val_error}, {test_error}")
    
    # convert to numpy arrays
    val_errors = np.array(val_errors)
    test_errors = np.array(test_errors)
    ood_errors = np.array(ood_errors)

    # calculate means and standard deviations of the means
    errors_mean_val[model_index] = np.mean(val_errors)
    errors_std_val[model_index] = np.std(val_errors)/np.sqrt(len(run_names))
    errors_mean_test[model_index] = np.mean(test_errors)
    errors_std_test[model_index] = np.std(test_errors)/np.sqrt(len(run_names))
    errors_mean_ood[model_index] = np.mean(ood_errors)
    errors_std_ood[model_index] = np.std(ood_errors)/np.sqrt(len(run_names))

# ------------ create a barplot for the errors ------------
# create the figure
plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "model"}
fig = figure_factory(figsize=(15,6), **plot_params)

# convert the model index to a letter
model_indices = list(chr(i+65) for i in np.arange(0,len(model_names)))
xvals = np.arange(0, 2.25*len(model_names),2.25)
plt.xticks(xvals, model_indices)

# plot the val errors
plt.bar(x=xvals-0.5, height=errors_mean_val, edgecolor = "k", width=0.5, color='tab:red', label='validation')
plt.errorbar(x=xvals-0.5, y=errors_mean_val, yerr=errors_std_val, fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

# plot the test errors
plt.bar(x=xvals, height=errors_mean_test, edgecolor = "k", width=0.5, color='tab:olive', label='circuit two')
plt.errorbar(x=xvals, y=errors_mean_test, yerr=errors_std_test, fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

# plot the ood (N50) errors
plt.bar(x=xvals+0.5, height=errors_mean_ood, edgecolor = "k", width=0.5, color='tab:grey', label='out-of-distribution')
plt.errorbar(x=xvals+0.5, y=errors_mean_ood,yerr=errors_std_ood, fmt='none', color='k', linewidth=1, markeredgewidth=1, capsize = 1.5)

plt.text(0.05, 0.95, plot_text, fontsize=8)
plt.legend(loc='best', fontsize=9)
plt.ylim(0,1)
plt.savefig("./plots/errors_models_single_bar.svg", format='svg')
plt.close()
