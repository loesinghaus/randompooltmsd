import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats
# import from local utilities
from plotting_utilities import figure_factory, PlotStyles
from regression_tree_utilities import y_transform, y_transform_fixed

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

errors_mean_train = np.zeros(len(model_names))
errors_std_train = np.zeros(len(model_names))
errors_mean_val = np.zeros(len(model_names))
errors_std_val = np.zeros(len(model_names))
errors_mean_test = np.zeros(len(model_names))
errors_std_test = np.zeros(len(model_names))
for model_index, model_name in enumerate(model_names):
    # this should match the settings of the tree generator
    run_names = [f"{model_name}SingleR{i*11}" for i in range(1,11)]

    best_average_index = 0
    # find best index average over runs
    for run_name in run_names:
         # -----------  read best error alpha ----------- :
        with open(f"./experiments/SingleCircuit/{model_name}/{run_name}/{run_name}_best_errors.txt", 'r') as f:
            f.readline()
            best_error_index = int(f.readline().split(',')[0])
            best_average_index += best_error_index
    best_average_index = int(best_average_index/len(run_names))
    print(best_average_index)

    train_errors = []
    val_errors = []
    test_errors = []
    for run_name in run_names:
        # ----------- load tree ----------- 
        my_model, random_seed = pickle.load(open(f"./experiments/SingleCircuit/{model_name}/{run_name}/"
        f"trees_pickle/{run_name}_alpha_{best_average_index}.p", "rb"))

        # ----------- load original training data ----------- 
        training_data = pd.read_excel("../input_data/strand_features_circuit_one.xlsx", sheet_name=model_name)

        # Define X and y
        X = training_data.drop("kinetics", axis=1)
        # scale kinetics for easier fitting
        y = training_data["kinetics"]
        y, y_inverse_transform, train_mean, train_std = y_transform(y)

        # use same training/test split as in the original training
        X_train, _, y_train, _ = train_test_split(X,y,test_size=0.3,random_state=random_seed)
        y_train_inv = y_inverse_transform(y_train)

        # ----------- load circuit two data ----------- 
        circuit_two_data = pd.read_excel("../input_data/strand_features_circuit_two.xlsx", sheet_name=model_name)
        X_val = circuit_two_data.drop("kinetics", axis=1)
        y_val = circuit_two_data["kinetics"]
        y_val_inv = y_val
        y_val = y_transform_fixed(y_val, train_mean, train_std)

        # ----------- load test data ----------- 
        test_data = pd.read_excel("../input_data/strand_features_N50.xlsx", sheet_name=model_name)

        X_test = test_data.drop("kinetics", axis=1)
        y_test = test_data["kinetics"]
        y_test_inv = y_test
        y_test = y_transform_fixed(y_test, train_mean, train_std)
 
        # ----------- make predictions using the tree ----------- 
        prediction_train = y_inverse_transform(my_model.predict(X_train))
        prediction_val = y_inverse_transform(my_model.predict(X_val))
        prediction_test = y_inverse_transform(my_model.predict(X_test))

        # calculate errors and the spearman correlation
        train_error = mean_squared_error(y_train, my_model.predict(X_train))
        #train_spearman,_ = scipy.stats.spearmanr(y_train, my_model.predict(X_train)) 
        test_error = mean_squared_error(y_test, my_model.predict(X_test))
        #test_spearman,_ = scipy.stats.spearmanr(y_test, my_model.predict(X_test))
        val_error = mean_squared_error(y_val, my_model.predict(X_val))
        #val_spearman,_ = scipy.stats.spearmanr(y_val, my_model.predict(X_val))

        train_errors.append(train_error)
        val_errors.append(val_error)
        test_errors.append(test_error)
    
    # convert to numpy arrays
    train_errors = np.array(train_errors)
    val_errors = np.array(val_errors)
    test_errors = np.array(test_errors)

    # calculate means and standard deviations of the mean
    errors_mean_train[model_index] = np.mean(train_errors)
    errors_std_train[model_index] = np.std(train_errors)/np.sqrt(len(train_errors))
    errors_mean_val[model_index] = np.mean(val_errors)
    errors_std_val[model_index] = np.std(val_errors)
    errors_mean_test[model_index] = np.mean(test_errors)
    errors_std_test[model_index] = np.std(test_errors)#/np.sqrt(len(test_errors))

# ------------ create a barplot for the errors ------------
# create the figure
plot_params = {"use_params": True, "ylabel": "mean squared error", "xlabel": "model"}
fig = figure_factory(figsize=(12,6), **plot_params)

# convert the model index to a letter
model_indices = list(chr(i+65) for i in np.arange(0,len(model_names)))
xvals = np.arange(0, len(model_names))
plt.xticks(xvals, model_indices)

# plot the val (circuit two) errors
plt.bar(x=xvals-0.15, height=errors_mean_val, edgecolor = "k", width=0.3, color='r', label='circuit two')
plt.errorbar(x=xvals-0.15, y=errors_mean_val, yerr=errors_std_val, fmt='none', color='k', markeredgewidth=1.0, capsize = 2)

# plot the test (N50) errors
plt.bar(x=xvals+0.15, height=errors_mean_test, edgecolor = "k", width=0.3, color='grey', label='out-of-distribution')
plt.errorbar(x=xvals+0.15, y=errors_mean_test,yerr=errors_std_test, fmt='none', color='k', markeredgewidth=1.0, capsize = 2)

plt.text(0.05, 0.95, plot_text, fontsize=8)
plt.legend(loc='best', fontsize=9)
plt.savefig("./plots/errors_models_single_test.svg", format='svg')
plt.close()
