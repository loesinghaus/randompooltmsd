import pickle
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from functools import partial
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from plotting_utilities import PlotStyles, cm2inch
from regression_tree_utilities import y_transform, y_transform_fixed


# Choose which models to plot
#model_names = ["All", "EnsembleFull", "Choice1", "Choice2", "Choice3", "Choice4", "Choice5", "Choice6",
# "Choice7", "Choice8", "Choice9", "Choice10", "Choice11"]
model_names = ["EnsembleFull", "Choice5", "Choice6", "Choice8", "Choice10", "Choice11"]
model_names = ["EnsembleFull", "Choice5", "Choice6", "Choice11"]
# Set titles for the different plots
titles = {"All": "A: All features", "EnsembleFull": "B: All ensemble features", "Choice1": "C: 0, 2, 3, 4, 5",
"Choice2": "D: 0, 2, 3, 4, 11", "Choice3": "E: 0, 2, 3, 4, 8",
"Choice4": "F: 0, 2, 3, 4, 5, 8", "Choice5": "G: 0, 2, 4",
"Choice6": "H: 0, 2", "Choice7": "I: 2, 4", "Choice8": "J: 0, 4",
"Choice9": "K: 1, 2", "Choice10": "L: 0", "Choice11": "M: 2"}

# the script plots either the train/test set or out-of-distribution data
plot_train_test = True

# set plot parameters
plt.rcParams.update(PlotStyles.font)
# 16, 14 for 6 plots
# 16, 10 for 4 plots
fig, axs = plt.subplots(2,2,sharex=True,sharey=True, figsize = cm2inch(16,10))
plt.yscale('log')

# loop over models for one specific random seed
for model_index, model_name in enumerate(model_names):
    # -----------  read best error alpha ----------- :
    with open(f"./experiments/Mixed/{model_name}/{model_name}MixedR11/{model_name}MixedR11_best_errors.txt", 'r') as f:
        f.readline()
        best_error_index = int(f.readline().split(',')[0])
        print(best_error_index)
        i = best_error_index

    # ----------- load tree ----------- 
    my_model, random_seed = pickle.load(open(f"./experiments/Mixed/{model_name}/{model_name}MixedR11/"
    f"trees_pickle/{model_name}MixedR11_alpha_{best_error_index}.p", "rb"))

    # ----------- load original training data ----------- 
    training_data = pd.read_excel("../input_data/strand_features_both_circuits.xlsx", sheet_name=model_name)

    # Define X and y
    X = training_data.drop("kinetics", axis=1)
    # scale kinetics for easier fitting
    y = training_data["kinetics"]
    y, y_inverse_transform, train_mean, train_std = y_transform(y)

    # use same training/test split as in the original training
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.3,random_state=random_seed)
    y_train_inv = y_inverse_transform(y_train)
    y_val_inv = y_inverse_transform(y_val)

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
    train_spearman,_ = scipy.stats.spearmanr(y_train, my_model.predict(X_train))
    test_error = mean_squared_error(y_test, my_model.predict(X_test))
    test_spearman,_ = scipy.stats.spearmanr(y_test, my_model.predict(X_test))
    val_error = mean_squared_error(y_val, my_model.predict(X_val))
    val_spearman,_ = scipy.stats.spearmanr(y_val, my_model.predict(X_val))
    
    # ----------- write results to excel sheet ----------- 
    output_df = pd.DataFrame()
    output_df.loc[model_name, "test"+str(i)] = test_error
    output_df.loc[model_name, "testRho^2"+str(i)] = test_spearman**2
    output_df.loc[model_name, "val"+str(i)] = val_error
    output_df.loc[model_name, "valRho^2"+str(i)] = val_spearman**2

    # ----------- prepare data for plotting ----------- 
    # sort training data
    sort_indices_train = np.argsort(y_train_inv)
    sorted_y_train = np.take(y_train_inv, sort_indices_train)
    sorted_prediction_train = np.take(prediction_train, sort_indices_train)
    # sort val data
    sort_indices_val = np.argsort(y_val_inv)
    sorted_y_val = np.take(y_val_inv, sort_indices_val)
    sorted_prediction_val = np.take(prediction_val, sort_indices_val)
    # sort test data
    sort_indices_test = np.argsort(y_test_inv)
    sorted_y_test = np.take(y_test_inv, sort_indices_test)
    sorted_prediction_test = np.take(prediction_test, sort_indices_test)

    # ----------- plotting -----------     
    ax = axs[int(model_index/2.0),model_index%2]
    size_params = {"lw": PlotStyles.linewidth, "s": PlotStyles.s, "alpha": PlotStyles.alpha}
    if plot_train_test:
        ax.set(ylim=(6E-9, 2E-3),
        title=titles[model_name])
        x_vals_train = np.arange(0,len(sorted_y_train))/len(sorted_y_train)
        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
        ax.scatter(x_vals_train, sorted_prediction_train, color='b', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.scatter(x_vals_val, sorted_prediction_val, color='r', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.plot(x_vals_train, sorted_y_train, linewidth=size_params["lw"], color='b', label=f"training, mse={train_error:.2f}")
        ax.plot(x_vals_val, sorted_y_val, color='r', linewidth=size_params["lw"], label=f"validation, mse={val_error:.2f}")
    else:
        ax.set(ylim=(1E-7, 2E-3),
        title=titles[model_name])
        x_vals_test = np.arange(0,1,1.0/len(sorted_y_test))
        ax.scatter(x_vals_test, sorted_prediction_test, color='k', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.plot(x_vals_test, sorted_y_test, linewidth=size_params["lw"], color='k', label=f"out-of-distribution data, mse={test_error:.2f}")

    ax.legend(loc='lower right', fontsize=8)
    if model_index%2 == 0:
        ax.set_ylabel(r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)", fontsize=9, labelpad = 9)
    if int(model_index/2.0) == int(len(model_names)/2-1):
        ax.set_xlabel("interfering strands", fontsize=9, labelpad = 9)

plt.tight_layout()
if plot_train_test:
    plt.savefig(f"./plots/tree_eval_R11_train_val.svg", format='svg')
else:
    plt.savefig(f"./plots/tree_eval_R11_ood.svg", format='svg')
plt.close()
     
#output_df.to_excel(f"eval_mixed_R11.xlsx")