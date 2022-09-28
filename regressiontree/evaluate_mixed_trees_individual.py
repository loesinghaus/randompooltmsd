import pickle
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from plotting_utilities import PlotStyles, cm2inch, sort_kinetics
from regression_tree_utilities import y_transform_fixed

# the script plots either the train/val/test set or out-of-distribution data
plot_train_val_test = True

# Choose which models to plot
#model_names = ["All", "EnsembleFull", "Choice1", "Choice2", "Choice3", "Choice4", "Choice5", "Choice6",
# "Choice7", "Choice8", "Choice9", "Choice10", "Choice11"]
model_names = ["EnsembleFull", "Choice5", "Choice6", "Choice8", "Choice10", "Choice11"]
#model_names = ["EnsembleFull", "Choice5", "Choice6", "Choice11"]
# Set titles for the different plots
titles = {"All": "A: All features", "EnsembleFull": "B: All ensemble features", "Choice1": "C: 0, 2, 3, 4, 5",
"Choice2": "D: 0, 2, 3, 4, 11", "Choice3": "E: 0, 2, 3, 4, 8",
"Choice4": "F: 0, 2, 3, 4, 5, 8", "Choice5": "G: 0, 2, 4",
"Choice6": "H: 0, 2", "Choice7": "I: 2, 4", "Choice8": "J: 0, 4",
"Choice9": "K: 1, 2", "Choice10": "L: 0", "Choice11": "M: 2"}

# set plot parameters
plt.rcParams.update(PlotStyles.font)
# 16, 14 for 6 plots
# 16, 10 for 4 plots
fig, axs = plt.subplots(3,2,sharex=True,sharey=True, figsize = cm2inch(16,14))
plt.yscale('log')

# loop over models for one specific random seed
for model_index, model_name in enumerate(model_names):
    # set the random seed to be used
    RX = "R33"
    run_name = f"{model_name}Mixed{RX}"

    # -----------  read best error alpha ----------- :
    with open(f"./experiments/Mixed/{model_name}/{run_name}/{run_name}_best_errors.txt", 'r') as f:
        f.readline()
        best_error_index = int(f.readline().split(',')[0])
        print(best_error_index)

    # ----------- load tree ----------- 
    my_model, random_seed = pickle.load(open(f"./experiments/Mixed/{model_name}/{run_name}/"
    f"trees_pickle/{run_name}_alpha_{best_error_index}.p", "rb"))

    # ----------- load original training data ----------- 
    training_data_dict = pickle.load(open(f"./experiments/Mixed/{model_name}/{run_name}/"
    f"{run_name}_inputs.p", "rb"))

    # Define X and y
    X_train = training_data_dict["X_train"]
    X_val = training_data_dict["X_val"]
    X_test = training_data_dict["X_test"]
    y_train = training_data_dict["y_train"]
    y_val = training_data_dict["y_val"]
    y_test = training_data_dict["y_test"]
    # load inverse transform
    y_inverse_transform = training_data_dict["y_inverse_transform"]
    transform_mean = training_data_dict["transform_mean"]
    transform_std = training_data_dict["transform_std"]
    # execute inverse transform
    y_train_inv = y_inverse_transform(y_train)
    y_val_inv = y_inverse_transform(y_val)
    y_test_inv = y_inverse_transform(y_test)

    # ----------- load ood data ----------- 
    ood_data = pd.read_excel("../input_data/strand_features_N50.xlsx", sheet_name=model_name)

    X_ood = ood_data.drop("kinetics", axis=1)
    y_ood = ood_data["kinetics"]
    y_ood_inv = y_ood
    y_ood = y_transform_fixed(y_ood, transform_mean, transform_std)
    
    # ----------- make predictions using the tree ----------- 
    prediction_train = y_inverse_transform(my_model.predict(X_train))
    prediction_val = y_inverse_transform(my_model.predict(X_val))
    prediction_test = y_inverse_transform(my_model.predict(X_test))
    prediction_ood = y_inverse_transform(my_model.predict(X_ood))

    # calculate errors and the spearman correlation
    train_error = mean_squared_error(y_train, my_model.predict(X_train))
    train_spearman,_ = scipy.stats.spearmanr(y_train, my_model.predict(X_train))
    ood_error = mean_squared_error(y_ood, my_model.predict(X_ood))
    ood_spearman,_ = scipy.stats.spearmanr(y_ood, my_model.predict(X_ood))
    val_error = mean_squared_error(y_val, my_model.predict(X_val))
    val_spearman,_ = scipy.stats.spearmanr(y_val, my_model.predict(X_val))
    test_error = mean_squared_error(y_test, my_model.predict(X_test))
    test_spearman,_ = scipy.stats.spearmanr(y_test, my_model.predict(X_test))
    
    # ----------- write results to excel sheet ----------- 
    """output_df = pd.DataFrame()
    output_df.loc[model_name, "ood"+str(i)] = ood_error
    output_df.loc[model_name, "oodRho^2"+str(i)] = ood_spearman**2
    output_df.loc[model_name, "val"+str(i)] = val_error
    output_df.loc[model_name, "valRho^2"+str(i)] = val_spearman**2"""

    # ----------- prepare data for plotting ----------- 
    # sort data
    sorted_y_train,sorted_prediction_train = sort_kinetics(y_train_inv, prediction_train)
    sorted_y_val,sorted_prediction_val = sort_kinetics(y_val_inv, prediction_val)
    sorted_y_test,sorted_prediction_test = sort_kinetics(y_test_inv, prediction_test)
    sorted_y_ood,sorted_prediction_ood = sort_kinetics(y_ood_inv, prediction_ood)

    # ----------- plotting -----------     
    ax = axs[int(model_index/2.0),model_index%2]
    size_params = {"lw": PlotStyles.linewidth, "s": PlotStyles.s, "alpha": PlotStyles.alpha}
    if plot_train_val_test:
        ax.set(ylim=(6E-9, 2E-3),
        title=titles[model_name])
        x_vals_train = np.arange(0,len(sorted_y_train))/len(sorted_y_train)
        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
        x_vals_test= np.arange(0,1,1.0/len(sorted_y_test))
        ax.scatter(x_vals_train, sorted_prediction_train, color='b', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.scatter(x_vals_val, sorted_prediction_val, color='tab:red', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.scatter(x_vals_test, sorted_prediction_test, color='tab:olive', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.plot(x_vals_train, sorted_y_train, linewidth=size_params["lw"], color='b', label=f"training, mse={train_error:.2f}")
        ax.plot(x_vals_val, sorted_y_val, color='tab:red', linewidth=size_params["lw"], label=f"validation, mse={val_error:.2f}")
        ax.plot(x_vals_test, sorted_y_test, color='tab:olive', linewidth=size_params["lw"], label=f"test, mse={test_error:.2f}")
    else:
        ax.set(ylim=(1E-7, 2E-3),
        title=titles[model_name])
        x_vals_ood = np.arange(0,1,1.0/len(sorted_y_ood))
        ax.scatter(x_vals_ood, sorted_prediction_ood, color='k', alpha=size_params["alpha"], s=size_params["s"],linewidths=0)
        ax.plot(x_vals_ood, sorted_y_ood, linewidth=size_params["lw"], color='k', label=f"out-of-distribution data, mse={ood_error:.2f}")

    ax.legend(loc='lower right', fontsize=8)
    # set x and y axis location
    if model_index%2 == 0:
        ax.set_ylabel(r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)", fontsize=9, labelpad = 9)
    if int(model_index/2.0) == int(len(model_names)/2-1):
        ax.set_xlabel("interfering strands", fontsize=9, labelpad = 9)

plt.tight_layout()
if plot_train_val_test:
    plt.savefig(f"./plots/mixed_tree_eval_{RX}_train_val_test.svg", format='svg')
else:
    plt.savefig(f"./plots/mixed_tree_eval_{RX}_ood.svg", format='svg')
plt.close()
     
#output_df.to_excel(f"eval_mixed_R11.xlsx")