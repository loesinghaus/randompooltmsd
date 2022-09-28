import json
from sklearn.model_selection import train_test_split
from plotting_utilities import figure_factory
from regression_tree_utilities import y_transform_fixed
from regression_tree_utilities import y_transform
import pandas as pd
import numpy as np
import feyn
import matplotlib.pyplot as plt
from plotting_utilities import cm2inch, PlotStyles
import matplotlib.gridspec as gridspec
import copy
import sympy

def plot_1d_response(input_data, model, figure_name, by="", output_name="", fixed_features=[""], y_inverse_transform=None):
    # sort input data
    input_data.sort_values(by=by, inplace=True)

    # process data
    by_series = input_data[by]
    fixed_input_data = pd.DataFrame(data=by_series)

    fixed_values_text = ""
    for fixed_feature in fixed_features:
        median = input_data[fixed_feature].median()
        fixed_input_data[fixed_feature] = np.ones(len(by_series))*median
        fixed_values_text += f"{fixed_feature}: {median}\n"

    fixed_values_text = fixed_values_text.strip('\n')
    predicted_values = y_inverse_transform(model.predict(fixed_input_data))

    # plot
    fig = figure_factory((8,8))
    print(by_series)
    # main plot
    ax = plt.subplot2grid(shape=(4,4),loc=(1,0),colspan=3,rowspan=3)
    ax.plot(by_series, predicted_values, color='b', linewidth=PlotStyles.linewidth, label=fixed_values_text)
    ax.scatter(by_series, y_inverse_transform(input_data[output_name]), color='grey', alpha=PlotStyles.alpha, s=PlotStyles.s, linewidths=0)
    ax.set_yscale('log')
    ax.set_ylabel(r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)", labelpad=9, fontsize=9)
    #ax.set_xlabel('Free bases in toe', labelpad=9, fontsize=9)
    ax.set_xlabel('ddG', labelpad=9, fontsize=9)
    ax.set_ylim((6E-9, 2E-3))
    #ax.set_xticks([0,5,10,15,20])
    #ax.set_xlim((-1,21))
    ax.legend(loc='best', fontsize=7)

    # top histogram
    ax_top = plt.subplot2grid(shape=(4,4),loc=(0,0),colspan=3)
    ax_top.hist(by_series, color='grey', bins=np.arange(0,21,1), rwidth=0.8, alpha=0.8)
    #ax_top.set_xticks([0,5,10,15,20])
    #ax_top.set_xlim((-1,21))
    ax_top.set_xticklabels([])
    
    # right histogram
    ax_right = plt.subplot2grid(shape=(4,4),loc=(1,3),rowspan=3)
    logarithmic_bins = [10**(x) for x in np.arange(-9,-2,0.3)]
    ax_right.hist(y_inverse_transform(input_data[output_name]), bins=logarithmic_bins, alpha=0.8, rwidth=0.8, color='grey', orientation='horizontal')
    ax_right.set_yscale('log')
    ax_right.set_ylim((6E-9, 2E-3))
    ax_right.set_yticklabels=([])
    
    plt.tick_params(labelsize=8)
    plt.savefig(figure_name)
    plt.close()
    

# ---------------- load training data ----------------
sheet_name = "Choice6"
root_directory = f"./models/with_BIC/{sheet_name}/R10"
models = [0]

input_excel_file_path = "../input_data/strand_features_both_circuits.xlsx"

df = pd.read_excel(input_excel_file_path, sheet_name=sheet_name)
# Define X and y
X_drop = df.drop("kinetics", axis=1)
# scale kinetics for easier fitting
y = df["kinetics"]
y, y_inverse_transform, transform_mean, transform_std = y_transform(y)
# merge dataframe again
input_data_full = X_drop.assign(kinetics=y)

print(transform_mean)
print(transform_std)
# ---------------- load ood data ----------------
input_excel_file_path = "../input_data/strand_features_N50.xlsx"

df = pd.read_excel(input_excel_file_path, sheet_name=sheet_name)
# Define X and y
X_ood = df.drop("kinetics", axis=1)
# scale kinetics for easier fitting
y_ood = df["kinetics"]
y_ood_inv = y_ood
y_ood = y_transform_fixed(y_ood, transform_mean, transform_std)

# merge dataframe again
input_data_ood = X_ood.assign(kinetics=y_ood)

# Apply train/val split
X_train, X_val, y_train, y_val = train_test_split(X_drop,y,test_size=0.4,random_state=40)
y_train_inv = y_inverse_transform(y_train)
y_val_inv = y_inverse_transform(y_val)

# merge dataframe again
input_data_train = X_train.assign(kinetics=y_train)
input_data_val = X_val.assign(kinetics=y_val)

for model in models:
    loaded_model = feyn.Model.load(f"{root_directory}/model{model}/model{model}.json")
    loaded_model.plot_signal(input_data_full, filename=f"./plots/model{model}_signal.html")

    # sympy model
    sympy_model = loaded_model.sympify(signif=2)
    print(sympy.latex(sympy_model))

    # plot 1D response
    # extremely dumb way to check if a parameter is in a model (there's surely a better way...)
    possible_parameters = X_drop.drop("ddG", axis=1).columns
    fixed_features = []
    for possible_parameter in possible_parameters:
        try:
            check = loaded_model.get_parameters(possible_parameter)
            fixed_features.append(possible_parameter)
        except ValueError:
            pass
    plot_1d_response(copy.deepcopy(input_data_full), model=loaded_model,
    figure_name=f"./plots/{sheet_name}_model{model}.svg",
    by="ddG", output_name="kinetics",
    fixed_features=fixed_features, y_inverse_transform=y_inverse_transform)
    