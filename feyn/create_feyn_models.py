import feyn
import pandas as pd
from regression_tree_utilities import y_transform_fixed
from regression_tree_utilities import y_transform
from plotting_utilities import figure_factory, sort_kinetics, PlotStyles
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_parameters(model, possible_parameters):
    outputs = []
    for possible_parameter in possible_parameters:
        try:
            model.get_parameters(possible_parameter)
            outputs.append(possible_parameter)
        except ValueError:
            pass
    return outputs

# ---------------- load training data ----------------
sheet_name = "Choice6"
input_excel_file_path = "../input_data/strand_features_both_circuits.xlsx"

df = pd.read_excel(input_excel_file_path, sheet_name=sheet_name)
# Define X and y
X_drop = df.drop("kinetics", axis=1)
# scale kinetics for easier fitting
y = df["kinetics"]
y, y_inverse_transform, transform_mean, transform_std = y_transform(y)

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

random_states = [10*i for i in np.arange(1,11)]

for random_state in random_states:
    # Apply train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_drop,y,test_size=0.4,random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5,random_state=random_state)
    y_train_inv = y_inverse_transform(y_train)
    y_val_inv = y_inverse_transform(y_val)
    y_test_inv = y_inverse_transform(y_test)

    # merge dataframe again
    input_data_train = X_train.assign(kinetics=y_train)
    input_data_val = X_val.assign(kinetics=y_val)
    input_data_test = X_test.assign(kinetics=y_test)

    # ---------------- fit model ----------------
    ql = feyn.QLattice()
    models = ql.auto_run(input_data_train, output_name='kinetics', max_complexity=20,
     kind="regression", loss_function="squared_error", criterion="bic", n_epochs=20)

    errors = {"train": [], "val": [], "test": [], "ood": []}
    parameters_all = []
    for model_index, model in enumerate(models):
        Path(f"./models/R{random_state}/model{model_index}").mkdir(parents=True, exist_ok=True)
        curr_directory = f"./models/R{random_state}/model{model_index}"
        model.save(f"{curr_directory}/model{model_index}.json")
        # extract model parameters
        parameters_all.append(extract_parameters(model, X_drop.columns))

        # As an .svg file
        with open(f'{curr_directory}/model{model_index}.svg', 'w') as fd:
            fd.write(model._repr_svg_())

        # ---------------- plot overview ----------------
        model.plot(input_data_train, input_data_val, filename=f"{curr_directory}/model{model_index}_plot.html")
        model.plot_regression(input_data_val, filename=f"{curr_directory}/model{model_index}_regression.png")
        plt.close()

        # ------------- Predict and calculate errors --------------
        train_predict = model.predict(input_data_train)
        val_predict = model.predict(input_data_val)
        test_predict = model.predict(input_data_test)
        ood_predict = model.predict(input_data_ood)

        train_error = mean_squared_error(y_train, train_predict)
        val_error = mean_squared_error(y_val, val_predict)
        test_error = mean_squared_error(y_test, test_predict)
        ood_error = mean_squared_error(y_ood, ood_predict)

        errors["train"].append(train_error)
        errors["val"].append(val_error)
        errors["test"].append(test_error)
        errors["ood"].append(ood_error)

        # apply inverse transform
        train_predict = y_inverse_transform(train_predict)
        val_predict = y_inverse_transform(val_predict)
        test_predict = y_inverse_transform(test_predict)
        ood_predict = y_inverse_transform(ood_predict)

        # ------------- Plotting training data figure --------------
        plot_params = {"use_params": True, "xlabel": "interfering strands",
        "yscale": "log", "ylabel": r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)"}
        fig = figure_factory((8,6),**plot_params)

        # sort the data
        sorted_y_train,sorted_prediction_train = sort_kinetics(y_train_inv, train_predict)
        sorted_y_val,sorted_prediction_val = sort_kinetics(y_val_inv, val_predict)
        sorted_y_test,sorted_prediction_test = sort_kinetics(y_test_inv, test_predict)
                            
        # plot train data
        x_vals_train = np.arange(0,1,1.0/len(sorted_y_train))
        plt.plot(x_vals_train, sorted_y_train, color='b', label=f'training, mse={train_error:.2f}', lw=PlotStyles.linewidth)
        plt.scatter(x_vals_train, sorted_prediction_train, color='b', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)

        # plot val data
        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
        plt.plot(x_vals_val, sorted_y_val, color='r', label=f'validation, mse={val_error:.2f}', lw=PlotStyles.linewidth)
        plt.scatter(x_vals_val, sorted_prediction_val, color='r', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)

        # plot test data
        x_vals_test = np.arange(0,1,1.0/len(sorted_y_test))
        plt.plot(x_vals_test, sorted_y_test, color='tab:olive', label=f'test, mse={test_error:.2f}', lw=PlotStyles.linewidth)
        plt.scatter(x_vals_test, sorted_prediction_test, color='tab:olive', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)

        plt.legend(loc='best', fontsize=9)
        plt.ylim(6E-9, 2E-3)
        plt.tight_layout()
        plt.savefig(f"{curr_directory}/model{model_index}_plot_train.svg", format='svg')
        plt.close()

        # ------------- Plotting ood data figure --------------
        plot_params = {"use_params": True, "xlabel": "interfering strands",
        "yscale": "log", "ylabel": r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)"}
        fig = figure_factory((8,6),**plot_params)

        # sort the data
        sorted_y_ood,sorted_prediction_ood = sort_kinetics(y_ood_inv, ood_predict)
                            
        # plot ood data
        x_vals_ood = np.arange(0,1,1.0/len(sorted_y_ood))
        plt.plot(x_vals_ood, sorted_y_ood, color='k', label=f'out-of-distribution data, mse={ood_error:.2f}', lw=PlotStyles.linewidth)
        plt.scatter(x_vals_ood, sorted_prediction_ood, color='k', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)

        plt.legend(loc='best', fontsize=9)
        plt.ylim(1E-7, 2E-3)
        plt.tight_layout()
        plt.savefig(f"{curr_directory}/model{model_index}_plot_ood.svg", format='svg')
        plt.close()
    
    # save errors to file
    with open(f"./models/R{random_state}/errors.txt", 'w') as f:
        f.write("model_index,train_error,val_error,test_error,ood_error\n")
        for model_index,_ in enumerate(models):
            f.write(f"{model_index},{errors['train'][model_index]},"
            f"{errors['val'][model_index]},{errors['test'][model_index]},{errors['ood'][model_index]}\n")

    # write parameters to file
    with open(f"./models/R{random_state}/features.txt", 'w') as f:
        f.write("model_index,parameters\n")
        for model_index,_ in enumerate(models):
            f.write(f"{model_index},")
            [f.write(f"{parameter},") for parameter in parameters_all[model_index]]
            f.write("\n")