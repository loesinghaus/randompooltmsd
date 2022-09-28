import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from pathlib import Path
import itertools

# import from local utilities
from plotting_utilities import figure_factory, PlotStyles, sort_kinetics
from regression_tree_utilities import *


# ---------------- Set number of different experiments ----------------
# the name of the experiment should match a sheet name in the input file
experiment_names = ["EnsembleFull"]#, "Ensemble", "Choice1", "Choice2",
 #"Choice3", "Choice4", "Choice5", "Choice6", "Choice7", "Choice8", "Choice9", "Choice10", "Choice11"]
# input file
input_excel_file_path = "../input_data/strand_features_both_circuits.xlsx"

# ---------------- Set parameters ----------------
# only the x least complex trees are calculated
# generally, anything over 30 is pointless due to overfitting
alpha_start_index = 30

# train test split ratio
train_validation_test = True
split_ratio = 0.4

# which information should be extracted?
# these two are costly:
plot_tree_svg = False
plot_fit = False
# these are not:
plot_errors = True
pickle_tree = True

# ------------ run mode -------------
# Here, one can run feature selection in different modes
# The standard mode is leaving all flags at False (all features are used)
# sequential drop: Drop features one by one from the right
no_of_drops = 1
sequential_drop = False
# random choice: Choose n random features (without duplicates) for no_of_drops experiments
random_choice = False 
random_size = 5
# exhaustive draw: Explore all possible choices of draw_size
exhaustive_draw = True
draw_size = 12

for experiment_name in experiment_names:
    print(f"Running experiment {experiment_name}.")

    # set the number of runs per experiment, their name, and the random seeds for the train/validation split
    run_names = [f"{experiment_name}DropOneMixedR{i*11}" for i in range(1,11)]
    random_seeds = [11*i for i in range(1,11)]
    
    for run_index, run_name in enumerate(run_names):
        try:
            random_seed = random_seeds[run_index]
            sheet_name = experiment_name

            if random_choice:
                all_drop_numbers = set()

            # ------------ create directories for results -------------
            Path(f"./experiments/{experiment_name}/{run_name}").mkdir(parents=True, exist_ok=True)
            Path(f"./experiments/{experiment_name}/{run_name}/trees_dot").mkdir(parents=True, exist_ok=True)
            if plot_tree_svg:
                Path(f"./experiments/{experiment_name}/{run_name}/trees_svg").mkdir(parents=True, exist_ok=True)
            if plot_fit:
                Path(f"./experiments/{experiment_name}/{run_name}/fits").mkdir(parents=True, exist_ok=True)
            if pickle_tree:
                Path(f"./experiments/{experiment_name}/{run_name}/trees_pickle").mkdir(parents=True, exist_ok=True)
            if plot_errors:
                Path(f"./experiments/{experiment_name}/{run_name}/errors").mkdir(parents=True, exist_ok=True) 

            # ---------------- load input data ----------------
            df = pd.read_excel(input_excel_file_path, sheet_name=sheet_name)
            # Define X and y
            X = df.drop("kinetics", axis=1)
            # scale kinetics for easier fitting
            y = df["kinetics"]
            y, y_inverse_transform, transform_mean, transform_std = y_transform(y)

            # ---------------- find all combinations for an exhaustive draw ----------------
            if exhaustive_draw:
                no_of_features = len(X.columns)
                column_combinations = list(itertools.combinations(range(no_of_features),draw_size))
                no_of_drops = len(column_combinations)
            
            # create error file
            with open(f"./experiments/{experiment_name}/{run_name}/{run_name}_best_errors.txt", 'a') as f:
                if random_choice or exhaustive_draw:
                    f.write(f"{X.columns}\n")
                    f.write("index_choices,alpha_index,alpha,val_error\n")
                elif sequential_drop:
                    f.write("drop_index,alpha_index,alpha,val_error\n")
                else:
                    if train_test_split:
                        f.write("alpha_index,alpha,val_error, test_error\n")
                    else:
                        f.write("alpha_index,alpha,val_error\n")

            # ---------------- loop over different combinations of feature choices ----------------
            for drop_index in range(no_of_drops):
                if sequential_drop or random_choice or exhaustive_draw:
                    print(f"drop_index: {drop_index}")

                # drop columns from X if appropriate
                if random_choice:
                    # if random choice mode is active, try to find a a new combination in less than 1000 tries
                    # if no new combination is found, stop the run
                    counter = 0
                    while(True):
                        # choose random integers (discard duplicates of previously drawn combinations)
                        rng = np.random.default_rng()
                        drop_numbers = rng.choice(len(X.columns), size=(len(X.columns)-random_size), replace=False)
                        drop_numbers.sort()
                        if not tuple(list(drop_numbers)) in all_drop_numbers:
                            all_drop_numbers.add(tuple(list(drop_numbers)))
                            break
                        else:
                            counter += 1
                        if counter > 1000:
                            raise StopIteration
                    X_drop = X.drop(X.columns[drop_numbers], axis=1)
                    chosen_indices = [i for i in range(len(X.columns)) if not i in list(drop_numbers)]
                    file_prefix = f"{run_name}_{chosen_indices}"
                    # set name of excel sheet for writing errors
                    excel_sheet_name = str(chosen_indices)[1:-1]
                elif exhaustive_draw:
                    column_combination = column_combinations[drop_index]
                    drop_columns = [i for i in range(len(X.columns)) if not i in list(column_combination)]
                    X_drop = X.drop(X.columns[drop_columns], axis=1)
                    file_prefix = f"{run_name}_{drop_columns}"
                    # set name of excel sheet for writing errors
                    # excel_sheet_name = str(column_combination)[1:-1]
                    # use dropped column instead:
                    excel_sheet_name = str(drop_columns)[1:-1]
                elif sequential_drop:
                    # drop the last x features
                    X_drop = X.drop([X.columns[-curr_index] for curr_index in range(1, drop_index+1)], axis=1)
                    file_prefix = f"{run_name}_drop{drop_index}"
                    # set name of excel sheet for writing errors
                    excel_sheet_name = f"drop{drop_index}"
                else:
                    X_drop = X
                    file_prefix = f"{run_name}"
                    # set name of excel sheet for writing errors
                    excel_sheet_name = run_name
                
                # Apply train/val split
                X_train, X_val, y_train, y_val = train_test_split(X_drop,y,test_size=split_ratio,random_state=random_seed)
                y_train_inv = y_inverse_transform(y_train)
                
                if train_validation_test:
                    X_val, X_test, y_val, y_test = train_test_split(X_val,y_val,test_size=0.5,random_state=random_seed)
                    y_val_inv = y_inverse_transform(y_val)
                    y_test_inv = y_inverse_transform(y_test)
                else:
                    y_val_inv = y_inverse_transform(y_val)

                # pickle the results
                pickle_inputs = {"X_train": X_train, "X_val": X_val, "y_train": y_train, "y_val": y_val,
                "y_inverse_transform": y_inverse_transform, "transform_mean": transform_mean,
                "transform_std": transform_std}
                if train_validation_test:
                    pickle_inputs["X_test"] = X_test
                    pickle_inputs["y_test"] = y_test
                if exhaustive_draw or random_choice or sequential_drop:
                    pickle.dump(pickle_inputs, open(f"./experiments/{experiment_name}/{run_name}/{run_name}_inputs_drop{drop_columns}.p","wb"))
                else:
                    pickle.dump(pickle_inputs, open(f"./experiments/{experiment_name}/{run_name}/{run_name}_inputs.p","wb"))

                # train the models
                train_errors = []
                val_errors = []
                if train_test_split:
                    test_errors = []

                # ---------------- find possible values of alpha ----------------
                model = DecisionTreeRegressor(criterion="squared_error", random_state=random_seed)
                path = model.cost_complexity_pruning_path(X_train, y_train)
                ccp_alphas, impurities = path.ccp_alphas, path.impurities
                # only use the last x values for actual trees
                ccp_alphas = ccp_alphas[-alpha_start_index:]
                # initialize numpy arrays for storing tree properties
                column_occurences = np.zeros((len(ccp_alphas),len(X_drop.columns)))
                leaf_numbers = np.zeros(len(ccp_alphas))
                importance_scores = np.zeros((len(ccp_alphas), len(X_drop.columns)))

                print(f"number of alphas used: {len(ccp_alphas)}")
                for index, ccp_alpha in enumerate(ccp_alphas):
                    # initialize and fit the model
                    model = DecisionTreeRegressor(criterion="squared_error", random_state=random_seed, ccp_alpha=ccp_alpha)
                    model.fit(X_train, y_train)

                    # pickle the resulting tree
                    if pickle_tree:
                        pickle.dump((model,random_seed), open(f"./experiments/{experiment_name}/{run_name}/trees_pickle/{file_prefix}_alpha_{index}.p","wb"))

                    # analyze the resulting tree
                    leaf_numbers[index], column_occurences[index] = analyze_tree(model, features=X_drop.columns)
                    importance_scores[index, :] = model.feature_importances_

                    # make predictions for the val and training set
                    prediction_val = model.predict(X_val)
                    prediction_train = model.predict(X_train)
                    if train_validation_test:
                        prediction_test = model.predict(X_test)
                    
                    # calculate the mean squared errors, add results to lists
                    train_errors.append(mean_squared_error(y_train, prediction_train))
                    val_errors.append(mean_squared_error(y_val, prediction_val))
                    if train_validation_test:
                        test_errors.append(mean_squared_error(y_test, prediction_test))

                    # plot the resulting tree as an svg
                    if plot_tree_svg:
                        fig = figure_factory((8,6))
                        plot_tree(model, feature_names=X_drop.columns)
                        plt.savefig(f"./experiments/{experiment_name}/{run_name}/trees_svg/{file_prefix}_alpha_{index}.svg", format='svg')
                        plt.close()
                    # save graphviz dot file for the tree
                    with open(f"./experiments/{experiment_name}/{run_name}/trees_dot/{file_prefix}_tree_alpha_{index}.dot", 'w') as f:
                        text = visualize_tree(my_tree=model,columns=X_drop.columns,y_inversion_func=y_inverse_transform)
                        f.write(text)
                    
                    # apply inverse function to the predicted kinetic values for visualization
                    prediction_train = y_inverse_transform(prediction_train)
                    prediction_val = y_inverse_transform(prediction_val)
                    if train_validation_test:
                        prediction_test = y_inverse_transform(prediction_test)

                    # sort the data
                    sorted_y_train,sorted_prediction_train = sort_kinetics(y_train_inv, prediction_train)
                    sorted_y_val,sorted_prediction_val = sort_kinetics(y_val_inv, prediction_val)
                    if train_validation_test:
                        sorted_y_test,sorted_prediction_test = sort_kinetics(y_test_inv, prediction_test)
                    
                    # plot the data
                    if plot_fit:
                        plot_params = {"use_params": True, "xlabel": "interfering strands",
                        "yscale": "log", "ylabel": r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)"}
                        fig = figure_factory((8,6),**plot_params)
                        
                        # plot train data
                        x_vals_train = np.arange(0,1,1.0/len(sorted_y_train))
                        plt.plot(x_vals_train, sorted_y_train, color='b', label='training data', lw=PlotStyles.linewidth)
                        plt.scatter(x_vals_train, sorted_prediction_train, color='b', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)

                        # plot val data
                        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
                        plt.plot(x_vals_val, sorted_y_val, color='r', label='validation data', lw=PlotStyles.linewidth)
                        plt.scatter(x_vals_val, sorted_prediction_val, color='r', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)
                        
                        # plot test data
                        if train_validation_test:
                            x_vals_test = np.arange(0,1,1.0/len(sorted_y_test))
                            plt.plot(x_vals_test, sorted_y_test, color='tab:olive', label='test data', lw=PlotStyles.linewidth)
                            plt.scatter(x_vals_test, sorted_prediction_test, color='tab:olive', s=PlotStyles.s, alpha=PlotStyles.alpha, linewidths=0)

                        plt.legend(loc='best', fontsize=9)
                        plt.savefig(f"./experiments/{experiment_name}/{run_name}/fits/{file_prefix}_fit_alpha_{index}.svg", format='svg')
                        plt.tight_layout()
                        plt.close()

                # ---------------- plots errors for different complexity penalties ----------------
                if plot_errors:
                    # create plots for errors
                    plot_params = {"use_params": True, "xlabel": "number of leaves", "ylabel": "mean squared error"}
                    x_axis = np.arange(len(ccp_alphas))
                    fig = figure_factory((8,6), **plot_params)

                    plt.plot(leaf_numbers, np.array(train_errors), color='b', label='training set', lw=PlotStyles.linewidth)
                    plt.plot(leaf_numbers, np.array(val_errors), color='r', label='validation set', lw=PlotStyles.linewidth)
                    if train_validation_test:
                        plt.plot(leaf_numbers, np.array(test_errors), color='tab:olive', label='test set', lw=PlotStyles.linewidth)
                    plt.legend(loc='best', fontsize=9)
                    plt.savefig(f"./experiments/{experiment_name}/{run_name}/errors/{file_prefix}_errors.svg", format='svg')
                    plt.tight_layout()
                    plt.close()

                # write all errors and importance scores to an excel file
                best_errors = pd.DataFrame()
                best_errors["no of leaves"] = leaf_numbers
                best_errors["train_error"] = np.array(train_errors)
                best_errors["val_error"] = np.array(val_errors)
                if train_validation_test:
                    best_errors["test_error"] = np.array(test_errors)
                importance_df = pd.DataFrame(data=importance_scores, columns=X_drop.columns)
                # column_count = pd.DataFrame(data=column_occurences, columns=X_drop.columns)
                errors_and_column_count = pd.concat([best_errors, importance_df], axis=1)

                excel_path = f"./experiments/{experiment_name}/{run_name}/{run_name}_errors_importance.xlsx"
                try:
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                        errors_and_column_count.to_excel(writer, sheet_name=excel_sheet_name)
                except FileNotFoundError:
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                        errors_and_column_count.to_excel(writer, sheet_name=excel_sheet_name)
                        
                # write the best errors for one drop index to a .txt file
                with open(f"./experiments/{experiment_name}/{run_name}/{run_name}_best_errors.txt", 'a') as f:
                    best_error_index = np.argmin(val_errors)
                    errors = np.array(val_errors)
                    if random_choice:
                        chosen_indices = [i for i in range(len(X.columns)) if not i in list(drop_numbers)]
                        f.write(f"{chosen_indices},{best_error_index},{ccp_alphas[best_error_index]},{errors[best_error_index]}\n")
                    elif exhaustive_draw:
                        f.write(f"{drop_columns},{best_error_index},{ccp_alphas[best_error_index]},{errors[best_error_index]}\n")
                    elif sequential_drop:
                        f.write(f"{drop_index},{best_error_index},{ccp_alphas[best_error_index]},{errors[best_error_index]}\n")
                    else:
                        if train_validation_test:
                            f.write(f"{best_error_index},{ccp_alphas[best_error_index]},"
                                f"{errors[best_error_index]},{test_errors[best_error_index]}\n")
                        else:
                            f.write(f"{best_error_index},{ccp_alphas[best_error_index]},{errors[best_error_index]}\n")
        except StopIteration:
            # This exists for when no new combination is found in a random choice experiment.
            print("Iteration stopped because no new combination of column choices was found.")
            pass