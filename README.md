# randompooltmsd
- Code for creating decision trees, neural nets, and Feyn models based on features of interfering strands as well as the code for NUPACK analysis and feature extraction.
- sequences and measurement data (normalized median of triplets)

## general info
The folder input_data contains excel files for the different types of training data. Within each excel files, each individual choice of features is given its own sheet. Within scripts, the choice of input features can be changed by altering the **sheet_name** parameter in the pandas import.
Plotting styles are partially set in each individual script file and partially in regressiontree.plot_utilities.
The **plots** folder generally contains all generated plots.
The **experiments** or **models** folder is where the results of individual runs are saved.
The **sequences** folder contains the important/known sequences used.
The **measurement** folder contains the normalized data of triplet measurements.
The **NUPACK and feature extraction** folder contains the code for the used NUPACK analysis and feature extraction.

## linear analysis
This package contains all linear evaluations of the features (e.g., correlations).

### linear_analysis.py
This script calculates correlations between features, correlations between each invididual feature and the kinetic constant, t-SNE and PCA analyses of features, linear fits between the input features and the kinetic constant, and sigmoid/tanh fits between the input features and the kinetic constants.
- "choose what to calculate" sets which of these is calculated in a single run
- The number of pca components to plot is set in "look at the first four components"
- The features that should be used for the fits is set in "pick components for the fit"
- Different choices for tanh/sigmoid fits are set in "pick tanh/sigmoid feature choices to plot"


## regression tree
This package generates and evaluates regression trees for the data.

### regression_tree_screening.py
This scripts generates the DecisionTreeRegressor models for a given set of features. 
- **experiment_name** sets the feature subsets that should be used. Multiple feature choices only make sense for the **sequential_drop** run option. Otherwise, the setting should generally be "All", "EnsembleFull" or "Ensemble".
- **alpha_start_index** determines the complexity up to which the trees should be generated. A value of 30 implies that the 30 least complex trees will be generated.
- **run_names** sets the names of each set of trees. **random_seeds** sets the random seeds used for each of these runs.


The script can be run in different modes depending on how sets of features should be chosen.
- If all modes are set to False, the algorithm uses all features in the **experiment_name** sheet. This setting is used for all individual feature choices in the publication.
- **sequential_drop** drops feature columns starting from the right (in the excel file) for up to **no_of_drops**-1 different columns.
- **random choice** draws **no_of_drops** feature subsets of size **random_size**. When no new feature subset is found after 1000 tries, the current run terminates. This mode is not used in the publication.
- **exhaustive_draw** finds all subsets of size **draw_size**. By using as size that is one smaller than the number of features, dropping of each individual column can be tested.


Depending on the settings, the script outputs a number of different files for each run:
- **{experiment_name}_errors_importance.xlsx** contains the different errors and importance scores for the different complexity trees generated during the run.
- **{experiment_name}_best_errors.xlsx** contains the errors and index of the best tree (as evaluated by the validation error) for that run.
- **{experiment_name}_inputs.xlsx** contains a dictionary with the input data and parameters for the relevant data transforms. This is used to aid evaluation of trees in the plot scripts.
- The folder **errors** contains plots of the errors versus the number of leaves.
- The folder **trees_dot** contains the graphviz dot files for the trees. These can be converted to an svg file via "dot -Tsvg tree.dot -o tree.svg". Graphviz needs to be installed separately.
- The folder **trees_pickle** contains a tuple with the DecisionTreeRegressor and the random seed.
- The folder **trees_svg** contains svg files for the trees generated using sklearn.
- The folder **fits** contains plots of the fits for each of the trees generated in the run.


### regression_tree_utilies.py
This module contains helper functions for transforming data for trees and to evaluate individual trees.
y_transform applies a logarithm to kinetic data and shift it to zero mean and unit variance. The reverse transform restores predicted kinetic data to the original domain. These functions are also used for all other prediction methods.

### plotting_utilities.py
This module contains helper functions for plotting. The style of figures can be adjusted here. Other packages also use this module.

### evaluate_importance_scores.py
This script extracts the importance scores from errors_importance.xlsx of runs that were performed for all possible values of alpha. It iterates over all runs and always uses the importance scores for the most complex tree.

### evaluate_drop_tree_errors.py
This scripts plots the validation, test, and out-of-distribution errors for trees in which one ensemble feature is left out.

### evalute_mixed_trees_individual.py
This script plots the actual fits for train/val/test data or ood data for a specific random seed.
- **model_names** sets which models are plotted. The number of subplots in "set plot parameters" should fit. "set x and y axis location" also needs to be adjusted individually.
- **RX** determines which random seed is used for plotting.
- **plot_train_val_test** determines whether train/val/test or ood data is plotted.

### evaluate_mixed_tree_errors.py
This script plots the errors of trees generated for a mixture of two circuits averaged over all runs. **model_names** determines the used feature subsets.

### evaluate_single_tee_errors.py
This script plots the errors of trees generated for a single circuit and and evaluated on the other averaged over all runs. **model_names** determines the used feature subsets.

## feyn
This package generates and evaluates Feyn models for the data.

### create_feyn_models.py
This script generates Feyn models for a given choice of input features and evaluates the model performance. The models are generated by running the auto_run function with a Bayesian information criterion for 20 epochs with a mean squared error loss function. Ten different models are generated per random seed, sorted by decreasing performance.Results are initially copied directly to the **models** folder and should be sorted into appropriate subfolders by hand.
- **sheet_name** determines the feature subset.
- **errors.txt** contains the train, val, test, and ood errors for the different models.
- **features.txt** contains a list of the features used in each model.
- **modelx.json** contains the actual model.
- **modelx.svg** and **modelx_plot.html** contain representations of the models.
-**modelx_plot_ood/train.svg** contains the fits generated by the models.

### evaluate_errors.py
Plots the errors of the different feature subsets average over all random seeds. For each random seed, the best model is determined by the minimum of the sum of the training and validation errors.
- **model_names** determines the feature subsets.
- **random_states** sets the random seeds.

### plot_models.py
This script outputs a mathematical formula representing a specific model and plots a 1D response (i.e., other parameters are fixed).
- **sheet_name** determines the feature subset.
- **root_directory** determines the specific run that is used (information criterion, random seed).
- **models** determines which model of the specific run is used.
- **plot_1d_response** plots the 1D response. The parameter **oneD_feature** changes the x-Axis. 

The outputs are:
- **modelX_signal.html** is a file that estimates the contribution of each feature to the overall signal.
- A latex representation of the model printed to the console.
- **{sheet_name}_modelX.svg** is a plot of the 1D response.

## nn model
This package generates and evaluates fully connected neural network models for the data.

### MLP_model.py
Contains the actual model definition. The loss function is mean squared error, the activation function is ReLU.

### kinetics_dataset.py
Loads the dataset. Can either be called using the "ood" mode, in which case a mean and std value needs to be supplied for the fixed transform, or using the "test", "val", or "test" modes, in which case a train/val/test split is introduced to the data. The dataset samples contain "features" and "kinetics" values.

### training.py
This script runs the actual training.
- **experiment_names** defines the feature subsets.
- Normalization can be beformed using different weight decay values or dropout values (chosen by the **weight_decay_FLAG**).
- **run_names** and **random_seeds** should be chosen to be compatible.

For each random seed, the following outputs are produced:
- **{run_name}_train/ood.svg** shows the actual fits.
- **{run_name}_errors.txt** contains the errors for each dropout/weight_decay value)
- **{run_name}_sorted_pairs** is a pickle file containing pairs of original y values and predictions with indices 0:test, 1:val, 2:test, 3:ood.
- **errors_{experiment_name}** in the folder **errors** contains the best errors for a given set of weight decay or dropout values.

### evaluate_single_model.py
Contains a helper function that creates a fit figure for a given model (provided either as an object or as a string path to the checkpoint) and a given set of dataloaders.

### evaluate_errors.py
Plots a bar graph of the errors in the **errors** folder for different choices of feature subsets.
- **model_names** sets the used feature subsets.
- **no_of_runs** sets the number of random seeds.

### evaluate_errors_normalization.py
Plots errors for different weight decay or dropout values averaged over different random seeds.
- **no_of_weights** sets the number of different weight decay/dropout values
- **no_of_runs** sets the number of random seeds
- **weight_decay_FLAG** determines whether weight decay or dropout is used for plotting

### compare_model_fits.py
Plots the fit for a specific random seed for a set of feature subsets.
- **model_names** sets which models are plotted. The number of subplots in "set plot parameters" should fit. "set x and y axis location" also needs to be adjusted individually.
- **RX** determines which random seed is used for plotting.
- **plot_train_val_test** determines whether train/val/test or ood data is plotted.

## comparison
This script plots a graph that compares the validation, test, and ood error for the different model types (regression tree, feyn, neural network)
- **evaluate_error_comp.py** plots a bar graph of the errors.
- **error_comp.xlsx** contains the error data.
- **error_overview.svg** is the output plot.
