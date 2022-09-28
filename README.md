# randompooltmsd
Code for creating decision trees, neural nets, and Feyn models based on features of interfering strands

## general info
The folder input_data contains excel files for the different types of training data. Within each excel files, each individual choice of features is given its own sheet. Within scripts, the choice of input features can be changed by altering the **sheet_name** parameter in the pandas import.
Plotting styles are partially set in each individual script file and partially in regressiontree.plot_utilities.
The **plots** folder generally contains all generated plots.
The **experiments** folder is where the results of individual runs are saved.

## linear analysis
This package contains all linear evaluations of the features (e.g., correlations).

### linear_analysis.py
This script calculates correlations between features, correlations between each invididual feature and the kinetic constant, t-SNE and PCA analyses of features, linear fits between the input features and the kinetic constant, and sigmoid/tanh fits between the input features and the kinetic constants.
<ul>
<li>"choose what to calculate" sets which of these is calculated in a single run</li>
<li>The number of pca components to plot is set in "look at the first four components"</li>
<li>The features that should be used for the fits is set in "pick components for the fit"</li>
<li>Different choices for tanh/sigmoid fits are set in "pick tanh/sigmoid feature choices to plot"</li>
</ul>

## regression tree
This package generates and evaluates regression trees for the data.

### regression_tree_screening.py
This scripts generates the DecisionTreeRegressor models for a given set of features. 
<ul>
<li>**experiment_name** sets the feature subsets that should be used. Multiple feature choices only make sense for the **sequential_drop** run option. Otherwise, the setting should generally be "All", "EnsembleFull" or "Ensemble".</li>
<li>**alpha_start_index** determines the complexity up to which the trees should be generated. A value of 30 implies that the 30 least complex trees will be generated.</li>
<li>**run_names** sets the names of each set of trees. **random_seeds** sets the random seeds used for each of these runs.</li>
</ul>

The script can be run in different modes depending on how sets of features should be chosen.
<ul>
<li>If all modes are set to False, the algorithm uses all features in the **experiment_name** sheet. This setting is used for all individual feature choices in the publication.</li>
<li>**sequential_drop** drops feature columns starting from the right (in the excel file) for up to **no_of_drops**-1 different columns.</li>
<li>**random choice** draws **no_of_drops** feature subsets of size **random_size**. When no new feature subset is found after 1000 tries, the current run terminates. This mode is not used in the publication.</li>
<li>**exhaustive_draw** finds all subsets of size **draw_size**. By using as size that is one smaller than the number of features, dropping of each individual column can be tested.</li>
</ul>

Depending on the settings, the script outputs a number of different files for each run:
<ul>
<li>**{experiment_name}_errors_importance.xlsx** contains the different errors and importance scores for the different complexity trees generated during the run.</li>
<li>**{experiment_name}_best_errors.xlsx** contains the errors and index of the best tree (as evaluated by the validation error) for that run.</li>
<li>**{experiment_name}_inputs.xlsx** contains a dictionary with the input data and parameters for the relevant data transforms. This is used to aid evaluation of trees in the plot scripts.</li>
<li>The folder **errors** contains plots of the errors versus the number of leaves.</li>
<li>The folder **trees_dot** contains the graphviz dot files for the trees.</li>
<li>The folder **trees_pickle** contains a tuple with the DecisionTreeRegressor and the random seed</li>
<li>The folder **trees_svg** contains svg files for the trees generated using sklearn.</li>
<li>The folder **fits** contains plots of the fits for each of the trees generated in the run.</li>
</ul>

### regression_tree_utilies.py
This module contains helper functions for transforming data for trees and to evaluate individual trees.
y_transform applies a logarithm to kinetic data and shift it to zero mean and unit variance. The reverse transform restores predicted kinetic data to the original domain. These functions are also used for all other prediction methods.

### plotting_utilities.py
This module contains helper functions for plotting. The style of figures can be adjusted here. Other packages also use this module.

### evaluate_importance_scores.py

