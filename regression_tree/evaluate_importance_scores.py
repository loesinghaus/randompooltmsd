import pandas as pd
import matplotlib.pyplot as plt
from plotting_utilities import cm2inch, PlotStyles

# set run names
run_names = [f"EnsembleFullMixedR{i*11}" for i in range(1,11)]
replace_ensemble=True

importance_scores_all = []
for run_name in run_names:
    usage_df = pd.read_excel(f"./experiments/TotalAlpha/EnsembleFull/{run_name}/{run_name}_errors_importance.xlsx", sheet_name=run_name)
    
    # rename columns:
    if replace_ensemble:
        replacement_dict = {column: f'{index}: {column.replace(" (ensemble)","")}' for index, column in enumerate(usage_df.columns[4:])}
    else:
        replacement_dict = {column: f'{index}: {column}' for index, column in enumerate(usage_df.columns[5:])}
    usage_df = usage_df.rename(columns=replacement_dict)
    # choose the feature columns
    feature_columns = usage_df.columns[5:]
    # extract importance scores
    importance_scores = pd.Series(0, feature_columns)
    usage_df = usage_df[feature_columns].iloc[0:1]
    for index, row in usage_df.iterrows():
        importance_scores += row/len(run_names)
    importance_scores_all.append(importance_scores)

# write importance scores into dataframe
combined_dataframe = pd.DataFrame(index=feature_columns)
for index, run_name in enumerate([f"run{i+1}" for i in range(len(run_names))]):
    combined_dataframe[run_name] = importance_scores_all[index]

# sort the dataframe by the highest cumulative importance scores
combined_dataframe['sum'] = combined_dataframe.sum(axis=1)
combined_dataframe.sort_values(by='sum',ascending=False,inplace=True)
combined_dataframe = combined_dataframe.drop(['sum'],axis=1)

# plot importance scores
plt.rcParams.update(PlotStyles.font)
ax = combined_dataframe.plot(kind='bar', stacked=True, figsize=cm2inch(6,6))
plt.ylabel("importance score", labelpad=9, fontsize=8)
plt.tick_params(labelsize=8)
plt.legend(fontsize=9)
if replace_ensemble:
    plt.savefig("./plots/importance_scores_ensemble_features.svg", format='svg')
else:
    plt.savefig("./plots/importance_scores_all_features.svg", format='svg')
