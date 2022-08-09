from statistics import mean
import scipy.stats
import scipy.optimize
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.manifold
import sklearn.decomposition
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from functools import partial
# local imports
from plotting_utilities import *
from regression_tree_utilities import y_transform

# sum of scaled tanh
def scaled_tanh(x, *params):
    no_of_features = len(x[0,:])
    a = params[0:no_of_features]
    b = params[no_of_features:2*no_of_features]
    S = params[2*no_of_features:3*no_of_features]
    C = params[-1]
    #return np.sum(S/(1+np.exp(-(a*x+b))),axis=1)
    return np.sum(S*np.tanh(a*x+b),axis=1)

# read dataframe
df = pd.read_excel("../input_data/strand_features_both_circuits.xlsx", sheet_name="EnsembleFull")
# Define X and y (manual dropping of questions here)
X1 = df.drop("kinetics", axis=1)
y1 = df["kinetics"]

# normalize input features for PCA
X1_numpy = X1.to_numpy()
X1_mean = np.mean(X1_numpy, axis=0, keepdims=True)
X1_std = np.std(X1_numpy, axis=0, keepdims=True)
X1_numpy = (X1_numpy-X1_mean)/X1_std
# take the logarithm of y1 for easier viewing
y1 = np.log10(y1)

# choose what to calculate
corr_feature_feature = False
corr_feature_kin = False
PCA_FLAG = False
linear_fits = False
sigmoid_fits = True

# ---------- correlation between features ----------
if corr_feature_feature:
    correlations = np.zeros((len(X1.columns), len(X1.columns)))
    features_text = ""
    for index1, column1 in enumerate(X1.columns):
        features_text += f"{index1}: {column1}\n"
        for index2, column2 in enumerate(X1.columns):
            correlations[index1, index2],_ = scipy.stats.pearsonr(X1[column1],X1[column2])
            correlations[index1, index2] = correlations[index1, index2]**2

    sns.set(rc = {'figure.figsize': cm2inch(10,8), **PlotStyles.font})
    no_of_features = len(X1.columns)
    tick_labels = list(range(no_of_features))[::2]
    ax = sns.heatmap(correlations, vmin=0, vmax=1, linewidths=.5,\
         xticklabels = tick_labels, yticklabels = tick_labels, cbar_kws={'label': r"Pearson $r^2$"})
    ax.text(0.05, 0.95, features_text, transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.tick_params(labelsize=8)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)

    plt.xlabel("feature", labelpad=9, fontsize=9)
    plt.ylabel("feature", labelpad=9, fontsize=9)

    plt.savefig('./plots/question_correlations.svg', format='svg')
    plt.close()

# ---------- correlation between features and kinetics ----------
if corr_feature_kin:
    correlations = pd.DataFrame(columns=["feature", "pearson_corr", "spearman_corr", "kendall_tau"])
    for column in X1:
        pearson_corr,_ = scipy.stats.pearsonr(X1[column],y1)
        spear_corr,_ = scipy.stats.spearmanr(X1[column],y1)
        kendall_tau,_ = scipy.stats.kendalltau(X1[column],y1, variant='c')
        row = {"feature": column, "pearson_corr": pearson_corr, "spearman_corr": spear_corr, "kendall_tau": kendall_tau}
        correlations = correlations.append(row, ignore_index=True)
    pearson_corr_sq = np.square(correlations["pearson_corr"])
    spearman_corr_sq = np.square(correlations["spearman_corr"])
    kendall_tau_sq = np.square(correlations["kendall_tau"])
    correlations["sq(pearson_corr)"] = pearson_corr_sq
    correlations["sq(spearman_corr)"] = spearman_corr_sq
    correlations["sq(kendall_tau)"] = kendall_tau_sq

    # sort correlation values?
    #correlations.sort_values(by="sq(spearman_corr)", axis=0, ascending=False, inplace=True)

    # set plot parameters
    # reset params after using sns
    mpl.rcParams.update(mpl.rcParamsDefault)

    # plot spearman correlation
    fig = figure_factory((8,6), use_params=True, ylabel=r"Spearman $\rho^2$", xlabel="feature")
    plt.bar(x=list(range(len(X1.columns))), height=correlations["sq(spearman_corr)"])
    plt.savefig("./plots/correlations_qu_kin_spearman.svg")
    plt.close()
    # plot pearson correlation
    fig = figure_factory((8,6), use_params=True, ylabel=r"Pearson $r^2$", xlabel="feature")
    plt.bar(x=list(range(len(X1.columns))), height=correlations["sq(pearson_corr)"])
    plt.savefig("./plots/correlations_qu_kin_pearson.svg")
    plt.close()

    correlations.to_excel("./plots/correlations_features_questions.xlsx")

# ---------- t-SNE and PCA ----------
if PCA_FLAG:
    # ---------- t-SNE ----------
    tsne = sklearn.manifold.TSNE(n_components=2)
    z = tsne.fit_transform(X1_numpy)

    df = pd.DataFrame()
    df["y"] = y1
    df["comp1"] = z[:,0]
    df["comp2"] = z[:,1]

    ax = sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
                    palette=sns.color_palette("rocket", as_cmap=True),
                    data=df)

    ax.tick_params(labelsize=8)
    cbar = ax.collections[0].colorbar

    plt.xlabel("component 1", labelpad=9, fontsize=9)
    plt.ylabel("component 2", labelpad=9, fontsize=9)

    plt.savefig('./plots/tsne.svg', format='svg')
    plt.close()
    # reset params after using sns
    mpl.rcParams.update(mpl.rcParamsDefault)

    # ---------- PCA -----------
    pca = sklearn.decomposition.PCA(n_components=None)
    z = pca.fit_transform(X1_numpy)
    
    df = pd.DataFrame()
    df["y"] = y1
    df["comp1"] = z[:,0]
    df["comp2"] = z[:,1]

    # plot PCA
    sns.set(rc = {'figure.figsize': cm2inch(8,6), **PlotStyles.font})
    ax = sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
                    palette=sns.color_palette("rocket", as_cmap=True),
                    data=df)

    # remove legend, add colormap
    ax.get_legend().remove()
    norm = plt.Normalize(np.min(y1), np.max(y1))
    sm = plt.cm.ScalarMappable(cmap="rocket", norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm)
    cbar.ax.set_ylabel(r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)")

    # plot params
    ax.tick_params(labelsize=8)
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("component 1", fontsize=9, labelpad=9)
    plt.ylabel("component 2", fontsize=9, labelpad=9)
    plt.savefig('./plots/pca.svg', format='svg')
    plt.close()

    # reset params after using sns
    mpl.rcParams.update(mpl.rcParamsDefault)

    # ---------- plot contribution of components ------------
    # look at the first four components
    components = pca.components_[0:4]
    explained_variance = np.around(pca.explained_variance_ratio_, decimals=2)

    # calculate spearman rho
    pca_spearman_corrs = []
    for index, variance in enumerate(explained_variance[0:4]):
        spear_corr,_ = scipy.stats.spearmanr(z[:,index],y1)
        pca_spearman_corrs.append(f"{variance}\n{np.around(spear_corr**2,3)}")

    no_of_features = len(X1.columns)
    sns.set(rc = {'figure.figsize': cm2inch(8,5), **PlotStyles.font})
    tick_labels = list(range(no_of_features))
    ax = sns.heatmap(components, vmin=0, vmax=np.max(components), linewidths=.5,\
        xticklabels=tick_labels, yticklabels=pca_spearman_corrs, cbar_kws={'label': r"PCA contribution"})

    features_text = ""
    for index1, column1 in enumerate(X1.columns):
        column1 = column1.replace(" (ensemble)", "")
        features_text += f"{index1}: {column1}\n"
    ax.text(0.05, 0.95, features_text, transform=ax.transAxes, fontsize=9, verticalalignment='top')
    ax.tick_params(labelsize=8)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=8)
    plt.xlabel("features", fontsize=9, labelpad=9)
    plt.ylabel("explained variance\n"r"Spearman $\rho^2$", fontsize=9, labelpad=9)
    plt.savefig('./plots/pca_component_contributions.svg')
    plt.close()

    # reset params after using sns
    mpl.rcParams.update(mpl.rcParamsDefault)

# ---------- linear regression ------------
if linear_fits:    
    # pick components for the fit
    X1_numpy = X1_numpy[:,[0,2,4]]
    y_fitting, y_inverse_transform, _, _ = y_transform(np.power(10, y1))
    X_train, X_val, y_train, y_val = train_test_split(X1_numpy,y_fitting,test_size=0.4,random_state=11)
    reg = LinearRegression().fit(X_train, y_train)
    #print(reg.coef_)

    # predict kinetics
    prediction_train = reg.predict(X_train)
    prediction_val = reg.predict(X_val)

    # calculate score
    score_train = reg.score(X_train,y_train)
    score_val = reg.score(X_val,y_val)

    # calculate MSE
    mse_train = mean_squared_error(y_train, prediction_train)
    mse_val = mean_squared_error(y_val, prediction_val)

    # transform and sort training data
    y_train_orig = y_inverse_transform(y_train)
    sorted_y_train, sorted_prediction_train = sort_kinetics(y_train_orig, y_inverse_transform(prediction_train))
    # transform and sort validation data
    y_val_orig = y_inverse_transform(y_val)
    sorted_y_val, sorted_prediction_val = sort_kinetics(y_val_orig, y_inverse_transform(prediction_val))
    
    # plot linear fit
    plot_params = {"use_params": True, "yscale": "log", "xlabel": "design",\
         "ylabel": r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)"}
    fig = figure_factory((8,6), **plot_params)

    # plot training data
    x_vals_train = np.arange(0,1,1.0/len(sorted_y_train))
    plt.plot(x_vals_train, sorted_y_train, color='b', label=f"training, $R^2$={score_train:.2f}, mse={mse_train:.2f}", linewidth=1.5)
    plt.scatter(x_vals_train, sorted_prediction_train, color='b', s=10, alpha=0.5)
    # plot val data
    x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
    plt.plot(x_vals_val, sorted_y_val, color='r', label=f"validation, $R^2$={score_val:.2f}, mse={mse_val:.2f}", linewidth=1.5)
    plt.scatter(x_vals_val, sorted_prediction_val, color='r', s=10, alpha=0.5)
    
    # additional parameters
    plt.ylim(6E-9, 2E-3)
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"./plots/linear_regression.svg", format='svg')
    plt.close()

if sigmoid_fits:
    #[[i for i in range(len(X1_numpy[0,:]))], [0,2,3,4,5], [0,2,3,4,11], [0,2,3,4,8], [0,2,3,4,5,8],
    Choices = [[0,2,4],[0,2],[2,4],[0,4],[1,2],[0],[2]]
    for choice_index, choice in enumerate(Choices):
        # ---------- fit scaled sigmoid ------------
        # pick components for the fit
        X1_numpy_fit = X1_numpy[:,choice]
        y_fitting, y_inverse_transform, _, _ = y_transform(np.power(10, y1))
        X_train, X_val, y_train, y_val = train_test_split(X1_numpy_fit,y_fitting,test_size=0.4,random_state=11)

        # WORKS QUITE OKAY; DO THIS IN PYTORCH INSTEAD
        no_features = len(X1_numpy_fit[0,:])
        p0 = np.concatenate((np.ones(no_features),np.zeros(no_features),np.ones(no_features)))
        random_vals = (np.random.rand(*p0.shape)-0.5)/50.0
        p0 += random_vals
        popt, pcov = scipy.optimize.curve_fit(scaled_tanh, X_train, y_train, p0=p0, maxfev=20000)
        
        # predict kinetics
        prediction_train = scaled_tanh(X_train, *popt)
        prediction_val =  scaled_tanh(X_val, *popt)

        # calculate MSE
        mse_train = mean_squared_error(y_train, prediction_train)
        mse_val = mean_squared_error(y_val, prediction_val)

        # transform and sort training data
        y_train_orig = y_inverse_transform(y_train)
        sorted_y_train, sorted_prediction_train = sort_kinetics(y_train_orig, y_inverse_transform(prediction_train))
        # transform and sort validation data
        y_val_orig = y_inverse_transform(y_val)
        sorted_y_val, sorted_prediction_val = sort_kinetics(y_val_orig, y_inverse_transform(prediction_val))

        # plot linear fit
        plot_params = {"use_params": True, "yscale": "log", "xlabel": "design",\
            "ylabel": r"$\mathrm{k}_{\mathrm{eff}}$ (1/nM$\cdot$s)"}
        fig = figure_factory((8,6), **plot_params)

        # plot training data
        x_vals_train = np.arange(0,1,1.0/len(sorted_y_train))
        plt.plot(x_vals_train, sorted_y_train, color='b', label=f"training, mse={mse_train:.2f}", linewidth=1.5)
        plt.scatter(x_vals_train, sorted_prediction_train, color='b', s=10, alpha=0.5)
        # plot val data
        x_vals_val = np.arange(0,1,1.0/len(sorted_y_val))
        plt.plot(x_vals_val, sorted_y_val, color='r', label=f"validation, mse={mse_val:.2f}", linewidth=1.5)
        plt.scatter(x_vals_val, sorted_prediction_val, color='r', s=10, alpha=0.5)
        
        # additional parameters
        plt.ylim(6E-9, 2E-3)
        plt.legend(loc='best', fontsize=8)
        plt.savefig(f"./plots/tanh_regression_model{chr(65+choice_index)}.svg", format='svg')
        plt.close()