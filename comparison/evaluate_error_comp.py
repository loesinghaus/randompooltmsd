import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# add parent to path
import path
import sys
 # directory reach
directory = path.Path(__file__).abspath() 
# setting path
sys.path.append(directory.parent.parent)
# local imports
from regressiontree.plotting_utilities import *

error_data = pd.read_excel("error_comp.xlsx", sheet_name="AllErrors")

fig = figure_factory((12,6))

# convert the model index to a letter
model_indices = ["t f n\nval", "t f n\ntest\nmodel B", "t f n\nood",
"t f n\nval", "t f n\ntest\nmodel G", "t f n\nood",
"t f n\nval", "t f n\ntest\nmodel H", "t f n\nood",]
xvals = np.arange(0, 1.5, 0.5)
tick_locations = [0,2,4,7,9,11,14,16,18]
tick_locations = [tick+0.5 for tick in tick_locations]
plt.xticks(tick_locations, model_indices)

error_bar_fmt = {"fmt":'none', "color":'k', "linewidth":1, "markeredgewidth":1, "capsize":1.5}

plt.bar(xvals, error_data["val"][0::3], color='r', width=0.5, edgecolor = "k")
plt.errorbar(xvals, error_data["val"][0::3], yerr=error_data["std val"][0::3],**error_bar_fmt)

plt.bar(xvals+2, error_data["test"][0::3], color='tab:olive', width=0.5, edgecolor = "k")
plt.errorbar(xvals+2, error_data["test"][0::3], yerr=error_data["std test"][0::3],**error_bar_fmt)

plt.bar(xvals+4, error_data["ood"][0::3], color='grey', width=0.5, edgecolor = "k")
plt.errorbar(xvals+4, error_data["ood"][0::3], yerr=error_data["std ood"][0::3],**error_bar_fmt)

plt.bar(xvals+7, error_data["val"][1::3], color='r', width=0.5, edgecolor = "k")
plt.errorbar(xvals+7, error_data["val"][1::3], yerr=error_data["std val"][1::3],**error_bar_fmt)

plt.bar(xvals+9, error_data["test"][1::3], color='tab:olive', width=0.5, edgecolor = "k")
plt.errorbar(xvals+9, error_data["test"][1::3], yerr=error_data["std test"][1::3],**error_bar_fmt)

plt.bar(xvals+11, error_data["ood"][1::3], color='grey', width=0.5, edgecolor = "k")
plt.errorbar(xvals+11, error_data["ood"][1::3], yerr=error_data["std ood"][1::3],**error_bar_fmt)

plt.bar(xvals+14, error_data["val"][2::3], color='r', width=0.5, edgecolor = "k")
plt.errorbar(xvals+14, error_data["val"][2::3], yerr=error_data["std val"][2::3],**error_bar_fmt)

plt.bar(xvals+16, error_data["test"][2::3], color='tab:olive', width=0.5, edgecolor = "k")
plt.errorbar(xvals+16, error_data["test"][2::3], yerr=error_data["std test"][2::3],**error_bar_fmt)

plt.bar(xvals+18, error_data["ood"][2::3], color='grey', width=0.5, edgecolor = "k")
plt.errorbar(xvals+18, error_data["ood"][2::3], yerr=error_data["std ood"][2::3],**error_bar_fmt)

plt.ylabel("mean squared error", fontsize=9, labelpad=9)
plt.savefig("error_overview.svg")