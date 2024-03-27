import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from plotting.utils import *


# wd = "/lustre/groups/ml01/workspace/lena.strasser/spapros-smk"
# wd = 'D:\\Projects\\spapros-smk'
wd = os.getcwd()

# How much variation is captured with a given number of genes?
# 6 selections (n=25, 50, ...)
# evaluations with 2 metrics for each selection for each single dataset as reference
fig_dir = "plotting/figures/"

# eval_batch = "eval_var_one_liver_four"
eval_batch = "eval_var_one_bone_marrow"
eval_on_full_name = "eval_on_full_bone_marrow"
results_dir = "results"
eval_overview_file = os.path.join(results_dir, "evaluation_overview.csv")
selection_params_file = os.path.join(results_dir, "selection_parameters.csv")

eval_overview = pd.read_csv(os.path.join(wd, eval_overview_file), index_col=0)
batch_mask = eval_overview["eval_batch"] == eval_batch
eval_overview = eval_overview[batch_mask]
eval_summary_list = eval_overview["eval_summary_file"].unique()
# metrics = eval_overview["metric"].unique()
n_datasets = len(eval_summary_list)  # Each dataset should be used as reference once
selection_names = eval_overview["selection_name"].unique()
selection_params = pd.read_csv(os.path.join(wd, selection_params_file), index_col=0)

metrics = set()
all_results = pd.DataFrame()
for eval_summary_file in eval_summary_list:
    eval_summary = pd.read_csv(os.path.join(wd, results_dir, eval_summary_file), index_col=0)
    selection_mask = eval_summary.index.isin(selection_names)
    eval_summary = eval_summary[selection_mask]
    metrics = metrics.union(set(eval_summary.columns))
    # assert all([x in metrics for x in eval_summary.columns]), f"Not all metrics are in the evaluation results. "
    eval_summary["n"] = [get_n_from_params(x, selection_params) for x in eval_summary.index]
    eval_dataset = os.path.basename(eval_summary_file).rstrip("_summary.csv")
    eval_summary["eval_dataset"] = eval_dataset
    all_results = pd.concat([all_results, eval_summary])


# sanity check
all_results[["selection_name_id", "selection_dataset_id"]] = all_results.index.str.extract(r'^(\D+_\d+)_([\w_]+)$').values
n_selections = all_results["selection_name_id"].nunique()
print(f"Found evaluation results of {n_selections} different selections.")
assert all([x in all_results.index for x in selection_names]), f"Not all selections are in the evaluation results. " \
                                                                  f"There should be {len(selection_names)}."
assert all([x == (n_datasets * n_selections) for x in all_results["selection_dataset_id"].value_counts()])

eval_on_full = get_results_of_batch(eval_on_full_name, results_dir=results_dir, permut_dir="D:/11. Semester/MA/")

# plot
plt.style.use("fivethirtyeight")
all_results.sort_values(by="n", inplace=True)
fig = plt.figure(figsize=(7, 6))
metrics = ['forest_clfs accuracy',
             'forest_clfs perct acc > 0.8',
             'knn_overlap_X_id mean_overlap_AUC batch_mean']
for metric in metrics:
    metric_results = all_results[["n", metric]].groupby(by="n").mean()
    plt.plot(all_results["n"].unique(), metric_results, "o-", label=metric)
    plt.scatter(all_results["n"], all_results[metric], s=2)
    color = plt.gca().get_lines()[-1].get_color()
    # if metric in eval_on_full:
        # plt.scatter(eval_on_full["n"], eval_on_full[metric], s=10, marker="X", color=color)
        # plt.scatter(eval_on_full["n"].mean(), eval_on_full[metric].mean(), s=50, marker="X", label=f"{metric} on full dataset", color=color)
plt.xlabel("Number of selected genes")
plt.ylabel("Performance")  # \n(mean over selection \nand evaluation datasets")
plt.title("Selection batch: " + eval_batch)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))
# plt.tight_layout()
plt.subplots_adjust(bottom=0.4, top=0.9, left=0.1, right=0.9)
plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_{eval_batch}_mean_with_eval_on_full.png", dpi=300)
plt.show()

# heatmap


