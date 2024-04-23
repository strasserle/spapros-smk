import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# from plotting.utils import *
from utils import *
from matplotlib.ticker import MaxNLocator


# eval_on_full: one selection (n=50), one evaluation (forest_clfs) for each permutation
# eval_on_one: evaluation on each single dataset once

# wd = "/lustre/groups/ml01/workspace/lena.strasser/spapros-smk"
# wd = 'D:\\Projects\\spapros-smk'

# setup 1: downloaded results with guacamole and run this script locally 
# wd = os.getcwd()
# data_dir = 'D:\\11. Semester\\MA\\benchmarking\\data\\test_data\\permut_data'
# fig_dir = "plotting/figures/"
# if not os.path.isdir(fig_dir): 
#     os.makedir(fig_dir)
# permut_dir = "D:/11. Semester/MA/"
# results_dir = "results_run2"
# plot_data_dir = "plotting/plot_data"

# setup 2: running in vscode in citrix: 
wd = os.getcwd()
print(wd)
data_dir = "/lustre/groups/ml01/workspace/lena.strasser/MA/benchmarking/data/test_data/permut_data"
fig_dir = "plotting/figures/"
if not os.path.isdir(fig_dir): 
    os.mkdir(fig_dir)
permut_dir = "/lustre/groups/ml01/workspace/lena.strasser/MA/"
results_dir = "results_run4"
# overview_dir = "results/bone_marrow_overviews"
overview_dir = "results_run4"

plot_data_dir = "plotting/plot_data"
if not os.path.isdir(plot_data_dir):
    os.mkdir(plot_data_dir)


dataset = 'HLCA'
# dataset = "eye"
# dataset = "bone_marrow"
# dataset= "liver_four"
# dataset = "liver"
# p, metrics = get_results_of_batch(f"eval_on_full_{dataset}", results_dir=results_dir, permut_dir=permut_dir, overview_dir=overview_dir)
eval_on_one, metrics = get_results_of_batch(f"eval_on_one_{dataset}", results_dir=results_dir, permut_dir=permut_dir, overview_dir=overview_dir)

# save
# file_name_p = os.path.join(plot_data_dir, f"permutations_{dataset}.csv")
# if not os.path.isfile(file_name_p): 
#     p.to_csv(file_name_p)
# else:
#     raise ValueError(f'File {file_name_p} already existing.')
print(metrics)
file_name_e = os.path.join(plot_data_dir, f"eval_on_one_{dataset}.csv")
if not os.path.isfile(file_name_e):
    eval_on_one.to_csv()
else:
    raise ValueError(f'File {file_name_e} already existing.')

# p = pd.read_csv(os.path.join(plot_data_dir, f"permutations_{dataset}"))
# metrics = {'forest_clfs accuracy', 'forest_clfs perct acc > 0.8'}
# eval_on_one = pd.read_csv(os.path.join(f"eval_on_one_{dataset}.txt"))

plt.style.use("fivethirtyeight")
bg_grey = "#f0f1f0"
# p.sort_values(by=["n_datasets"], inplace=True)

# Number of datasets in permutation
# fig = plt.figure(figsize=(10, 18))
# for metric in metrics:
#     plt.plot(p["n_datasets"], p[metric], "x", markersize=10, label=f"{metric} on full dataset")
#     color = plt.gca().get_lines()[-1].get_color()
#     plt.plot(p["n_datasets"].unique(), p[["n_datasets", metric]].groupby(by="n_datasets").mean(), "-x", markersize=10, label=f"{metric} mean over selections", color=color)
#     plt.scatter(eval_on_one["n_datasets"], eval_on_one[metric], s=minmax_norm(eval_on_one["n_cells"], min=10, max=40), label=f"{metric} on single datasets", color=color)
#     plt.plot(eval_on_one["n_datasets"].unique(), eval_on_one[["n_datasets", metric]].groupby(by="n_datasets").mean(), "--o", markersize=10, label=f"{metric} mean over selections and evaluations", color=color)
# legend_1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
# legend_sizes = (eval_on_one["n_cells"].min().round(-2), int(eval_on_one["n_cells"].mean().round(-3)), eval_on_one["n_cells"].max().round(-4))
# legend_labels = [str(size) for size in legend_sizes]
# legend_proxies = [plt.Line2D([0], [0], marker='o', color=legend_1.get_frame().get_facecolor(), markerfacecolor='black', markersize=np.sqrt(size), label=label)
#                   for size, label in zip(minmax_norm(pd.Series(legend_sizes), min=10, max=40), legend_labels)]
# ax_1 = plt.gca()
# ax_2 = plt.gca().twinx()
# ax_2.set_axis_off()
# legend_2 = ax_2.legend(handles=legend_proxies, title='Number of cells', loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# # plt.title("Reference dataset: " + eval_dataset + "\nSelection batch: " + eval_batch)
# plt.title(f"Dataset: {dataset}")
# ax_1.set_ylabel("Performance")
# ax_1.set_xlabel("Number of datasets in permutation")  # Permutation size in selection dataset / Selection on permutation of X datasets
# plt.tight_layout()
# plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_datasets_{dataset}_with_on_one.png")
# plt.show()

# mean per selection, one plot per metric
mean_eval = eval_on_one[["selection_data_id", "n_datasets", "n_cells"] + list(metrics)].groupby(by="selection_data_id").mean()
# grid: one plot per metric
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
# get default color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
for metric, ax, color in zip(metrics, axes, prop_cycle.by_key()['color']):
    # plt.plot(p["n_datasets"], p[metric], "x", markersize=10, label=f"{metric} on full dataset")
    # plt.plot(p["n_datasets"].unique(), p[["n_datasets", metric]].groupby(by="n_datasets").mean(), "-x", markersize=10, label=f"{metric} mean over selections", color=color)
    ax.scatter(mean_eval["n_datasets"], mean_eval[metric], s=minmax_norm(mean_eval["n_cells"], min=10, max=40), label=f"{metric} per selection", color=color)
    ax.plot(mean_eval["n_datasets"].unique(), mean_eval.groupby("n_datasets").mean()[metric], "--x", markersize=15, label=f"{metric} mean over evaluations", color=color)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(metric, fontsize=15, pad=20)  # fontweight="bold"
legend_1 = axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
legend_1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
legend_sizes = (eval_on_one["n_cells"].min().round(-2), int(eval_on_one["n_cells"].mean().round(-3)), eval_on_one["n_cells"].max().round(-4))
legend_labels = [str(size) for size in legend_sizes]
legend_proxies = [plt.Line2D([0], [0], marker='o', color=legend_1.get_frame().get_facecolor(), markerfacecolor='black', markersize=np.sqrt(size), label=label)
                  for size, label in zip(minmax_norm(pd.Series(legend_sizes), min=10, max=40), legend_labels)]
ax_1 = plt.gca()
ax_2 = plt.gca().twinx()
ax_2.set_axis_off()
legend_2 = ax_2.legend(handles=legend_proxies, title='Number of cells', loc='upper left', bbox_to_anchor=(1.1, 1))
ax_3 = ax_2.twiny()
ax_3.set_axis_off()
# third legend: one grey dot saying "score per selection" and one dotted line with a dot saying "mean over evaluations" also in grey, below legend_2
legend_3 = ax_3.legend(handles=[plt.Line2D([0], [0], marker='o', color=bg_grey, markerfacecolor='grey', markersize=10, label="score per selection"),
                                plt.Line2D([0], [0], marker='x', color='grey', markerfacecolor='grey', markersize=15, linestyle="--", label="mean over evaluations")], loc='upper left', bbox_to_anchor=(1.05, 0.5))
# suptitle in same fontsize as axes titles
plt.suptitle(f"Dataset: {dataset}", x=0.5, fontsize=ax.title.get_fontsize(), va='center')
fig.text(0.4, 0.03, 'Number of datasets in permutation', ha='center', va='center')
fig.text(0.02, 0.5, 'Performance', ha='center', va='center', rotation='vertical')
plt.tight_layout()
plt.subplots_adjust(left=0.07, bottom=0.1)
plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_datasets_{dataset}_per_sel.png")
# plt.show()

# TODO add eval on full 

# Number of cells in permutation
# p.sort_values(by=["n_cells"], inplace=True)
# for metric in metrics:
#     # plt.plot(p["n_cells"], p[metric], "-o", label=metric)
#     # color = plt.gca().get_lines()[-1].get_color()
#     plt.scatter(eval_on_one["n_datasets"], eval_on_one[metric], s=minmax_norm(eval_on_one["n_cells"], min=10, max=40), label=f"{metric} on single datasets", color=color)
# plt.legend()
# plt.title("Reference dataset: " + eval_dataset + "\nSelection batch: " + eval_batch)
# plt.ylabel("Performance")
# plt.xlabel("Number of cells in permutation")
# # plt.title("Style: " + style)
# plt.tight_layout()
# plt.xlim(0, 1000)
# # plt.xlim(27000, 30000)
# plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_cells_{eval_dataset}_{eval_batch}_lim.png")
# plt.show()

# with limit
# p.sort_values(by=["n_cells"], inplace=True)
# eval_on_one.sort_values(by=["n_cells"], inplace=True)
# fig = plt.figure(figsize=(10, 8))
# for metric in metrics:
#     plt.plot(p["n_cells"], p[metric], "x", markersize=10, label=f"{metric} on full dataset")
#     color = plt.gca().get_lines()[-1].get_color()
#     plt.scatter(eval_on_one["n_cells"], eval_on_one[metric], s=minmax_norm(eval_on_one["n_datasets"], min=10, max=40), label=f"{metric} on single datasets", color=color)
#     plt.plot(eval_on_one["n_cells"].unique(), eval_on_one[["n_cells", metric]].groupby(by="n_cells").mean(), "--o", markersize=10, label=f"{metric} mean over selections and evaluations", color=color)
# legend_1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))
# legend_sizes = (eval_on_one["n_datasets"].min(), int(eval_on_one["n_datasets"].mean()), eval_on_one["n_datasets"].max())
# legend_labels = [str(size) for size in legend_sizes]
# legend_proxies = [plt.Line2D([0], [0], marker='o', color=legend_1.get_frame().get_facecolor(), markerfacecolor='black', markersize=np.sqrt(size), label=label)
#                   for size, label in zip(minmax_norm(pd.Series(legend_sizes), min=10, max=40), legend_labels)]
# ax_1 = plt.gca()
# ax_2 = plt.gca().twinx()
# ax_2.set_axis_off()
# legend_2 = ax_2.legend(handles=legend_proxies, title='Number of datasets', loc='upper left', bbox_to_anchor=(1.05, 1))
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# # plt.title("Reference dataset: " + eval_dataset + "\nSelection batch: " + eval_batch)
# plt.title(f"Dataset: {dataset}")
# ax_1.set_ylabel("Performance")
# plt.xlabel("Number of cells in permutation")
# plt.tight_layout()
# # plt.xlim(0, 1000)
# # plt.xlim(27000, 30000)
# plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_cells_{dataset}_with_on_one_limr.png")
# plt.show()

# # exclude huge dataset
# id_large = "'998d8dbd-2f42-4611-9973-2da95db46c29"
# large_mask = [id_large not in l for l in p["permutations"]]
# p = p[large_mask]
#
# p.sort_values(by=["n_datasets"], inplace=True)
# plt.style.use("fivethirtyeight")
# for metric in metrics:
#     plt.plot(p["n_datasets"], p[metric], "-o", label=metric)
# plt.legend()
# plt.title("Reference dataset: " + eval_dataset + "\nSelection batch: " + eval_batch)
# plt.ylabel("Performance")
# plt.xlabel("Number of datasets in permutation")
# # plt.title("Style: " + style)
# plt.tight_layout()
# plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_datasets_{eval_dataset}_{eval_batch}_small.png")
# plt.show()
#
# p.sort_values(by=["n_datasets"], inplace=True)
# plt.style.use("fivethirtyeight")
# for metric in metrics:
#     plt.plot(p["n_datasets"].unique(), p[[metric, "n_datasets"]].groupby(by="n_datasets").mean(), "-o", label=metric)
# plt.legend()
# plt.title("Reference dataset: " + eval_dataset + "\nSelection batch: " + eval_batch)
# plt.ylabel("Performance")
# plt.xlabel("Number of datasets in permutation")
# # plt.title("Style: " + style)
# plt.tight_layout()
# plt.savefig(fig_dir + f"scatterlineplot_performance_vs_n_datasets_{eval_dataset}_{eval_batch}_small_mean.png")
# plt.show()

# e = eval_on_one
# cmap = {eds: np.random.rand(3,) for eds in e['eval_dataset'].unique()}
# shapemap = {sel}

# plt.scatter(e['n_datasets'], e['cluster_similarity nmi_21_60'], s=10, c=e['eval_dataset'].map(cmap), label="Evaluation dataset")
# plt.legend()

# plt.savefig('plotting/figures/tmp1.png')

# import seaborn as sns
# sns.boxplot(x=e['n_datasets'], y=e['cluster_similarity nmi_21_60'])

# plt.savefig('plotting/figures/tmp2.png')

# f = e[['permutations', 'cluster_similarity nmi_21_60', 'n_datasets', 'permutation_name']].copy()
# f[[f'ds_{i}' for i in range(7)]]= f['permutations'].str.lstrip("(").str.rstrip().str.rstrip(')').str.rstrip(',').str.split(", ", expand=True)
# f = f.melt(id_vars=['n_datasets', 'cluster_similarity nmi_21_60', "permutation_name"], value_vars=[f'ds_{i}' for i in range(7)]).dropna(how='any')
# plt.figure(figsize=(18, 8))
# sns.boxplot(data=f, x='n_datasets', y='cluster_similarity nmi_21_60', hue="value")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.title('Distribution over evaluation datasets and permutations')
# plt.subplots_adjust(top=0.9)
# plt.savefig('plotting/figures/tmp3.png')

# del f['variable']
# mean_f = f.groupby(by=['permutation_name', 'value']).mean()
# plt.figure(figsize=(18, 8))
# sns.boxplot(data=mean_f, x='n_datasets', y='cluster_similarity nmi_21_60', hue="value")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()
# plt.title('Distribution over permutations, mean over evaluation datasets')
# plt.subplots_adjust(top=0.9)
# plt.savefig('plotting/figures/tmp4.png')

# # mean_f['n_datasets'].value_counts()
# # n_datasets
# # 6.0    42
# # 5.0    35
# # 4.0    28
# # 3.0    21
# # 2.0    14
# # 1.0     7
# # 7.0     7




# plt.close('all')






