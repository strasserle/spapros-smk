import pandas as pd
import os

# dataset = "bone_marrow"
dataset="eye"
plot_data_dir = "plotting/plot_data"

p = pd.read_csv(os.path.join(plot_data_dir, f"permutations_{dataset}.csv"), index_col=0)
eval_on_one  = pd.read_csv(os.path.join(plot_data_dir, f"eval_on_one_{dataset}.csv"), index_col=0)

metric = 'cluster_similarity nmi_21_60'

eval_on_one[f"{metric}_rank"] = eval_on_one[metric].rank(ascending=False)
eval_on_one["permutations_list"] = eval_on_one["permutations"].str.strip('"(').str.strip(')"').str.split(', ').apply(lambda x: [y.replace("'", "").replace('"', "") for y in x])

n_datasets = eval_on_one["n_datasets"].max()
ds_ids = eval_on_one[eval_on_one["n_datasets"] == n_datasets]["permutations_list"].values[0]
ds_ids = [x.replace("'", "").replace('"', "").strip() for x in ds_ids]
dr = pd.Series(index=ds_ids)
for i, ds_id in enumerate(ds_ids):
    ds_mask = eval_on_one["permutations_list"].apply(lambda x: ds_id in x)
    dr[ds_id] = eval_on_one[ds_mask][f"{metric}_rank"].mean()

dr.to_csv(os.path.join(plot_data_dir, f"rank_datasets_{dataset}.csv"))