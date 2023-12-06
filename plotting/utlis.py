import pandas as pd
import os
import re


def get_n_from_params(selection_name, selection_params):
    """

    Args:
        selection_name: Selection name, e.g. spapros_0_liver_1
        selection_params: Dataframe with selection paramseters.

    Returns:
        n: Number of genes in selection.

    """

    selection_id = re.search(r'_\d+_', selection_name).group(0).strip("_")
    selection_params = selection_params.loc[int(selection_id)]
    n = selection_params["n"]
    return n


def get_results_of_batch(eval_batch, results_dir = "results"):
    """Read and merge eval_overview_file, selection_params_file, data_params_file

    Args:
        eval_batch:
            Name of evaluation batch
        results_dir:
            Either relative to current working directory or absolute path to directory containing eval_overview_file,
            selection_params_file and data_params_file.


    Returns:
        Dataframe with evaluation results, selection and data parameters; one row for each selection

    """

    eval_overview_file = os.path.join(results_dir, "evaluation_overview.csv")
    selection_params_file = os.path.join(results_dir, "selection_parameters.csv")
    data_params_file = os.path.join(results_dir, "data_parameters.csv")

    eval_overview = pd.read_csv(eval_overview_file, index_col=0)
    batch_mask = eval_overview["eval_batch"] == eval_batch
    eval_overview = eval_overview[batch_mask]
    eval_summary_list = eval_overview["eval_summary_file"].unique()
    selection_names = eval_overview[batch_mask]["selection_name"].unique()
    selection_params = pd.read_csv(selection_params_file, index_col=0)
    data_params = pd.read_csv(data_params_file)

    metrics = set()
    all_results = pd.DataFrame()
    for eval_summary_file in eval_summary_list:
        eval_summary = pd.read_csv(os.path.join(wd, results_dir, eval_summary_file), index_col=0)
        selection_mask = eval_summary.index.isin(selection_names)
        eval_summary = eval_summary[selection_mask]
        eval_summary.dropna(axis=1, inplace=True)
        metrics = metrics.union(set(eval_summary.columns))
        eval_summary["n"] = [get_n_from_params(x, selection_params) for x in eval_summary.index]
        eval_dataset = os.path.basename(eval_summary_file).rstrip("_summary.csv")
        eval_summary["eval_dataset"] = eval_dataset
        eval_summary["selection_data_id"] = eval_summary.index.str.split("_", expand=True).get_level_values(-1).astype(int)
        eval_summary = pd.merge(eval_summary, data_params, left_on="selection_data_id", right_on="data_id")
        all_results = pd.concat([all_results, eval_summary])


    # get the number of datasets for a permutation
    all_results[["permutation_file", "permutation_name"]] = all_results["permutation"].str.split(":", expand=True)
    permutation_files = all_results["permutation_file"].unique()
    for permutation_file in permutation_files:
        permutations = pd.read_csv(permutation_file, index_col=0)
        n_cells = permutations.sum(axis=0).reset_index(drop=False).rename(columns={0: "n_cells", "index": "permutation_name"})
        permutations_info = pd.read_csv(permutation_file.replace(".csv", "_info.csv"), index_col=0).reset_index(drop=True)
        permutations_info.rename(columns={"n": "n_datasets"}, inplace=True)
        permutations_info = pd.concat([permutations_info, n_cells], axis=1)
        # TODO for eval on one: collect all info and plot / aggregate

    p = pd.merge(all_results, permutations_info, on="permutation_name")

    return p 
