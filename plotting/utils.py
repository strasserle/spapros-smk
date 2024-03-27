import pandas as pd
import os
import re


def check_selected_n(results_dir, verbosity=1):
    """Checks whether the number of selected genes equals the requested n.

    Args:
        results_dir: Absolute path of results directory.
        verbosity: 0: no output, 1: print if selected_n != requested_n, 2: print check for each selection
    Returns:

    """
    selection_params_file = os.path.join(results_dir, "selection_parameters.csv")
    selection_params = pd.read_csv(os.path.join(selection_params_file), index_col=0)
    sel_dir = os.path.join(results_dir, "selection")
    for sel_file_path in os.listdir(sel_dir):
        if (not os.path.isfile(os.path.join(sel_dir, sel_file_path))) or sel_file_path.endswith("_info.csv"):
            continue
        selection = pd.read_csv(os.path.join(sel_dir, sel_file_path), index_col=0)
        if len(selection.columns) > 1:
            raise ValueError("More than one selection in file {sel_file_path}. Update 'check_selected_n' function")
        selected_n = selection.sum()[0]
        selection_name = os.path.basename(sel_file_path).rstrip(".csv")
        requested_n = get_n_from_params(selection_name, selection_params)
        if selected_n != requested_n and verbosity > 0:
            print(f"Selected {selected_n} genes for {selection_name} but requested {requested_n}.")
        elif verbosity > 1:
            print(f"Selected {selected_n} genes for {selection_name} as requested.")



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


def get_results_of_batch(eval_batch, results_dir="results", permut_dir=os.getcwd(), overview_dir="results"):
    """Read and merge eval_overview_file, selection_params_file, data_params_file

    Args:
        eval_batch:
            Name of evaluation batch
        results_dir:
            Either relative to current working directory or absolute path to directory containing eval_overview_file,
            selection_params_file and data_params_file.
        permut_dir:
            Directory from where the permutation path is relative to.


    Returns:
        Dataframe with evaluation results, selection and data parameters; one row for each selection

    """

    eval_overview_file = os.path.join(overview_dir, "evaluation_overview.csv")
    selection_params_file = os.path.join(overview_dir, "selection_parameters.csv")
    data_params_file = os.path.join(overview_dir, "data_parameters.csv")

    eval_overview = pd.read_csv(eval_overview_file, index_col=0)
    batch_mask = eval_overview["eval_batch"] == eval_batch
    eval_overview = eval_overview[batch_mask]
    eval_summary_list = eval_overview["eval_summary_file"].unique()
    selection_names = eval_overview["selection_name"].unique()
    selection_params = pd.read_csv(selection_params_file, index_col=0)
    data_params = pd.read_csv(data_params_file)

    metrics = set()
    all_results = pd.DataFrame()
    for eval_summary_file in eval_summary_list:
        eval_summary = pd.read_csv(os.path.join(results_dir, eval_summary_file), index_col=0)
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
        permutations = pd.read_csv(os.path.join(permut_dir, permutation_file), index_col=0)
        n_cells = permutations.sum(axis=0).reset_index(drop=False).rename(columns={0: "n_cells", "index": "permutation_name"})
        permutations_info = pd.read_csv(os.path.join(permut_dir, permutation_file.replace(".csv", "_info.csv")), index_col=0).reset_index(drop=True)
        permutations_info.rename(columns={"n": "n_datasets"}, inplace=True)
        permutations_info = pd.concat([permutations_info, n_cells], axis=1)
        # TODO for eval on one: collect all info and plot / aggregate

    if len(permutation_files) > 0:
        all_results = pd.merge(all_results, permutations_info, on="permutation_name")

    # sort metrics alphabetically
    metrics = sorted(metrics)

    return all_results, metrics


def minmax_norm(series, min, max):
    return (min + (series - series.min()) * (max - min) / (
                series.max() - series.min()))
