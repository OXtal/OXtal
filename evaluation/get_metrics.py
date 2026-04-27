
import argparse
from pathlib import Path
import pandas as pd


def load_refcodes(txt_path: Path):
    """Load CSD refcodes from a text file (one per line)."""
    with txt_path.open("r") as f:
        lines = [line.strip() for line in f]
    return [x for x in lines if x]


def analyze_dataset(results: pd.DataFrame, dataset_name: str, codes, csv_label: str):
    """
    Reproduce the original analysis & prints for a single dataset
    given a results DataFrame and a list of refcodes.

    Returns a dict of metrics for later tabulation.
    """
    # Deduplicate while preserving order
    seen = set()
    unique_codes = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique_codes.append(c)

    n_dataset = len(unique_codes)
    print(f"Processing dataset: {dataset_name} with {n_dataset} entries")
    print(unique_codes)

    # Default metrics (in case we bail early)
    metrics = {
        # "csv_file": csv_label,
        "dataset": dataset_name,
        "n_dataset": n_dataset,
        "n_found": 0,
        "n_filtered": 0,
        "col_s": None,
        "pac_S": None,
        "n_pass_C": 0,
        "pac_C": None,
        "rec_S": None,
        "n_conf_C": 0,
        "rec_C": None,
        "n_match_C": 0,
        "match_rate": None,
    }

    if n_dataset == 0:
        print("No entries in this dataset list, skipping.")
        print("===================================================")
        return metrics

    # Filter rows whose csd_refcode is in the dataset list
    filtered_results = results[results["csd_refcode"].str.contains('|'.join(unique_codes), na=False)]
    n_filtered = len(filtered_results)
    n_found = filtered_results["csd_refcode"].nunique()

    metrics["n_found"] = n_found
    metrics["n_filtered"] = n_filtered

    print(f"{n_found} out of {n_dataset} found in results")
    print("===================================================")

    if n_filtered == 0:
        print(f"No rows in results for dataset {dataset_name}, skipping rate calculations.")
        print("===================================================")
        return metrics

    # Clash rate over samples
    clash_rate = filtered_results["clash"].sum() / n_filtered
    metrics["col_s"] = clash_rate
    print(f"**Clash rate for {dataset_name}: {clash_rate:.4f}")

    # Packing_sample pass rate over samples
    packing_sample_pass_rate = filtered_results["passed"].sum() / n_filtered
    metrics["pac_S"] = packing_sample_pass_rate
    print(f"**Packing_sample pass rate for {dataset_name}: {packing_sample_pass_rate:.4f}")

    # Packing_crystal pass rate over unique crystals in dataset
    passed_df = filtered_results[filtered_results["passed"] == True]
    passed_codes = set(passed_df["csd_refcode"])
    n_passed_crystal = len(passed_codes)
    metrics["n_pass_C"] = n_passed_crystal
    packing_crystal_pass_rate = n_passed_crystal / n_dataset
    metrics["pac_C"] = packing_crystal_pass_rate

    print(f"Number of passed for {dataset_name}: {n_passed_crystal}")
    print(f"**Packing_crystal pass rate for {dataset_name}: {packing_crystal_pass_rate:.4f}")
    print(passed_codes)

    # Recovery_sample and Recovery_crystal
    df_conf = filtered_results[
        (filtered_results["rmsd_1"] < 0.5) & (filtered_results["clash"] == False)
    ]
    recovery_sample_pass_rate = len(df_conf) / n_filtered
    metrics["rec_S"] = recovery_sample_pass_rate

    print(f"**Recovery_sample pass rate for {dataset_name}: {recovery_sample_pass_rate:.4f}")

    conformer_codes = set(df_conf["csd_refcode"])
    n_conformers = len(conformer_codes)
    metrics["n_conf_C"] = n_conformers
    print(f"Number of conformers for {dataset_name}: {n_conformers}")
    print(conformer_codes)

    recovery_crystal_rate = n_conformers / n_dataset
    metrics["rec_C"] = recovery_crystal_rate
    print(f"**Recovery_crystal rate for {dataset_name}: {recovery_crystal_rate:.4f}")

    # Match rate
    match_df = filtered_results[
        (filtered_results["passed"] == True)
        & (filtered_results["rmsd_15"] < 2.0)
        & (filtered_results["clash"] == False)
    ]
    match_codes = set(match_df["csd_refcode"])
    n_matches = len(match_codes)
    metrics["n_match_C"] = n_matches

    print(f"Number of matches for {dataset_name}: {n_matches}")
    print(match_codes)

    match_rate = n_matches / n_dataset
    metrics["match_rate"] = match_rate
    print(f"**Match rate for {dataset_name}: {match_rate:.4f}")
    print("===================================================")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze multiple result CSVs for multiple datasets.")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Paths to result CSV files to analyze."
    )
    parser.add_argument(
        "--txt-dir",
        type=Path,
        default=Path("."),
        help="Directory containing dataset TXT files (default: current directory)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["rigid", "flexible", "comp5", "comp6", "comp7"],
        help="Dataset names (TXT files are expected to be <name>.txt)."
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to write a summary CSV of all metrics."
    )

    args = parser.parse_args()

    all_metrics = []

    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        print("\n############################################")
        print(f"Analyzing results file: {csv_path}")
        print("############################################\n")

        results = pd.read_csv(csv_path)

        # Basic sanity check
        required_cols = {"csd_refcode", "clash", "passed", "rmsd_1", "rmsd_15"}
        missing = required_cols - set(results.columns)
        if missing:
            print(f"[ERROR] {csv_path} is missing required columns: {missing}")
            continue

        for dataset_name in args.datasets:
            txt_path = args.txt_dir / f"{dataset_name}.txt"
            if not txt_path.exists():
                print(f"[WARN] Missing dataset file: {txt_path} – skipping {dataset_name}")
                print("===================================================")
                continue

            dataset_codes = load_refcodes(txt_path)
            metrics = analyze_dataset(results, dataset_name, dataset_codes, csv_label=str(csv_path))
            all_metrics.append(metrics)

    # Create summary table
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        print("\n================ SUMMARY TABLE ================\n")
        # Print without index for a clean console table
        print(summary_df.to_string(index=False, float_format="{:.3f}".format))

        if args.out_csv is not None:
            summary_df.to_csv(args.out_csv, index=False)
            print(f"\nSummary metrics written to: {args.out_csv}")
    else:
        print("No metrics collected (check inputs / datasets).")


if __name__ == "__main__":
    main()