import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


# ============================================================
# 0. Project path configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 1. Import PLRT-Hybrid modules
# ============================================================

try:
    from plrt_hybrid.preprocessing import (
        KEY_COLS,
        RAW_NUM_COLS,
        STD_NUM_COLS,
        build_key_lookup,
        coerce_numeric,
    )
    from plrt_hybrid.simulation import (
        run_plrt,
        run_plrt_hybrid,
    )
    from plrt_hybrid.metrics import get_r2

except ImportError as exc:
    raise ImportError(
        "\nCould not import the PLRT-Hybrid modules.\n"
        "Confirm that the following files exist:\n"
        "  src/plrt_hybrid/preprocessing.py\n"
        "  src/plrt_hybrid/simulation.py\n"
        "  src/plrt_hybrid/metrics.py\n\n"
        f"Original import error:\n{exc}"
    ) from exc


# ============================================================
# 2. Default configuration
# ============================================================

DEFAULT_RES_ID = 41
DEFAULT_FREQUENCIES = [1, 2, 7, 30, 90, 180]
DEFAULT_METHODS = ["plrt", "plrt_hybrid"]

DATA_DIR = PROJECT_ROOT / "data" / "example"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
SUMMARY_DIR = OUTPUT_DIR / "summary"

KEY_FILE = DATA_DIR / "41_key.csv"
META_FILE = DATA_DIR / "41_meta.csv"
RAW_FILE = DATA_DIR / "41_raw.csv"
STD_FILE = DATA_DIR / "41_std.csv"


METHOD_FUNCTIONS: dict[str, Callable] = {
    "plrt": run_plrt,
    "plrt_hybrid": run_plrt_hybrid,
}


# ============================================================
# 3. Command-line arguments
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PLRT and PLRT-Hybrid for the example reservoir "
            "at multiple initialization frequencies."
        )
    )

    parser.add_argument(
        "--res-id",
        type=int,
        default=DEFAULT_RES_ID,
        help=f"Reservoir ID. Default: {DEFAULT_RES_ID}",
    )

    parser.add_argument(
        "--frequencies",
        type=int,
        nargs="+",
        default=DEFAULT_FREQUENCIES,
        help=(
            "Initialization frequencies in days. "
            "Default: 1 2 7 30 90 180"
        ),
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHOD_FUNCTIONS.keys()),
        default=DEFAULT_METHODS,
        help="Models to run. Default: plrt plrt_hybrid",
    )

    parser.add_argument(
        "--skip-save-predictions",
        action="store_true",
        help="Do not save full prediction CSV files.",
    )

    return parser.parse_args()


# ============================================================
# 4. File and directory checks
# ============================================================

def create_output_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def validate_input_files() -> None:
    required_files = [
        KEY_FILE,
        META_FILE,
        RAW_FILE,
        STD_FILE,
    ]

    missing_files = [
        file_path for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "The following required example files were not found:\n"
            f"{missing_text}\n\n"
            "Expected directory:\n"
            f"  {DATA_DIR}"
        )


# ============================================================
# 5. Data loading and preparation
# ============================================================

def load_example_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load metadata, monthly scaling parameters, raw data,
    and standardized data.
    """

    meta = pd.read_csv(META_FILE)
    key = pd.read_csv(KEY_FILE)
    raw = pd.read_csv(RAW_FILE)
    std = pd.read_csv(STD_FILE)

    return meta, key, raw, std


def prepare_dataframe_dates(df: pd.DataFrame, name: str) -> None:
    """
    Convert the date column to datetime and create month if needed.

    The DataFrame is modified in place.
    """

    if "date" not in df.columns:
        return

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    invalid_date_count = int(df["date"].isna().sum())

    if invalid_date_count > 0:
        raise ValueError(
            f"{name} contains {invalid_date_count} invalid date values."
        )

    if "month" not in df.columns:
        df["month"] = df["date"].dt.month


def prepare_example_data(
    meta: pd.DataFrame,
    key: pd.DataFrame,
    raw: pd.DataFrame,
    std: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Convert required columns to numeric types, parse dates,
    and sort the example data.
    """

    coerce_numeric(key, KEY_COLS)
    coerce_numeric(std, STD_NUM_COLS)
    coerce_numeric(raw, RAW_NUM_COLS)

    prepare_dataframe_dates(meta, "metadata")
    prepare_dataframe_dates(key, "standardization parameters")
    prepare_dataframe_dates(raw, "raw data")
    prepare_dataframe_dates(std, "standardized data")

    if "res_id" in key.columns:
        key["res_id"] = pd.to_numeric(
            key["res_id"],
            errors="coerce",
        )

    if "res_id" in raw.columns:
        raw["res_id"] = pd.to_numeric(
            raw["res_id"],
            errors="coerce",
        )

    if "res_id" in std.columns:
        std["res_id"] = pd.to_numeric(
            std["res_id"],
            errors="coerce",
        )

    if "month" in key.columns:
        key["month"] = pd.to_numeric(
            key["month"],
            errors="coerce",
        )

    if "date" in raw.columns:
        raw = raw.sort_values(
            ["res_id", "date"]
        ).reset_index(drop=True)

    if "date" in std.columns:
        std = std.sort_values(
            ["res_id", "date"]
        ).reset_index(drop=True)

    return meta, key, raw, std


# ============================================================
# 6. Data validation
# ============================================================

def validate_required_columns(
    raw: pd.DataFrame,
    std: pd.DataFrame,
    key: pd.DataFrame,
) -> None:
    raw_required = {
        "res_id",
        "date",
        "storage_pre",
        "inflow",
        "release",
    }

    std_required = {
        "res_id",
        "date",
        "month",
        "storage_pre",
        "release_pre",
        "inflow",
        "sto_diff",
        "release_roll7",
        "inflow_roll7",
        "storage_x_inflow",
        "inflow2",
        "release_pre2",
        "max_sto",
        "rts",
        "is_weekend",
        "sin_DOY",
        "cos_DOY",
        "quarter",
    }

    key_required = {
        "res_id",
        "month",
    }.union(KEY_COLS)

    missing_raw = sorted(raw_required - set(raw.columns))
    missing_std = sorted(std_required - set(std.columns))
    missing_key = sorted(key_required - set(key.columns))

    error_messages: list[str] = []

    if missing_raw:
        error_messages.append(
            "Missing raw-data columns: "
            + ", ".join(missing_raw)
        )

    if missing_std:
        error_messages.append(
            "Missing standardized-data columns: "
            + ", ".join(missing_std)
        )

    if missing_key:
        error_messages.append(
            "Missing standardization-parameter columns: "
            + ", ".join(missing_key)
        )

    if error_messages:
        raise ValueError("\n".join(error_messages))


def validate_reservoir_data(
    res_id: int,
    raw: pd.DataFrame,
    std: pd.DataFrame,
    key: pd.DataFrame,
) -> None:
    raw_res = raw.loc[raw["res_id"] == res_id].copy()
    std_res = std.loc[std["res_id"] == res_id].copy()
    key_res = key.loc[key["res_id"] == res_id].copy()

    if raw_res.empty:
        raise ValueError(
            f"res_id={res_id} was not found in {RAW_FILE.name}."
        )

    if std_res.empty:
        raise ValueError(
            f"res_id={res_id} was not found in {STD_FILE.name}."
        )

    if key_res.empty:
        raise ValueError(
            f"res_id={res_id} was not found in {KEY_FILE.name}."
        )

    raw_res = raw_res.sort_values("date").reset_index(drop=True)
    std_res = std_res.sort_values("date").reset_index(drop=True)

    if len(raw_res) != len(std_res):
        raise ValueError(
            f"Raw and standardized data lengths differ for res_id={res_id}: "
            f"{len(raw_res)} versus {len(std_res)}."
        )

    if not raw_res["date"].equals(std_res["date"]):
        mismatch = raw_res.loc[
            raw_res["date"] != std_res["date"],
            ["date"],
        ]

        raise ValueError(
            f"Raw and standardized dates do not match for res_id={res_id}. "
            f"Number of mismatched rows: {len(mismatch)}."
        )

    available_months = set(
        pd.to_numeric(
            key_res["month"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .tolist()
    )

    required_months = set(
        std_res["month"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    missing_months = sorted(
        required_months - available_months
    )

    if missing_months:
        raise ValueError(
            f"Standardization parameters are unavailable for months: "
            f"{missing_months}"
        )

    duplicate_key_rows = (
        key_res.groupby(["res_id", "month"])
        .size()
        .loc[lambda values: values > 1]
    )

    if not duplicate_key_rows.empty:
        print(
            "Warning: duplicate key rows were found. "
            "build_key_lookup() will use the first row "
            "for each reservoir-month combination."
        )


# ============================================================
# 7. Model execution
# ============================================================

def get_observed_release(
    raw: pd.DataFrame,
    res_id: int,
) -> np.ndarray:
    observed = (
        raw.loc[raw["res_id"] == res_id]
        .sort_values("date")["release"]
        .to_numpy(dtype=float)
    )

    return observed


def validate_prediction_output(
    prediction_df: pd.DataFrame,
    observed: np.ndarray,
    method: str,
    frequency: int,
) -> None:
    required_output_columns = {
        "date",
        "release_pred_raw",
        "storage_pred_raw",
        "plrt_mode",
    }

    missing_columns = sorted(
        required_output_columns - set(prediction_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{method}, frequency={frequency}: "
            "prediction output is missing columns: "
            + ", ".join(missing_columns)
        )

    if len(prediction_df) != len(observed):
        raise ValueError(
            f"{method}, frequency={frequency}: "
            f"prediction length ({len(prediction_df)}) differs from "
            f"observation length ({len(observed)})."
        )

    predicted = pd.to_numeric(
        prediction_df["release_pred_raw"],
        errors="coerce",
    )

    missing_predictions = int(predicted.isna().sum())

    if missing_predictions > 0:
        raise ValueError(
            f"{method}, frequency={frequency}: "
            f"{missing_predictions} release predictions are missing."
        )

    if not np.isfinite(predicted.to_numpy()).all():
        raise ValueError(
            f"{method}, frequency={frequency}: "
            "non-finite release predictions were generated."
        )


def run_single_model(
    method: str,
    model_function: Callable,
    res_id: int,
    frequency: int,
    raw: pd.DataFrame,
    std: pd.DataFrame,
    key_lookup: dict,
    save_predictions: bool,
) -> dict[str, int | float | str]:
    print(
        f"Running {method:12s} | "
        f"reservoir={res_id} | "
        f"frequency={frequency:3d} days"
    )

    prediction_df = model_function(
        res_id=res_id,
        std_df=std,
        raw_df=raw,
        key_lookup=key_lookup,
        frequency=frequency,
    )

    observed = get_observed_release(
        raw=raw,
        res_id=res_id,
    )

    validate_prediction_output(
        prediction_df=prediction_df,
        observed=observed,
        method=method,
        frequency=frequency,
    )

    predicted = prediction_df[
        "release_pred_raw"
    ].to_numpy(dtype=float)

    r2 = get_r2(
        actual=observed,
        pred=predicted,
    )

    prediction_df = prediction_df.copy()

    if "release_observed" not in prediction_df.columns:
        prediction_df["release_observed"] = observed

    prediction_df["method"] = method
    prediction_df["frequency"] = frequency

    output_file = (
        PREDICTION_DIR
        / f"reservoir_{res_id}_{method}_freq_{frequency}.csv"
    )

    if save_predictions:
        prediction_df.to_csv(
            output_file,
            index=False,
        )

    mode_count = int(
        prediction_df["plrt_mode"].notna().sum()
    )

    unique_modes = sorted(
        prediction_df["plrt_mode"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        f"  Completed | R2={r2:.4f} | "
        f"n={len(prediction_df)} | "
        f"modes={unique_modes}"
    )

    return {
        "res_id": res_id,
        "method": method,
        "frequency": frequency,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "n_observations": int(len(observed)),
        "n_mode_predictions": mode_count,
        "modes_used": ",".join(
            str(mode) for mode in unique_modes
        ),
        "prediction_file": (
            str(output_file.relative_to(PROJECT_ROOT))
            if save_predictions
            else ""
        ),
        "status": "success",
        "error": "",
    }


def run_all_models(
    res_id: int,
    methods: list[str],
    frequencies: list[int],
    raw: pd.DataFrame,
    std: pd.DataFrame,
    key_lookup: dict,
    save_predictions: bool,
) -> pd.DataFrame:
    results: list[dict] = []

    for method in methods:
        model_function = METHOD_FUNCTIONS[method]

        for frequency in frequencies:
            try:
                result = run_single_model(
                    method=method,
                    model_function=model_function,
                    res_id=res_id,
                    frequency=frequency,
                    raw=raw,
                    std=std,
                    key_lookup=key_lookup,
                    save_predictions=save_predictions,
                )

            except Exception as exc:
                print(
                    f"  Failed | {method} | "
                    f"frequency={frequency} | {exc}"
                )

                result = {
                    "res_id": res_id,
                    "method": method,
                    "frequency": frequency,
                    "r2": np.nan,
                    "n_observations": np.nan,
                    "n_mode_predictions": np.nan,
                    "modes_used": "",
                    "prediction_file": "",
                    "status": "failed",
                    "error": str(exc),
                }

            results.append(result)

    return pd.DataFrame(results)


# ============================================================
# 8. Summary reporting
# ============================================================

def print_dataset_summary(
    res_id: int,
    meta: pd.DataFrame,
    raw: pd.DataFrame,
    std: pd.DataFrame,
    key: pd.DataFrame,
) -> None:
    raw_res = (
        raw.loc[raw["res_id"] == res_id]
        .sort_values("date")
        .reset_index(drop=True)
    )

    std_res = (
        std.loc[std["res_id"] == res_id]
        .sort_values("date")
        .reset_index(drop=True)
    )

    key_res = (
        key.loc[key["res_id"] == res_id]
        .sort_values("month")
        .reset_index(drop=True)
    )

    print("\nExample dataset")
    print("----------------")
    print(f"Reservoir ID:            {res_id}")
    print(f"Raw rows:                {len(raw_res)}")
    print(f"Standardized rows:       {len(std_res)}")
    print(f"Parameter rows:          {len(key_res)}")

    if not raw_res.empty:
        print(
            "Date range:              "
            f"{raw_res['date'].min().date()} to "
            f"{raw_res['date'].max().date()}"
        )

    if "release" in raw_res.columns:
        release_missing = int(
            raw_res["release"].isna().sum()
        )
        print(
            f"Missing release values:  {release_missing}"
        )

    if not meta.empty:
        meta_res = (
            meta.loc[meta["res_id"] == res_id]
            if "res_id" in meta.columns
            else meta
        )

        print(
            f"Metadata rows:            {len(meta_res)}"
        )


def print_result_summary(results: pd.DataFrame) -> None:
    print("\nModel summary")
    print("-------------")

    display_columns = [
        "method",
        "frequency",
        "r2",
        "n_observations",
        "status",
    ]

    summary_display = results[
        display_columns
    ].copy()

    summary_display["r2"] = summary_display["r2"].map(
        lambda value: (
            f"{value:.4f}"
            if pd.notna(value)
            else "NA"
        )
    )

    print(
        summary_display.to_string(index=False)
    )

    failed = results.loc[
        results["status"] == "failed"
    ]

    if not failed.empty:
        print("\nFailed runs")
        print("-----------")

        for _, row in failed.iterrows():
            print(
                f"{row['method']}, "
                f"frequency={row['frequency']}: "
                f"{row['error']}"
            )


# ============================================================
# 9. Main workflow
# ============================================================

def main() -> None:
    args = parse_arguments()

    invalid_frequencies = [
        frequency
        for frequency in args.frequencies
        if frequency < 1
    ]

    if invalid_frequencies:
        raise ValueError(
            "Initialization frequencies must be positive integers. "
            f"Invalid values: {invalid_frequencies}"
        )

    create_output_directories()
    validate_input_files()

    print("PLRT-Hybrid example")
    print("===================")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")

    meta, key, raw, std = load_example_data()

    meta, key, raw, std = prepare_example_data(
        meta=meta,
        key=key,
        raw=raw,
        std=std,
    )

    validate_required_columns(
        raw=raw,
        std=std,
        key=key,
    )

    validate_reservoir_data(
        res_id=args.res_id,
        raw=raw,
        std=std,
        key=key,
    )

    print_dataset_summary(
        res_id=args.res_id,
        meta=meta,
        raw=raw,
        std=std,
        key=key,
    )

    key_lookup = build_key_lookup(key)

    results = run_all_models(
        res_id=args.res_id,
        methods=args.methods,
        frequencies=args.frequencies,
        raw=raw,
        std=std,
        key_lookup=key_lookup,
        save_predictions=not args.skip_save_predictions,
    )

    summary_file = (
        SUMMARY_DIR
        / f"reservoir_{args.res_id}_performance_summary.csv"
    )

    results.to_csv(
        summary_file,
        index=False,
    )

    print_result_summary(results)

    successful_runs = int(
        (results["status"] == "success").sum()
    )
    total_runs = len(results)

    print("\nOutput")
    print("------")
    print(
        f"Summary file: "
        f"{summary_file.relative_to(PROJECT_ROOT)}"
    )

    if not args.skip_save_predictions:
        print(
            "Prediction directory: "
            f"{PREDICTION_DIR.relative_to(PROJECT_ROOT)}"
        )

    print(
        f"Successful runs: {successful_runs}/{total_runs}"
    )

    if successful_runs != total_runs:
        raise RuntimeError(
            f"{total_runs - successful_runs} model run(s) failed. "
            "See the summary above and the error column in the "
            "performance summary CSV."
        )

    print("\nAll requested model runs completed successfully.")


if __name__ == "__main__":
    main()

