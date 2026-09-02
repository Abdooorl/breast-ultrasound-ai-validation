from typing import Callable

import numpy as np
import pandas as pd


def patient_cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], dict],
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Patient-cluster bootstrap.

    Patients are sampled with replacement. Every time a patient is
    selected, all images belonging to that patient are included in
    the bootstrap sample.
    """

    if "patient_id" not in df.columns:
        raise ValueError("Dataframe must contain patient_id.")

    patient_ids = df["patient_id"].unique()

    if len(patient_ids) < 2:
        raise ValueError("At least two patients are required.")

    rng = np.random.default_rng(seed)

    patient_rows = {
        patient_id: df.index[
            df["patient_id"] == patient_id
        ].to_numpy()
        for patient_id in patient_ids
    }

    bootstrap_results: list[dict] = []

    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                patient_rows[patient_id]
                for patient_id in sampled_patients
            ]
        )

        sample_df = df.loc[sampled_indices].copy()

        try:
            metrics = metric_fn(sample_df)
        except ValueError:
            continue

        bootstrap_results.append(metrics)

    return {
        "n_bootstrap_requested": n_bootstrap,
        "n_bootstrap_valid": len(bootstrap_results),
        "seed": seed,
        "n_patients": int(len(patient_ids)),
        "samples": bootstrap_results,
    }


def percentile_ci(
    values: list[float],
    confidence_level: float = 0.95,
) -> dict:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]

    if len(values_array) == 0:
        return {
            "lower": None,
            "upper": None,
            "n_valid": 0,
        }

    alpha = 1.0 - confidence_level

    lower = np.quantile(
        values_array,
        alpha / 2,
    )

    upper = np.quantile(
        values_array,
        1 - alpha / 2,
    )

    return {
        "lower": float(lower),
        "upper": float(upper),
        "n_valid": int(len(values_array)),
    }