"""Ablaciones predefinidas de grupos de predictores para XGBoost."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from Diagnosis_Algorithms import DEFAULT_CSV
from Performance_Diagnostics import incertidumbre_diferencias_modelos
from SpatioTemporal_Evaluation import (construir_predicciones,
    metricas_desagregadas, metricas_por_banda_dqo, validar_cohorte)
from XGBoost_Algorithm import XGBoost_Algorithm


ABLATIONS = {
    "baseline": (),
    "sin_dqo_historica": ("historical_dqo",),
    "sin_geografia": ("geography",),
    "sin_tiempo": ("time",),
    "sin_covariables_contemporaneas": ("contemporary_covariates",),
}


def ejecutar_ablaciones(
    archivo_csv=DEFAULT_CSV, *, train_end="2015-08-11",
    validation_end="2020-02-06", test_end=None, random_state=42,
    coverage_threshold=.4, max_depth=5, verbose=True,
):
    """Compare grupos retirados sin cambiar hiperparámetros entre variantes.

    La configuración corresponde al XGBoost ya seleccionado. Validación es la
    referencia primaria para interpretar una ablación; prueba se conserva como
    evaluación secundaria y nunca se usa para reajustar la variante.
    """
    blocks, feature_rows = [], []
    reference = None
    train_targets = None
    for variant, removed in ABLATIONS.items():
        if verbose:
            print(f"Entrenando {variant}: elimina {removed or 'ningún grupo'}", flush=True)
        data = Data_Manage(
            archivo_csv, sequence_length=5, random_state=random_state,
            transformar_target_log=False,
        ).preparar_evaluacion(
            train_end=train_end, validation_end=validation_end, test_end=test_end,
            coverage_threshold=coverage_threshold, censored_target_policy="exclude",
            ablate_groups=removed,
        )
        if reference is None:
            reference = data
            train_targets = data.partitions["train"].metadata.y_true.to_numpy()
        else:
            for split in ("train", "validation", "test"):
                pd.testing.assert_frame_equal(
                    data.partitions[split].metadata,
                    reference.partitions[split].metadata,
                )
        estimator = XGBoost_Algorithm(random_state=random_state, max_depth=max_depth).fit(data)
        feature_rows.append({"variant": variant, "removed_groups": "|".join(removed),
                             "feature_count": len(data.feature_cols)})
        for split in ("validation", "test"):
            part = data.partitions[split]
            prediction = np.maximum(data.inverse_target(estimator.predict(part)), 0)
            blocks.append(construir_predicciones(
                part.metadata, part.metadata.y_true.to_numpy(), prediction, variant, split
            ))
    predictions = pd.concat(blocks, ignore_index=True)
    validar_cohorte(predictions, modelos_esperados=ABLATIONS)
    global_metrics = metricas_desagregadas(predictions)["global"]
    bands, boundaries = metricas_por_banda_dqo(predictions, train_targets)
    baseline = global_metrics.loc[global_metrics.model.eq("baseline"),
                                  ["split", "MAE", "RMSE", "R2"]].set_index("split")
    comparison = global_metrics.copy()
    for metric in ("MAE", "RMSE", "R2"):
        comparison[f"delta_{metric}_vs_baseline"] = comparison.apply(
            lambda row: row[metric] - baseline.at[row["split"], metric], axis=1
        )
    return {
        "features": pd.DataFrame(feature_rows),
        "predictions": predictions,
        "global": comparison,
        "dqo_band": bands,
        "dqo_band_boundaries": boundaries,
        "uncertainty": incertidumbre_diferencias_modelos(
            predictions, n_bootstrap=2000, random_state=random_state
        ),
        "interpretation_rule": "delta MAE/RMSE > 0 empeora al retirar el grupo; priorizar validación",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: ablaciones controladas de XGBoost.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    result = ejecutar_ablaciones(args.csv)
    print("\nPREDICTORES")
    print(result["features"].to_string(index=False))
    print("\nABLACIONES")
    print(result["global"][["model", "split", "N", "MAE", "RMSE", "R2",
        "delta_MAE_vs_baseline", "delta_RMSE_vs_baseline", "delta_R2_vs_baseline"
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
