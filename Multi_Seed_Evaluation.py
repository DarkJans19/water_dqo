"""Robustez de las configuraciones finales frente a varias semillas."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from Diagnosis_Algorithms import DEFAULT_CSV
from LSTM_Algorithm import LSTM_Algorithm
from Performance_Diagnostics import resumen_metricas
from SpatioTemporal_Evaluation import construir_predicciones, validar_cohorte
from XGBoost_Algorithm import XGBoost_Algorithm


DEFAULT_SEEDS = (7, 21, 42, 84, 123)
XGB_NAME = "XGBoost_selected_raw_cov40_depth5"
LSTM_NAME = "LSTM_selected_32_d03_physicalRMSE"


def _metrics_by_seed(predictions):
    rows = []
    for (model, split, fold), group in predictions.groupby(["model", "split", "fold"], sort=True):
        rows.append({"model": model, "split": split, "seed": int(fold.removeprefix("seed_")),
                     **resumen_metricas(group.y_true, group.y_pred)})
    return pd.DataFrame(rows)


def _high_band_by_seed(predictions, threshold):
    high = predictions.loc[predictions.y_true > threshold]
    rows = []
    for (model, split, fold), group in high.groupby(["model", "split", "fold"], sort=True):
        rows.append({"model": model, "split": split, "seed": int(fold.removeprefix("seed_")),
                     "train_Q3": threshold, **resumen_metricas(group.y_true, group.y_pred)})
    return pd.DataFrame(rows)


def ejecutar_multisemilla(
    archivo_csv=DEFAULT_CSV, *, seeds=DEFAULT_SEEDS, epochs=60, verbose=True,
):
    """Repita las configuraciones congeladas; no vuelva a seleccionar con test."""
    seeds = tuple(dict.fromkeys(int(seed) for seed in seeds))
    if len(seeds) < 3:
        raise ValueError("Use al menos tres semillas; cinco es el valor recomendado.")
    xgb_data = Data_Manage(
        archivo_csv, sequence_length=5, transformar_target_log=False
    ).preparar_evaluacion(
        train_end="2015-08-11", validation_end="2020-02-06",
        coverage_threshold=.4, censored_target_policy="exclude",
    )
    lstm_data = Data_Manage(
        archivo_csv, sequence_length=5, transformar_target_log=True
    ).preparar_evaluacion(
        train_end="2015-08-11", validation_end="2020-02-06",
        coverage_threshold=.7, censored_target_policy="exclude",
    )
    for split in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(
            xgb_data.partitions[split].metadata, lstm_data.partitions[split].metadata
        )
    blocks, training = [], []
    for seed in seeds:
        if verbose:
            print(f"Semilla {seed}: XGBoost", flush=True)
        models = (
            (XGB_NAME, XGBoost_Algorithm(random_state=seed, max_depth=5), xgb_data),
            (LSTM_NAME, LSTM_Algorithm(
                random_state=seed, units=32, dropout=.3,
                monitor_original_rmse=True, epochs=epochs,
            ), lstm_data),
        )
        for model_name, estimator, data in models:
            if verbose and model_name == LSTM_NAME:
                print(f"Semilla {seed}: LSTM", flush=True)
            estimator.fit(data, inverse_target=data.inverse_target)
            training.append({"model": model_name, "seed": seed,
                             "best_epoch": estimator.training_report.get("best_epoch"),
                             "epochs_run": estimator.training_report.get("epochs_run"),
                             "deterministic_ops": estimator.training_report.get("deterministic_ops_enabled")})
            for split in ("validation", "test"):
                part = data.partitions[split]
                prediction = np.maximum(data.inverse_target(estimator.predict(part)), 0)
                blocks.append(construir_predicciones(
                    part.metadata, part.metadata.y_true, prediction,
                    model_name, split, fold=f"seed_{seed}",
                ))
    predictions = pd.concat(blocks, ignore_index=True)
    validar_cohorte(predictions, modelos_esperados=[XGB_NAME, LSTM_NAME])
    metrics = _metrics_by_seed(predictions)
    threshold = float(np.quantile(xgb_data.partitions["train"].metadata.y_true, .75))
    high = _high_band_by_seed(predictions, threshold)
    test = metrics.loc[metrics.split.eq("test")].copy()
    test["rank_MAE"] = test.groupby("seed")["MAE"].rank(method="min")
    summary = test.groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"), mean_MAE=("MAE", "mean"), sd_MAE=("MAE", "std"),
        min_MAE=("MAE", "min"), max_MAE=("MAE", "max"),
        mean_RMSE=("RMSE", "mean"), sd_RMSE=("RMSE", "std"),
        mean_R2=("R2", "mean"), sd_R2=("R2", "std"),
        MAE_wins=("rank_MAE", lambda values: int((values == 1).sum())),
    )
    high_test = high.loc[high.split.eq("test")]
    high_summary = high_test.groupby("model", as_index=False).agg(
        mean_high_MAE=("MAE", "mean"), sd_high_MAE=("MAE", "std"),
        mean_high_bias=("bias", "mean"), sd_high_bias=("bias", "std"),
        mean_high_underestimation_pct=("underestimation_pct", "mean"),
        high_bias_negative_all_seeds=("bias", lambda values: bool((values < 0).all())),
    )
    prediction_variability = (predictions.loc[predictions.split.eq("test")]
        .groupby(["model", "sample_id"], as_index=False)
        .agg(prediction_mean=("y_pred", "mean"), prediction_sd=("y_pred", "std")))
    variability_summary = prediction_variability.groupby("model", as_index=False).agg(
        median_prediction_sd=("prediction_sd", "median"),
        p95_prediction_sd=("prediction_sd", lambda values: float(values.quantile(.95))),
        max_prediction_sd=("prediction_sd", "max"),
    )
    return {"per_seed": metrics, "high_dqo_per_seed": high, "summary": summary,
            "high_dqo_summary": high_summary,
            "prediction_variability": prediction_variability,
            "variability_summary": variability_summary,
            "training": pd.DataFrame(training), "predictions": predictions,
            "seeds": seeds, "protocol": "frozen selected configurations; same temporal cohort"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: estabilidad multisemilla de modelos finales.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args(argv)
    result = ejecutar_multisemilla(args.csv, seeds=args.seeds, epochs=args.epochs)
    print("\nRESULTADOS POR SEMILLA")
    print(result["per_seed"].query("split == 'test'").to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    print("\nRESUMEN")
    print(result["summary"].to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    print("\nVARIACIÓN DE PREDICCIONES")
    print(result["variability_summary"].to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    print("\nROBUSTEZ EN DQO ALTA")
    print(result["high_dqo_summary"].to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
