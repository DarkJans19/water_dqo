"""Protocolo prospectivo para datos posteriores al período ya inspeccionado."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from LSTM_Algorithm import LSTM_Algorithm
from Performance_Diagnostics import resumen_metricas
from SpatioTemporal_Evaluation import construir_predicciones, metricas_por_banda_dqo, validar_cohorte
from XGBoost_Algorithm import XGBoost_Algorithm


DEVELOPMENT_DATA_END = pd.Timestamp("2024-11-23")


def evaluar_confirmacion_futura(archivo_actualizado, *, external_start,
                                random_state=42, epochs=60, verbose=True):
    """Evalúe configuraciones congeladas únicamente después de external_start.

    ``archivo_actualizado`` debe conservar el esquema IDEAM e incluir el historial
    antiguo y las observaciones nuevas. El intervalo 2020-02-07..external_start-1
    queda fuera de las métricas: ya participó en decisiones de desarrollo.
    """
    external_start = pd.Timestamp(external_start).normalize()
    if external_start <= DEVELOPMENT_DATA_END:
        raise ValueError(f"La confirmación externa debe comenzar después de {DEVELOPMENT_DATA_END.date()}.")
    common = dict(train_end="2015-08-11", validation_end="2020-02-06",
                  test_start=external_start, censored_target_policy="exclude")
    xgb_data = Data_Manage(archivo_actualizado, sequence_length=5, transformar_target_log=False).preparar_evaluacion(
        coverage_threshold=.4, **common)
    lstm_data = Data_Manage(archivo_actualizado, sequence_length=5, transformar_target_log=True).preparar_evaluacion(
        coverage_threshold=.7, **common)
    for split in ("validation", "test"):
        pd.testing.assert_frame_equal(xgb_data.partitions[split].metadata,
                                      lstm_data.partitions[split].metadata)
    configurations = (
        ("XGBoost_selected_raw_cov40_depth5_frozen", XGBoost_Algorithm(
            random_state=random_state, max_depth=5), xgb_data),
        ("LSTM_selected_32_d03_physicalRMSE_frozen", LSTM_Algorithm(
            random_state=random_state, units=32, dropout=.3,
            monitor_original_rmse=True, epochs=epochs), lstm_data),
    )
    rows, blocks = [], []
    for name, estimator, data in configurations:
        if verbose:
            print(f"Ajustando configuración congelada: {name}", flush=True)
        estimator.fit(data, inverse_target=data.inverse_target)
        part = data.partitions["test"]
        prediction = np.maximum(data.inverse_target(estimator.predict(part)), 0)
        rows.append({"model": name, "external_start": external_start,
                     "external_end": part.metadata.date.max(),
                     **resumen_metricas(part.metadata.y_true, prediction)})
        blocks.append(construir_predicciones(part.metadata, part.metadata.y_true,
                                              prediction, name, "test", fold="future_confirmation"))
    predictions = pd.concat(blocks, ignore_index=True)
    validar_cohorte(predictions)
    bands, boundaries = metricas_por_banda_dqo(
        predictions, xgb_data.partitions["train"].metadata.y_true
    )
    path = Path(archivo_actualizado)
    return {"global": pd.DataFrame(rows), "dqo_band": bands,
            "dqo_band_boundaries": boundaries, "predictions": predictions,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "protocol": "configurations frozen from development; only post-2024 observations scored",
            "warning": "Run once for confirmatory reporting; later tuning makes this dataset developmental."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: confirmación verdaderamente futura.")
    parser.add_argument("--csv", required=True, help="CSV actualizado con historial y datos nuevos.")
    parser.add_argument("--external-start", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args(argv)
    result = evaluar_confirmacion_futura(args.csv, external_start=args.external_start, epochs=args.epochs)
    print(result["global"].to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    print("\n", result["warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
