"""Experimento XGBoost ponderado para reducir subestimación de DQO alta."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from Diagnosis_Algorithms import DEFAULT_CSV
from Performance_Diagnostics import incertidumbre_diferencias_modelos
from SpatioTemporal_Evaluation import construir_predicciones, metricas_desagregadas, metricas_por_banda_dqo
from XGBoost_Algorithm import XGBoost_Algorithm


def ejecutar_experimento_picos(
    archivo_csv=DEFAULT_CSV, *, weights=(1.0, 2.0, 4.0, 8.0),
    max_global_mae_increase=.05, random_state=42, verbose=True,
):
    """Seleccione ponderación sólo con validación y evalúe una vez en test.

    Entre candidatos cuyo MAE global de validación no empeora más del límite,
    elige el menor MAE en la banda alta. La banda y los pesos usan Q3 de train.
    """
    if 1.0 not in weights or max_global_mae_increase < 0:
        raise ValueError("weights debe incluir 1.0 y la tolerancia debe ser no negativa.")
    data = Data_Manage(
        archivo_csv, sequence_length=5, random_state=random_state,
        transformar_target_log=False,
    ).preparar_evaluacion(
        train_end="2015-08-11", validation_end="2020-02-06",
        coverage_threshold=.4, censored_target_policy="exclude",
    )
    train_targets = data.partitions["train"].metadata.y_true.to_numpy()
    models, validation_rows = {}, []
    for weight in weights:
        name = f"peso_{weight:g}"
        if verbose:
            print(f"Entrenando {name}", flush=True)
        model = XGBoost_Algorithm(
            random_state=random_state, max_depth=5, high_dqo_weight=weight
        ).fit(data)
        models[name] = model
        part = data.partitions["validation"]
        prediction = np.maximum(data.inverse_target(model.predict(part)), 0)
        block = construir_predicciones(part.metadata, part.metadata.y_true, prediction, name, "validation")
        global_row = metricas_desagregadas(block)["global"].iloc[0]
        bands, _ = metricas_por_banda_dqo(block, train_targets)
        high = bands.loc[bands.dqo_band.eq("alta")].iloc[0]
        validation_rows.append({
            "candidate": name, "high_dqo_weight": weight,
            "global_MAE": global_row.MAE, "global_RMSE": global_row.RMSE,
            "global_R2": global_row.R2, "global_bias": global_row.bias,
            "high_N": high.N, "high_MAE": high.MAE, "high_RMSE": high.RMSE,
            "high_bias": high.bias, "high_underestimation_pct": high.underestimation_pct,
        })
    validation = pd.DataFrame(validation_rows)
    baseline_mae = validation.loc[validation.high_dqo_weight.eq(1), "global_MAE"].iat[0]
    validation["within_global_mae_constraint"] = validation.global_MAE <= baseline_mae * (1 + max_global_mae_increase)
    eligible = validation.loc[validation.within_global_mae_constraint]
    selected = eligible.sort_values(["high_MAE", "global_MAE"]).iloc[0].candidate
    validation["selected"] = validation.candidate.eq(selected)
    test_blocks = []
    for name in dict.fromkeys(["peso_1", selected]):
        part = data.partitions["test"]
        prediction = np.maximum(data.inverse_target(models[name].predict(part)), 0)
        test_blocks.append(construir_predicciones(
            part.metadata, part.metadata.y_true, prediction, name, "test"
        ))
    test_predictions = pd.concat(test_blocks, ignore_index=True)
    test_global = metricas_desagregadas(test_predictions)["global"]
    test_bands, boundaries = metricas_por_banda_dqo(test_predictions, train_targets)
    uncertainty = (incertidumbre_diferencias_modelos(test_predictions, random_state=random_state)
                   if selected != "peso_1" else pd.DataFrame())
    return {
        "selection_rule": f"min high-band validation MAE subject to global validation MAE <= baseline * {1+max_global_mae_increase:.3f}",
        "validation": validation, "selected": selected,
        "test_global": test_global, "test_dqo_band": test_bands,
        "dqo_band_boundaries": boundaries, "uncertainty": uncertainty,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: ponderación de picos seleccionada en validación.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    result = ejecutar_experimento_picos(args.csv)
    print("\nSELECCIÓN EN VALIDACIÓN")
    print(result["validation"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSELECCIONADO: {result['selected']}")
    print("\nPRUEBA GLOBAL")
    print(result["test_global"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPRUEBA POR BANDA")
    print(result["test_dqo_band"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
