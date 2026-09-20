"""Transferencia espacial mediante validación cruzada por estaciones disjuntas."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from Diagnosis_Algorithms import DEFAULT_CSV
from Performance_Diagnostics import resumen_metricas, incertidumbre_diferencias_modelos
from SpatioTemporal_Evaluation import construir_predicciones, metricas_desagregadas, validar_cohorte
from XGBoost_Algorithm import XGBoost_Algorithm


SCENARIOS = {
    "historial_local_disponible": (),
    "estacion_nueva_sin_dqo_historica": ("historical_dqo",),
}


def construir_folds_estaciones(frame, n_splits=5, random_state=42):
    """Distribuya estaciones completas equilibrando N dentro de cada zona."""
    eligible = frame.loc[frame.eligible & frame.y_true.notna()]
    station_rows = []
    for station, group in eligible.groupby("station", sort=False):
        zones = group.hydro_zone.dropna().astype(str)
        zone = zones.mode().iat[0] if len(zones) else "sin_dato"
        station_rows.append({"station": station, "N": len(group), "hydro_zone": zone})
    stations = pd.DataFrame(station_rows)
    if len(stations) < n_splits * 3:
        raise ValueError("Se requieren suficientes estaciones para train/validation/test disjuntos.")
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(n_splits)]
    fold_total = np.zeros(n_splits, dtype=int)
    for _, zone_group in stations.groupby("hydro_zone", sort=True):
        order = rng.permutation(len(zone_group))
        candidates = zone_group.iloc[order].sort_values("N", ascending=False, kind="stable")
        zone_total = np.zeros(n_splits, dtype=int)
        for row in candidates.itertuples(index=False):
            minimum = np.flatnonzero(zone_total == zone_total.min())
            chosen = minimum[np.argmin(fold_total[minimum])]
            folds[chosen].append(row.station)
            zone_total[chosen] += row.N
            fold_total[chosen] += row.N
    assignment = pd.concat([
        stations.loc[stations.station.isin(names)].assign(fold=f"F{i+1}")
        for i, names in enumerate(folds)
    ], ignore_index=True)
    if assignment.station.duplicated().any() or len(assignment) != len(stations):
        raise RuntimeError("La asignación de estaciones no es una partición válida.")
    return assignment


def _band_metrics(predictions):
    rows = []
    for (scenario, split, band), group in predictions.groupby(
        ["model", "split", "dqo_band"], observed=True, sort=True
    ):
        rows.append({"scenario": scenario, "split": split, "dqo_band": band,
                     **resumen_metricas(group.y_true, group.y_pred)})
    return pd.DataFrame(rows)


def ejecutar_leave_station_out(
    archivo_csv=DEFAULT_CSV, *, n_splits=5, random_state=42,
    coverage_threshold=.4, max_depth=5, verbose=True,
):
    """Evalúe XGBoost en estaciones nunca usadas para ajustar el fold.

    Es una prueba de transferencia espacial, no un pronóstico hacia años futuros:
    otros sitios de train pueden contener fechas contemporáneas. El segundo
    escenario elimina lags y tiempo desde la última DQO para representar una
    estación sin historia local del objetivo.
    """
    inventory = Data_Manage(
        archivo_csv, sequence_length=5, random_state=random_state,
        transformar_target_log=False,
    ).preparar_evaluacion(coverage_threshold=coverage_threshold)
    assignment = construir_folds_estaciones(inventory.frame, n_splits, random_state)
    fold_names = [f"F{i+1}" for i in range(n_splits)]
    blocks, audits, boundaries = [], [], []
    for fold_index, test_fold in enumerate(fold_names):
        validation_fold = fold_names[(fold_index + 1) % n_splits]
        test_stations = set(assignment.loc[assignment.fold.eq(test_fold), "station"])
        validation_stations = set(assignment.loc[assignment.fold.eq(validation_fold), "station"])
        for scenario, ablations in SCENARIOS.items():
            if verbose:
                print(f"{test_fold}/{scenario}", flush=True)
            data = Data_Manage(
                archivo_csv, sequence_length=5, random_state=random_state,
                transformar_target_log=False,
            ).preparar_evaluacion(
                coverage_threshold=coverage_threshold,
                validation_stations=validation_stations, test_stations=test_stations,
                ablate_groups=ablations,
            )
            estimator = XGBoost_Algorithm(random_state=random_state, max_depth=max_depth).fit(data)
            train_stations = set(data.partitions["train"].metadata.station)
            audits.append({
                "fold": test_fold, "scenario": scenario,
                "train_stations": len(train_stations),
                "validation_stations": len(validation_stations),
                "test_stations": len(test_stations),
                "train_validation_overlap": len(train_stations & validation_stations),
                "train_test_overlap": len(train_stations & test_stations),
                "validation_test_overlap": len(validation_stations & test_stations),
            })
            quantiles = np.quantile(data.partitions["train"].metadata.y_true, [.25, .5, .75])
            boundaries.extend({"fold": test_fold, "scenario": scenario, "quantile": q,
                               "train_dqo": value} for q, value in zip((.25, .5, .75), quantiles))
            labels = np.array(["baja", "media_baja", "media_alta", "alta"], dtype=object)
            for split in ("validation", "test"):
                part = data.partitions[split]
                prediction = np.maximum(data.inverse_target(estimator.predict(part)), 0)
                block = construir_predicciones(
                    part.metadata, part.metadata.y_true.to_numpy(), prediction,
                    scenario, split, fold=test_fold,
                )
                block["dqo_band"] = labels[np.searchsorted(
                    quantiles, block.y_true.to_numpy(), side="left"
                )]
                blocks.append(block)
    predictions = pd.concat(blocks, ignore_index=True)
    validar_cohorte(predictions, modelos_esperados=SCENARIOS)
    return {
        "assignment": assignment,
        "fold_audit": pd.DataFrame(audits),
        "predictions": predictions,
        "global": metricas_desagregadas(predictions)["global"],
        "dqo_band": _band_metrics(predictions),
        "dqo_band_boundaries": pd.DataFrame(boundaries),
        "uncertainty": incertidumbre_diferencias_modelos(
            predictions, n_bootstrap=2000, random_state=random_state
        ),
        "scope": "spatial transfer across unseen stations; not future-time forecasting",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: prueba leave-station-out de XGBoost.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    result = ejecutar_leave_station_out(args.csv, n_splits=args.folds)
    print("\nAUDITORÍA DE SOLAPAMIENTO")
    print(result["fold_audit"].to_string(index=False))
    print("\nTRANSFERENCIA ESPACIAL")
    print(result["global"].query("split == 'test'")[[
        "model", "N", "MAE", "RMSE", "R2", "bias"
    ]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nBANDAS DE DQO")
    print(result["dqo_band"].query("split == 'test'").to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
