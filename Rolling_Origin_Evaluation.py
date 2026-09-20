"""Evaluación con varios orígenes y horizontes temporales no solapados."""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from Diagnosis_Algorithms import DEFAULT_CSV, ejecutar_evaluacion


DEFAULT_ORIGINS = (
    ("O1", "2014-12-31", "2016-12-31", "2018-12-31"),
    ("O2", "2016-12-31", "2018-12-31", "2020-12-31"),
    ("O3", "2018-12-31", "2020-12-31", "2022-12-31"),
    ("O4", "2020-12-31", "2022-12-31", "2024-12-31"),
)


def _validar_origenes(origins):
    rows = []
    previous_test_end = None
    for name, train_end, validation_end, test_end in origins:
        train_end, validation_end, test_end = map(pd.Timestamp, (train_end, validation_end, test_end))
        if not train_end < validation_end < test_end:
            raise ValueError(f"{name}: se requiere train_end < validation_end < test_end.")
        test_start = validation_end + pd.Timedelta(days=1)
        if previous_test_end is not None and test_start <= previous_test_end:
            raise ValueError("Las ventanas de prueba de los orígenes no deben solaparse.")
        rows.append((str(name), train_end, validation_end, test_end))
        previous_test_end = test_end
    if len(rows) < 2:
        raise ValueError("Se requieren al menos dos orígenes temporales.")
    return rows


def resumir_estabilidad(global_metrics, band_metrics):
    test_global = global_metrics.query("split == 'test'").copy()
    test_bands = band_metrics.query("split == 'test'").copy()
    test_global["rank_MAE"] = test_global.groupby("origin")["MAE"].rank(method="min")
    high = test_bands.loc[test_bands.dqo_band.eq("alta")]
    rows = []
    for model, group in test_global.groupby("model", sort=True):
        high_model = high.loc[high.model.eq(model)]
        rows.append({
            "model": model,
            "origins": int(group.origin.nunique()),
            "mean_MAE": float(group.MAE.mean()),
            "sd_MAE_between_origins": float(group.MAE.std(ddof=1)),
            "mean_RMSE": float(group.RMSE.mean()),
            "mean_R2": float(group.R2.mean()),
            "MAE_wins": int(group.rank_MAE.eq(1).sum()),
            "high_DQO_mean_bias": float(high_model.bias.mean()),
            "high_DQO_underestimated_in_all_origins": bool((high_model.bias < 0).all()),
            "high_DQO_mean_underestimation_pct": float(high_model.underestimation_pct.mean()),
        })
    return pd.DataFrame(rows)


def ejecutar_origenes_temporales(
    archivo_csv=DEFAULT_CSV, *, origins=DEFAULT_ORIGINS,
    models=("XGBoost", "LSTM", "SVM"), random_state=42,
    search_xgboost=True, regularize_lstm=True, epochs=60,
    censored_target_policy="exclude", verbose=True,
):
    """Entrene de nuevo en cada origen y devuelva tablas combinadas.

    Cada prueba dura dos años y no se solapa con las demás. XGBoost puede hacer
    su selección interna sólo con la validación correspondiente a cada origen.
    Las configuraciones no se reajustan mirando ninguna de las pruebas.
    """
    origins = _validar_origenes(origins)
    global_blocks, band_blocks, boundary_blocks, uncertainty_blocks = [], [], [], []
    split_rows = []
    for name, train_end, validation_end, test_end in origins:
        if verbose:
            print(f"{name}: train <= {train_end.date()}, valid <= {validation_end.date()}, test <= {test_end.date()}", flush=True)
        result = ejecutar_evaluacion(
            archivo_csv=archivo_csv, models=models,
            train_end=train_end, validation_end=validation_end, test_end=test_end,
            random_state=random_state, epochs=epochs,
            censored_target_policy=censored_target_policy,
            improve_xgboost=search_xgboost and "XGBoost" in models,
            regularize_lstm=regularize_lstm,
            no_plots=True, no_error_analysis=True, verbose=False,
        )
        for source, destination in (
            (result["metrics"]["global"], global_blocks),
            (result["metrics"]["dqo_band"], band_blocks),
            (result["dqo_band_boundaries"], boundary_blocks),
            (result["diagnostics"]["model_difference_uncertainty"], uncertainty_blocks),
        ):
            block = source.copy()
            block.insert(0, "origin", name)
            destination.append(block)
        split_rows.append({"origin": name, **result["split_config"],
                           **{f"{split}_N": len(part.y) for split, part in result["prepared"].partitions.items()}})
    global_metrics = pd.concat(global_blocks, ignore_index=True)
    band_metrics = pd.concat(band_blocks, ignore_index=True)
    return {
        "origins": pd.DataFrame(split_rows),
        "global": global_metrics,
        "dqo_band": band_metrics,
        "dqo_band_boundaries": pd.concat(boundary_blocks, ignore_index=True),
        "model_difference_uncertainty": pd.concat(uncertainty_blocks, ignore_index=True),
        "stability": resumir_estabilidad(global_metrics, band_metrics),
        "protocol": "expanding training; consecutive validation/test windows; non-overlapping two-year tests",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: evaluación de estabilidad con cuatro orígenes temporales.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--models", nargs="+", default=["XGBoost", "LSTM", "SVM"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-xgboost-search", action="store_true")
    args = parser.parse_args(argv)
    result = ejecutar_origenes_temporales(
        args.csv, models=args.models, epochs=args.epochs,
        search_xgboost=not args.no_xgboost_search,
    )
    print("\nRESULTADOS POR ORIGEN")
    print(result["global"].query("split == 'test'")[
        ["origin", "model", "N", "MAE", "RMSE", "R2", "bias"]
    ].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nESTABILIDAD")
    print(result["stability"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
