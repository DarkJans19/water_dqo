"""Evaluacion auditable de predicciones fuera de muestra de DQO.

La unidad de observacion es ``sample_id``. Los identificadores y metadatos se
conservan junto al objetivo y la prediccion, nunca se reconstruyen desde una
matriz escalada. Los DataFrames de salida usan DQO en su unidad original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_COLUMNS = (
    "sample_id", "station", "date", "latitude", "longitude", "elevation",
    "year", "month", "season", "season_source", "gap_days", "history_count",
    "target_censored", "hydro_zone", "hydro_subzone", "altitude_band",
)
GROUP_LEVELS = {
    "global": [],
    "station": ["station"],
    "year": ["year"],
    "season": ["season"],
    "year_season": ["year", "season"],
    "station_year": ["station", "year"],
    "station_season": ["station", "season"],
    "station_year_season": ["station", "year", "season"],
    "hydro_zone_year": ["hydro_zone", "year"],
    "hydro_subzone_year": ["hydro_subzone", "year"],
    "altitude_band_year": ["altitude_band", "year"],
}


def _vector(values, metadata: pd.DataFrame, name: str) -> np.ndarray:
    """Series require exact index alignment; arrays explicitly use row order."""
    if isinstance(values, (pd.Series, pd.DataFrame)):
        if not values.index.equals(metadata.index):
            raise ValueError(f"{name}: indice desalineado respecto a metadata.")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} debe ser numerico.") from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or len(array) != len(metadata):
        raise ValueError(f"{name}: se requiere un valor por fila de metadata.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contiene NaN o valores infinitos.")
    return array


def _validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(METADATA_COLUMNS) - set(metadata.columns))
    if missing:
        raise ValueError(f"Faltan columnas de metadata: {missing}.")
    if metadata.empty:
        raise ValueError("La cohorte de evaluacion esta vacia.")
    result = metadata.copy()
    for column in ("sample_id", "station"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contiene identificadores vacios.")
    result["sample_id"] = result["sample_id"].astype(str)
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError("date contiene fechas ausentes.")
    if result["target_censored"].isna().any() or not result["target_censored"].isin([True, False]).all():
        raise ValueError("target_censored debe contener valores booleanos sin ausencias.")
    result["target_censored"] = result["target_censored"].astype(bool)
    for column in ("latitude", "longitude", "elevation", "gap_days", "history_count"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        present = result[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(present).all():
            raise ValueError(f"{column} contiene valores infinitos.")
    for column, expected in (("year", result["date"].dt.year), ("month", result["date"].dt.month)):
        supplied = pd.to_numeric(result[column], errors="raise")
        if supplied.isna().any() or not np.array_equal(supplied.to_numpy(), expected.to_numpy()):
            raise ValueError(f"{column} no coincide con date.")
        result[column] = supplied.astype(int)
    return result


def construir_predicciones(
    metadata: pd.DataFrame,
    y_true,
    y_pred,
    model: str,
    split: str,
    fold: str = "holdout",
) -> pd.DataFrame:
    """Construya el registro largo de predicciones preservando todos los metadatos.

    ``y_true`` y ``y_pred`` deben estar en la escala original. Para Series o
    DataFrames se exige el mismo indice que ``metadata``; para arrays se usa el
    orden posicional. Un array de forma (n, 1), habitual en LSTM, es valido.
    """
    result = _validate_metadata(metadata)
    if result["sample_id"].duplicated().any():
        raise ValueError("sample_id debe ser unico dentro de cada modelo/split/fold.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model debe identificar el modelo entrenado.")
    if split not in ("validation", "test"):
        raise ValueError("split debe ser validation o test: evaluacion fuera de muestra.")
    if not isinstance(fold, str) or not fold.strip():
        raise ValueError("fold debe ser una cadena no vacia.")
    result["y_true"] = _vector(y_true, metadata, "y_true")
    result["y_pred"] = _vector(y_pred, metadata, "y_pred")
    result["model"] = model
    result["split"] = split
    result["fold"] = fold
    try:
        with np.errstate(over="raise", invalid="raise"):
            error = result["y_pred"].to_numpy() - result["y_true"].to_numpy()
            result["error"] = error
            result["absolute_error"] = np.abs(error)
            result["squared_error"] = np.square(error)
    except FloatingPointError as exc:
        raise ValueError("El calculo del error excede el rango numerico.") from exc
    return result


def _validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = set(METADATA_COLUMNS) | {"y_true", "y_pred", "model", "split", "fold"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Faltan columnas de predicciones: {missing}.")
    result = _validate_metadata(predictions)
    for column in ("model", "fold"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} contiene identificadores vacios.")
    if not result["split"].isin(["validation", "test"]).all():
        raise ValueError("Solo se admiten splits validation/test.")
    for column in ("y_true", "y_pred"):
        result[column] = _vector(result[column], result, column)
    if result.duplicated(["model", "split", "fold", "sample_id"]).any():
        raise ValueError("Predicciones duplicadas para model/split/fold/sample_id.")
    return result


def validar_cohorte(
    predictions: pd.DataFrame,
    modelos_esperados: Iterable[str] | None = None,
) -> None:
    """Exija las mismas muestras y verdad de referencia para todos los modelos.

    Compara claves ``(split, fold, sample_id)`` y todos los metadatos, incluso si
    las filas llegan en distinto orden. Una comparacion parcial entre modelos
    nunca queda disimulada por un promedio. Si solo se entrena un modelo, pasa.
    """
    frame = _validate_predictions(predictions)
    models = sorted(frame["model"].unique())
    if modelos_esperados is not None and set(models) != set(modelos_esperados):
        raise ValueError("Los modelos presentes no coinciden con modelos_esperados.")
    keys = ["split", "fold", "sample_id"]
    compare_columns = [column for column in METADATA_COLUMNS if column != "sample_id"] + ["y_true"]
    baseline = frame.loc[frame["model"].eq(models[0])].set_index(keys).sort_index()
    for model in models[1:]:
        candidate = frame.loc[frame["model"].eq(model)].set_index(keys).sort_index()
        if not baseline.index.equals(candidate.index):
            raise ValueError(f"Cohorte distinta entre {models[0]} y {model}.")
        try:
            pd.testing.assert_frame_equal(
                baseline[compare_columns], candidate[compare_columns],
                check_dtype=False, check_exact=True,
            )
        except AssertionError as exc:
            raise ValueError(f"Metadatos o y_true distintos entre {models[0]} y {model}.") from exc


def _group_metrics(group: pd.DataFrame, min_samples: int) -> dict:
    observed = group["y_true"].to_numpy(dtype=float)
    predicted = group["y_pred"].to_numpy(dtype=float)
    error = predicted - observed
    n = len(group)
    r2_defined = n >= 2 and not np.all(observed == observed[0])
    denominator = np.sum(np.square(observed - np.mean(observed)))
    r2 = 1.0 - np.sum(np.square(error)) / denominator if r2_defined else np.nan
    return {
        "N": n,
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "R2": float(r2),
        "R2_reported": float(r2) if n >= min_samples else np.nan,
        "bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "underestimation_pct": float(100.0 * np.mean(error < 0)),
        "overestimation_pct": float(100.0 * np.mean(error > 0)),
        "exact_pct": float(100.0 * np.mean(error == 0)),
        "insufficient_samples": bool(n < min_samples),
        "r2_defined": bool(r2_defined),
        "n_censored": int(group["target_censored"].sum()),
        "n_folds": int(group["fold"].nunique()),
    }


def metricas_desagregadas(
    predictions: pd.DataFrame,
    min_samples: int = 5,
) -> dict[str, pd.DataFrame]:
    """Calcule MAE/RMSE/R2 por territorio, calendario y sus intersecciones.

    Cada fila es una metrica calculada sobre observaciones, no un promedio de
    metricas de estaciones. Los folds se agregan por modelo/split y se informa
    ``n_folds``. R2 es NaN con un solo dato o un objetivo constante; los grupos
    pequenos se conservan con ``insufficient_samples``. ``dropna=False`` evita
    perder grupos cuya temporada no pudo asignarse. Las coordenadas reportadas
    son medianas y cualquier discrepancia observada queda marcada.
    """
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples debe ser un entero positivo.")
    frame = _validate_predictions(predictions)
    outputs = {}
    for name, dimensions in GROUP_LEVELS.items():
        group_columns = ["model", "split", *dimensions]
        rows = []
        for key, group in frame.groupby(group_columns, dropna=False, sort=True, observed=True):
            row = dict(zip(group_columns, key))
            row.update(_group_metrics(group, min_samples))
            row["season_sources"] = "|".join(sorted(group["season_source"].dropna().astype(str).unique()))
            row["season_interpretation"] = "calendar_label_not_observed_climate"
            row["period_start"] = group["date"].min()
            row["period_end"] = group["date"].max()
            row["n_years"] = int(group["year"].nunique())
            row["n_stations"] = int(group["station"].nunique())
            if "station" in dimensions:
                spatial_columns = ["latitude", "longitude", "elevation"]
                for column in spatial_columns:
                    values = group[column].dropna()
                    row[column] = float(values.median()) if not values.empty else np.nan
                row["spatial_metadata_conflict"] = bool(
                    any(group[column].dropna().nunique() > 1 for column in spatial_columns)
                )
            rows.append(row)
        outputs[name] = pd.DataFrame(rows)
    return outputs


def metricas_por_banda_dqo(
    predictions: pd.DataFrame,
    train_targets,
    min_samples: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúe por cuartiles cuyos límites se fijan exclusivamente con train.

    Devuelve la tabla de métricas y otra tabla auditable con los límites. Los
    límites repetidos se conservan: una banda puede quedar vacía si la DQO de
    entrenamiento tiene muchos empates, en vez de mirar test para corregirla.
    """
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples debe ser un entero positivo.")
    train = np.asarray(train_targets, dtype=float).reshape(-1)
    if not len(train) or not np.isfinite(train).all():
        raise ValueError("train_targets debe contener DQO de entrenamiento finita.")
    frame = _validate_predictions(predictions)
    quantiles = np.quantile(train, [0.25, 0.50, 0.75])
    labels = np.array(["baja", "media_baja", "media_alta", "alta"], dtype=object)
    # side='left' asigna un valor igual al cuantil a la banda inferior.
    frame["dqo_band"] = labels[np.searchsorted(quantiles, frame["y_true"].to_numpy(float), side="left")]
    rows = []
    for (model, split, band), group in frame.groupby(
        ["model", "split", "dqo_band"], sort=True, observed=True
    ):
        row = {"model": model, "split": split, "dqo_band": band}
        row.update(_group_metrics(group, min_samples))
        rows.append(row)
    boundaries = pd.DataFrame({
        "quantile": [0.25, 0.50, 0.75],
        "train_dqo": quantiles,
        "source": "training_only",
        "train_N": len(train),
    })
    return pd.DataFrame(rows), boundaries


def guardar_evaluacion(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    min_samples: int = 5,
) -> dict[str, pd.DataFrame]:
    """Valide la cohorte y exporte predicciones y tablas reproducibles.

    Los R2 no definidos se serializan como campos vacios del CSV. Los numeros
    conservan su precision; el formato visual se aplica solamente al graficar.
    """
    validar_cohorte(predictions)
    tables = metricas_desagregadas(predictions, min_samples=min_samples)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination / "predictions.csv", index=False, encoding="utf-8")
    for name, table in tables.items():
        table.to_csv(destination / f"metrics_{name}.csv", index=False, encoding="utf-8")
    return tables
