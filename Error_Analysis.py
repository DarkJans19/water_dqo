"""Diagnóstico asociativo del error fuera de muestra, sin ajustar sobre prueba.

SHAP explica un modelo auxiliar del error absoluto, no el modelo de DQO ni
relaciones causales. El modelo auxiliar aprende exclusivamente de validación.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ERROR_FEATURES = (
    "elevation", "latitude", "longitude", "gap_days", "history_count",
    "month_sin", "month_cos",
)
MIN_VALIDATION_SAMPLES = 20
MIN_TEST_SAMPLES = 5


def _features(rows: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=rows.index)
    for name in ERROR_FEATURES[:5]:
        values = rows[name] if name in rows else pd.Series(np.nan, index=rows.index)
        features[name] = pd.to_numeric(values, errors="coerce")
    if "month" in rows:
        month = pd.to_numeric(rows["month"], errors="coerce")
    else:
        month = pd.to_datetime(rows["date"], errors="coerce").dt.month
    month = month.where(month.between(1, 12))
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    return features.replace([np.inf, -np.inf], np.nan)


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "model"


def _export_descriptive(rows: pd.DataFrame, directory: Path, save) -> list[str]:
    """Conservar observaciones y asociaciones de rango sin interpretación causal."""
    features = _features(rows)
    metadata = [name for name in (
        "sample_id", "model", "split", "fold", "station", "date", "year", "month",
        "season", "season_source", "absolute_error", "error",
    ) if name in rows]
    detail = pd.concat([rows[metadata], features], axis=1)
    detail_path = directory / "error_vs_factors.csv"
    save(detail, detail_path)
    associations = []
    bins = []
    for (model, split), group in detail.groupby(["model", "split"], dropna=False):
        for feature in ERROR_FEATURES:
            paired = group[[feature, "absolute_error"]].dropna()
            correlation = np.nan
            if len(paired) >= 3 and paired[feature].nunique() > 1 and paired["absolute_error"].nunique() > 1:
                correlation = paired[feature].rank().corr(paired["absolute_error"].rank())
            associations.append({
                "model": model, "split": split, "factor": feature,
                "N": len(paired), "spearman_error": correlation,
            })
            if len(paired) and paired[feature].nunique() > 1:
                labels = pd.qcut(paired[feature], q=min(4, paired[feature].nunique()), duplicates="drop")
                for label, subset in paired.groupby(labels, observed=True):
                    bins.append({
                        "model": model, "split": split, "factor": feature,
                        "factor_bin": str(label), "N": len(subset),
                        "factor_mean": subset[feature].mean(),
                        "MAE": subset["absolute_error"].mean(),
                        "median_absolute_error": subset["absolute_error"].median(),
                    })
    correlation_path = directory / "factor_error_associations.csv"
    save(pd.DataFrame(associations, columns=["model", "split", "factor", "N", "spearman_error"]), correlation_path)
    bin_path = directory / "factor_error_bins.csv"
    save(pd.DataFrame(bins, columns=["model", "split", "factor", "factor_bin", "N", "factor_mean", "MAE", "median_absolute_error"]), bin_path)
    return [str(detail_path), str(correlation_path), str(bin_path)]


def _shap_explanations(model, x_test, test_rows, model_name, directory, random_state, max_samples, save):
    import shap  # Dependencia opcional: el llamador registra el fallo y usa permutación.

    sample = x_test.sample(n=min(max_samples, len(x_test)), random_state=random_state)
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample, check_additivity=True))
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != sample.shape or not np.isfinite(values).all():
        raise ValueError(f"SHAP devolvió una matriz inválida: {values.shape}")
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    reconstructed = expected + values.sum(axis=1)
    if not np.allclose(reconstructed, model.predict(sample), rtol=1e-4, atol=1e-6):
        raise ValueError("No se cumple la aditividad SHAP del predictor de error absoluto")
    metadata_columns = [c for c in ("sample_id", "station", "date", "year", "season", "split", "fold") if c in test_rows]
    metadata = test_rows.loc[sample.index, metadata_columns].reset_index(drop=True)
    long_frames = []
    for position, feature in enumerate(sample.columns):
        frame = metadata.copy()
        frame["model"] = model_name
        frame["feature"] = feature
        frame["feature_value_imputed"] = sample[feature].to_numpy()
        frame["shap_value"] = values[:, position]
        frame["absolute_shap_value"] = np.abs(values[:, position])
        frame["expected_absolute_error"] = expected
        frame["surrogate_prediction_absolute_error"] = reconstructed
        long_frames.append(frame)
    long = pd.concat(long_frames, ignore_index=True)
    path = directory / "shap_error_samples.csv"
    save(long, path)
    paths = [str(path)]
    coverage = []
    for grouping in ("station", "year", "season"):
        if grouping not in test_rows:
            continue
        total = test_rows.groupby(grouping, dropna=False).size().rename("test_N")
        explained = test_rows.loc[sample.index].groupby(grouping, dropna=False).size().rename("explained_N")
        counts = pd.concat([total, explained], axis=1).fillna({"explained_N": 0}).reset_index()
        for record in counts.to_dict("records"):
            coverage.append({
                "model": model_name, "grouping": grouping, "group": record[grouping],
                "test_N": int(record["test_N"]), "explained_N": int(record["explained_N"]),
            })
    coverage_path = directory / "shap_group_coverage.csv"
    save(pd.DataFrame(coverage, columns=["model", "grouping", "group", "test_N", "explained_N"]), coverage_path)
    paths.append(str(coverage_path))
    for name, groups in (
        ("global", ["model", "feature"]),
        ("station", ["model", "station", "feature"]),
        ("year", ["model", "year", "feature"]),
        ("season", ["model", "season", "feature"]),
    ):
        if not set(groups).issubset(long.columns):
            continue
        summary = long.groupby(groups, dropna=False).agg(
            N=("shap_value", "size"),
            mean_abs_shap=("absolute_shap_value", "mean"),
            mean_signed_shap=("shap_value", "mean"),
        ).reset_index()
        path = directory / f"shap_error_{name}.csv"
        save(summary, path)
        paths.append(str(path))
    return {"method": "SHAP_TreeExplainer", "explained_samples": len(sample), "files": paths}


def analizar_error(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    max_samples: int = 300,
    export_csv: bool = False,
) -> dict:
    """Explicar variación espacial/temporal del error mediante un surrogate.

    Se ajustan RF e imputación sobre errores de validación exclusivamente. La
    prueba solo evalúa el surrogate y proporciona las filas que se explican.
    El baseline es la mediana del error absoluto de validación. Si SHAP no está
    disponible, la importancia por permutación mide incremento de MAE del
    surrogate en prueba; no equivale a atribuciones SHAP locales.
    """
    if max_samples < 1:
        raise ValueError("max_samples debe ser positivo")
    required = {"sample_id", "model", "split", "date", "absolute_error"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Faltan columnas para analizar el error: {sorted(missing)}")
    directory = Path(output_dir)
    tables = {}

    def save(table, path):
        tables[str(path.relative_to(directory).with_suffix(''))] = table.copy()
        if export_csv:
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)

    rows = predictions.copy().reset_index(drop=True)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    rows["absolute_error"] = pd.to_numeric(rows["absolute_error"], errors="coerce")
    valid_errors = np.isfinite(rows["absolute_error"]) & rows["absolute_error"].ge(0)
    invalid_count = int((~valid_errors).sum())
    rows = rows.loc[valid_errors].copy()
    report = {
        "purpose": "Diagnóstico asociativo post hoc del error absoluto; no identifica causas ni explica directamente el predictor de DQO.",
        "fit_split": "validation", "evaluation_split": "test",
        "features": list(ERROR_FEATURES),
        "minimum_validation_samples": MIN_VALIDATION_SAMPLES,
        "minimum_test_samples": MIN_TEST_SAMPLES,
        "max_explained_samples_per_model": int(max_samples),
        "invalid_error_rows_excluded": invalid_count,
        "caveats": [
            "La importancia depende de la capacidad predictiva del surrogate y de factores correlacionados.",
            "Las atribuciones de grupos pequeños son descriptivas, sin inferencia causal ni intervalos de confianza.",
            "SHAP usa una muestra aleatoria reproducible de prueba; shap_group_coverage.csv identifica grupos sin representación.",
            "Un surrogate con R² no positivo o sin mejora frente al baseline se marca low_predictive_skill; sus importancias no son evidencia predictiva sólida.",
            "La estacionalidad depende del calendario declarado; no demuestra un régimen climático local.",
            "gap_days es tiempo entre registros observados: no prueba ausencia de monitoreo ni distingue fallos de reporte.",
        ],
        "files": _export_descriptive(rows, directory, save),
        "models": {},
    }
    for number, (model_name, model_rows) in enumerate(rows.groupby("model", sort=True)):
        model_name = str(model_name)
        train = model_rows.loc[model_rows["split"].eq("validation")].copy()
        test = model_rows.loc[model_rows["split"].eq("test")].copy()
        initial_train_count = len(train)
        # Protección frente a predicciones de múltiples folds con IDs repetidos
        # o validaciones futuras respecto a la prueba más temprana.
        train = train.loc[~train["sample_id"].isin(test["sample_id"])]
        earliest_test = test["date"].min()
        if pd.notna(earliest_test):
            train = train.loc[train["date"].lt(earliest_test)]
        result = {
            "validation_samples": len(train), "test_samples": len(test),
            "validation_rows_excluded_overlap_or_future": initial_train_count - len(train),
            "status": "insufficient_samples",
        }
        report["models"][model_name] = result
        if len(train) < MIN_VALIDATION_SAMPLES or len(test) < MIN_TEST_SAMPLES:
            result["reason"] = "Se requieren al menos 20 residuos de validación anteriores a prueba y 5 residuos de prueba."
            continue
        x_train = _features(train)
        x_test = _features(test)
        available = x_train.columns[x_train.notna().any()].tolist()
        result["unavailable_features"] = [c for c in ERROR_FEATURES if c not in available]
        if not available:
            result.update(status="no_features", reason="No hay factores observados en validación.")
            continue
        medians = x_train[available].median()
        x_train = x_train[available].fillna(medians)
        x_test = x_test[available].fillna(medians)
        result["imputation_medians_validation"] = {c: float(value) for c, value in medians.items()}
        surrogate = RandomForestRegressor(
            n_estimators=160, min_samples_leaf=max(2, min(5, len(train) // 10)),
            random_state=random_state, n_jobs=1,
        )
        surrogate.fit(x_train, train["absolute_error"])
        estimate = surrogate.predict(x_test)
        baseline_value = float(train["absolute_error"].median())
        baseline_mae = float(mean_absolute_error(test["absolute_error"], np.full(len(test), baseline_value)))
        surrogate_mae = float(mean_absolute_error(test["absolute_error"], estimate))
        skill = 1 - surrogate_mae / baseline_mae if baseline_mae > 0 else np.nan
        surrogate_r2 = r2_score(test["absolute_error"], estimate) if test["absolute_error"].nunique() > 1 else np.nan
        result.update(
            status="ok" if np.isfinite(skill) and skill > 0 and np.isfinite(surrogate_r2) and surrogate_r2 > 0 else "low_predictive_skill",
            surrogate_test_MAE=surrogate_mae,
            surrogate_test_RMSE=float(np.sqrt(mean_squared_error(test["absolute_error"], estimate))),
            surrogate_test_R2=_finite_number(surrogate_r2),
            baseline_absolute_error_median_validation=baseline_value,
            baseline_test_MAE=baseline_mae,
            MAE_skill_against_validation_median=_finite_number(skill),
        )
        model_dir = directory / f"{number:02d}_{_safe_name(model_name)}"
        diagnostic_columns = [c for c in ("sample_id", "station", "date", "year", "season", "split", "fold", "absolute_error") if c in test]
        diagnostic = test[diagnostic_columns].copy()
        diagnostic["model"] = model_name
        diagnostic["surrogate_predicted_absolute_error"] = estimate
        diagnostic["baseline_absolute_error"] = baseline_value
        diagnostic_path = model_dir / "surrogate_test_predictions.csv"
        save(diagnostic, diagnostic_path)
        result["files"] = [str(diagnostic_path)]
        try:
            explanation = _shap_explanations(
                surrogate, x_test, test, model_name, model_dir, random_state, max_samples, save,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"{model_name}: SHAP no disponible/incompatible ({reason}); se usa importancia por permutación en prueba.",
                RuntimeWarning, stacklevel=2,
            )
            sample = x_test.sample(n=min(max_samples, len(x_test)), random_state=random_state)
            importance = permutation_importance(
                surrogate, sample, test.loc[sample.index, "absolute_error"],
                scoring="neg_mean_absolute_error", n_repeats=10,
                random_state=random_state, n_jobs=1,
            )
            table = pd.DataFrame({
                "model": model_name, "feature": available,
                "MAE_increase_mean": importance.importances_mean,
                "MAE_increase_std": importance.importances_std,
                "N": len(sample), "method": "permutation_test_surrogate",
            }).sort_values("MAE_increase_mean", ascending=False)
            path = model_dir / "permutation_error_global.csv"
            save(table, path)
            explanation = {
                "method": "permutation_test_surrogate", "shap_failure": reason,
                "explained_samples": len(sample), "files": [str(path)],
                "limitation": "Importancia global del surrogate en prueba; no proporciona atribuciones locales ni SHAP por estación/año/temporada.",
            }
        result["explanation"] = explanation
    report["tables"] = tables
    if not export_csv:
        report["files"] = []
        for result in report["models"].values():
            result["files"] = []
            if "explanation" in result:
                result["explanation"]["files"] = []
    return report
