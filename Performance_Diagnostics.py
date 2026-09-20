"""Diagnóstico descriptivo reproducible; ninguna selección se hace sobre test."""
import numpy as np
import pandas as pd
from itertools import combinations


def resumen_metricas(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    residual = prediction-y
    denominator = np.square(y-y.mean()).sum()
    return dict(N=len(y), MAE=float(np.abs(residual).mean()),
                RMSE=float(np.sqrt(np.square(residual).mean())),
                R2=float(1-np.square(residual).sum()/denominator) if len(y)>1 and denominator>0 else np.nan,
                bias=float(residual.mean()),
                median_absolute_error=float(np.median(np.abs(residual))),
                underestimation_pct=float(100*np.mean(residual < 0)),
                overestimation_pct=float(100*np.mean(residual > 0)))


def resumen_macro_estaciones(station_metrics: pd.DataFrame) -> pd.DataFrame:
    """Contraste entre agregación por observaciones y por estaciones.

    El resumen macro usa sólo estaciones con evidencia mínima según la bandera
    ya calculada, e informa explícitamente cuántas observaciones quedan fuera.
    """
    required = {"model", "split", "station", "N", "MAE", "RMSE", "bias", "insufficient_samples"}
    missing = required - set(station_metrics)
    if missing:
        raise ValueError(f"Faltan columnas de métricas por estación: {sorted(missing)}")
    rows = []
    for (model, split), group in station_metrics.groupby(["model", "split"], sort=True):
        usable = group.loc[~group["insufficient_samples"]]
        rows.append({
            "model": model, "split": split,
            "stations_total": int(len(group)),
            "stations_with_sufficient_evidence": int(len(usable)),
            "N_total": int(group["N"].sum()),
            "N_in_macro": int(usable["N"].sum()),
            "observation_weighted_MAE_in_macro": float(np.average(usable["MAE"], weights=usable["N"])) if len(usable) else np.nan,
            "macro_mean_MAE": float(usable["MAE"].mean()) if len(usable) else np.nan,
            "macro_median_MAE": float(usable["MAE"].median()) if len(usable) else np.nan,
            "macro_mean_RMSE": float(usable["RMSE"].mean()) if len(usable) else np.nan,
            "macro_mean_bias": float(usable["bias"].mean()) if len(usable) else np.nan,
        })
    return pd.DataFrame(rows)


def incertidumbre_diferencias_modelos(predictions: pd.DataFrame, *, n_bootstrap=2000, random_state=42):
    """IC percentil pareado, remuestreando estaciones completas.

    Preserva la dependencia temporal dentro de estación. Una diferencia menor
    que cero favorece al primer modelo indicado en la fila.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError("n_bootstrap debe ser un entero >= 100.")
    rng = np.random.default_rng(random_state)
    rows = []
    for split, split_frame in predictions.groupby("split", sort=True):
        models = sorted(split_frame["model"].unique())
        truth = split_frame.drop_duplicates("sample_id").set_index("sample_id")["y_true"]
        station = split_frame.drop_duplicates("sample_id").set_index("sample_id")["station"]
        pred = split_frame.pivot(index="sample_id", columns="model", values="y_pred").reindex(truth.index)
        if pred.isna().any().any():
            raise ValueError("La incertidumbre pareada requiere la misma cohorte en todos los modelos.")
        station_names = station.unique()
        for model_a, model_b in combinations(models, 2):
            aggregate = []
            for station_name in station_names:
                mask = station.eq(station_name).to_numpy()
                observed = truth.to_numpy()[mask]
                error_a = pred[model_a].to_numpy()[mask] - observed
                error_b = pred[model_b].to_numpy()[mask] - observed
                aggregate.append((mask.sum(), np.abs(error_a).sum(), np.abs(error_b).sum(),
                                  np.square(error_a).sum(), np.square(error_b).sum(),
                                  error_a.sum(), error_b.sum()))
            aggregate = np.asarray(aggregate, dtype=float)
            draws = rng.integers(0, len(aggregate), size=(n_bootstrap, len(aggregate)))
            sampled = aggregate[draws].sum(axis=1)
            n = sampled[:, 0]
            boot = {
                "MAE": sampled[:, 1]/n - sampled[:, 2]/n,
                "RMSE": np.sqrt(sampled[:, 3]/n) - np.sqrt(sampled[:, 4]/n),
                "bias": sampled[:, 5]/n - sampled[:, 6]/n,
            }
            full_n = aggregate[:, 0].sum()
            point = {
                "MAE": aggregate[:, 1].sum()/full_n - aggregate[:, 2].sum()/full_n,
                "RMSE": np.sqrt(aggregate[:, 3].sum()/full_n) - np.sqrt(aggregate[:, 4].sum()/full_n),
                "bias": aggregate[:, 5].sum()/full_n - aggregate[:, 6].sum()/full_n,
            }
            for metric, values in boot.items():
                low, high = np.quantile(values, [0.025, 0.975])
                rows.append({"split": split, "model_a": model_a, "model_b": model_b,
                             "metric": metric, "difference_a_minus_b": float(point[metric]),
                             "ci95_low": float(low), "ci95_high": float(high),
                             "probability_a_better": float(np.mean(values < 0)),
                             "stations": len(station_names), "N": int(full_n),
                             "bootstrap_unit": "station", "n_bootstrap": n_bootstrap})
    return pd.DataFrame(rows)


def diagnosticar_desempeno(prepared, predictions, station_metrics=None, *, n_bootstrap=2000, random_state=42):
    frame = prepared.frame
    train = prepared.partitions['train'].metadata
    partitions = []
    for split, part in prepared.partitions.items():
        m = part.metadata
        partitions.append(dict(split=split, N=len(m), stations=m.station.nunique(),
            start=m.date.min(), end=m.date.max(), target_mean=m.y_true.mean(),
            target_median=m.y_true.median(), target_max=m.y_true.max()))
    baseline_rows = []
    for split in ('validation','test'):
        m = prepared.partitions[split].metadata
        last = m.sample_id.map(frame.set_index('sample_id').dqo_lag_1).fillna(train.y_true.median())
        for name, values in [('Media entrenamiento', np.full(len(m),train.y_true.mean())),
                             ('Mediana entrenamiento', np.full(len(m),train.y_true.median())),
                             ('Ultima DQO observada', last.to_numpy())]:
            baseline_rows.append(dict(model=name, split=split, **resumen_metricas(m.y_true, values)))
    concentration, generalization = [], []
    for model, group in predictions.query("split == 'test'").groupby('model'):
        count = max(1, int(np.ceil(len(group)*.05)))
        total = group.squared_error.sum()
        high = group[group.y_true >= 100]
        concentration.append(dict(model=model, worst_5pct_N=count,
            squared_error_share_worst_5pct=group.nlargest(count,'squared_error').squared_error.sum()/total if total else 0,
            high_DQO_threshold=100, high_DQO_N=len(high),
            high_DQO_bias=high.error.mean(), high_DQO_MAE=high.absolute_error.mean()))
        seen = group.station.isin(train.station)
        for label, subset in [('Con etiquetas en train', group[seen]),('Sin etiquetas en train',group[~seen])]:
            if len(subset):
                generalization.append(dict(model=model, station_group=label, stations=subset.station.nunique(),
                    **resumen_metricas(subset.y_true,subset.y_pred)))
    features = prepared.quality_report['selected_chemical_features']
    missing = frame[frame.eligible].groupby('split')[features].agg(lambda s: s.isna().mean()).T.reset_index(names='feature')
    station_n = prepared.partitions['test'].metadata.groupby('station').size()
    spans = []
    for _, g in frame.groupby('station', sort=False):
        spans.append((g.date - g.date.shift(prepared.split_config['sequence_length']-1)).dt.total_seconds()/86400)
    span = pd.concat(spans).dropna()
    report = prepared.quality_report
    observations = [
        f"{report['source_rows']} filas largas no equivalen a muestras independientes: {report['visits']} visitas, {report['visits_with_usable_target']} con DQO utilizable.",
        f"Política de censura: {report['censored_target_policy']}. Etiquetas censuradas usadas en entrenamiento: {report.get('censored_training_labels_used',0)}; en evaluación: {report.get('censored_evaluation_labels_used',0)}. La sustitución es aproximada; evaluar solo DQO exacta limita el alcance a concentraciones cuantificadas.",
        f"Mediana entre visitas: {frame.gap_days.median():.0f} días; {report['gaps_over_180_days']} intervalos superan 180 días. No se inventan fechas ni se interpolan etiquetas.",
        f"Una ventana de {prepared.split_config['sequence_length']} visitas abarca una mediana de {span.median():.0f} días; no equivale a pasos temporales uniformes.",
        f"Prueba cubre {len(station_n)} estaciones: {(station_n<5).sum()} tienen menos de 5 observaciones. R² local puede ser inestable aunque el global sea positivo.",
        "La DQO se conserva sin recortar extremos. El RMSE pondera fuertemente pocos errores grandes; consulte su concentración y sesgo para DQO >=100 (umbral descriptivo, no normativo).",
        f"Transformación log1p de la preparación de referencia: {prepared.target_log}. Si hay búsqueda por modelo, consulte model_prepared y selection_trials. Optimizar pérdida en log no equivale a minimizar RMSE en mg O2/L.",
        "No se demuestra una causa con estos descriptivos. Cambios de cobertura, irregularidad y estaciones sin etiquetas previas son hipótesis evaluables; no justifican mezclar futuro con pasado.",
    ]
    sample_flow = pd.DataFrame([
        ('Filas largas originales (todas las propiedades)', report['source_rows']),
        ('Filas tras quitar duplicados exactos', report['source_rows']-report['exact_duplicate_rows']),
        ('Visitas: estación + fecha/hora únicas', len(frame)),
        ('Visitas con etiqueta DQO utilizable', int(frame.y_true.notna().sum())),
        ('Etiquetas sin historial suficiente', int((frame.y_true.notna() & ~frame.eligible).sum())),
        ('Ejemplos elegibles dentro de la evaluación', int((frame.eligible & frame.split.isin(['train','validation','test'])).sum())),
        ('Ejemplos elegibles fuera de la ventana', int((frame.eligible & ~frame.split.isin(['train','validation','test'])).sum())),
        *[(f'Ejemplos {s}',len(p.y)) for s,p in prepared.partitions.items()],
    ],columns=['etapa','cantidad'])
    station_macro = resumen_macro_estaciones(station_metrics) if station_metrics is not None else pd.DataFrame()
    model_uncertainty = incertidumbre_diferencias_modelos(
        predictions, n_bootstrap=n_bootstrap, random_state=random_state
    ) if predictions.model.nunique() > 1 else pd.DataFrame()
    return dict(sample_flow=sample_flow, partitions=pd.DataFrame(partitions), baselines=pd.DataFrame(baseline_rows),
                error_concentration=pd.DataFrame(concentration), station_generalization=pd.DataFrame(generalization),
                station_macro=station_macro, model_difference_uncertainty=model_uncertainty,
                feature_missingness=missing, observations=observations)
