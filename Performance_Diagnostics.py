"""Diagnóstico descriptivo reproducible; ninguna selección se hace sobre test."""
import numpy as np
import pandas as pd


def resumen_metricas(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    residual = prediction-y
    denominator = np.square(y-y.mean()).sum()
    return dict(N=len(y), MAE=float(np.abs(residual).mean()),
                RMSE=float(np.sqrt(np.square(residual).mean())),
                R2=float(1-np.square(residual).sum()/denominator) if len(y)>1 and denominator>0 else np.nan)


def diagnosticar_desempeno(prepared, predictions):
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
        ('Ejemplos elegibles (train + validation + test)', int(frame.eligible.sum())),
        *[(f'Ejemplos {s}',len(p.y)) for s,p in prepared.partitions.items()],
    ],columns=['etapa','cantidad'])
    return dict(sample_flow=sample_flow, partitions=pd.DataFrame(partitions), baselines=pd.DataFrame(baseline_rows),
                error_concentration=pd.DataFrame(concentration), station_generalization=pd.DataFrame(generalization),
                feature_missingness=missing, observations=observations)
