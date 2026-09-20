"""Punto de entrada: entrena, imprime MAE/RMSE/R² y devuelve DataFrames.

Uso: python Diagnosis_Algorithms.py
No crea JSON ni dashboard. Los CSV son opcionales (--export-csv).
En Python/notebook: resultados = ejecutar_evaluacion(); resultados['metrics']['global'].
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from SpatioTemporal_Evaluation import (construir_predicciones, metricas_desagregadas,
    metricas_por_banda_dqo, guardar_evaluacion, validar_cohorte)
from Performance_Diagnostics import diagnosticar_desempeno

DEFAULT_CSV = "Data_Histórica_de_Calidad_de_Agua_Superficial_y_Sedimentos_20260920.csv"
MODEL_NAMES = ("XGBoost", "LSTM", "SVM")


def _normalise_models(models):
    tokens = models.replace(',', ' ').split() if isinstance(models, str) else models
    aliases = {name.lower(): name for name in MODEL_NAMES}
    names = [aliases.get(token.strip().lower()) for item in tokens for token in item.split(',')]
    if not names or None in names or len(set(names)) != len(names):
        raise ValueError("Seleccione modelos distintos entre XGBoost, LSTM y SVM.")
    return names


def _build_model(name, *, random_state, quick, sequence_length, epochs, batch_size):
    if name == "XGBoost":
        from XGBoost_Algorithm import XGBoost_Algorithm
        return XGBoost_Algorithm(random_state=random_state, quick=quick)
    if name == "SVM":
        from SVM_Algorithm import SVM_Algorithm
        return SVM_Algorithm(random_state=random_state, quick=quick)
    from LSTM_Algorithm import LSTM_Algorithm
    return LSTM_Algorithm(sequence_length=sequence_length, random_state=random_state,
                          quick=quick, epochs=epochs, batch_size=batch_size)


def _nombre_salida_modelo(name, estimator, data, *, improve_xgboost=False, selection_trials=None,
                          regularize_lstm=False, target_log=True, coverage_threshold=0.7):
    """Nombre inequívoco para tablas, archivos y comparaciones del notebook."""
    if name == "XGBoost" and improve_xgboost:
        selected = selection_trials.loc[selection_trials["selected"]].iloc[0]
        transform = "log1p" if bool(selected["target_log"]) else "raw"
        coverage = int(round(float(selected["coverage_threshold"]) * 100))
        depth = int(selected["max_depth"])
        return f"XGBoost_selected_{transform}_cov{coverage}_depth{depth}"
    if name == "XGBoost":
        # Conserva el nombre corto histórico sólo para la configuración por defecto.
        if target_log and abs(float(coverage_threshold) - .7) < 1e-12:
            return "XGBoost_original"
        transform = "log1p" if target_log else "raw"
        coverage = int(round(float(coverage_threshold) * 100))
        return f"XGBoost_original_{transform}_cov{coverage}"
    if name == "LSTM":
        units = int(getattr(estimator, "units", 64))
        # d01 representa dropout=0.1; d03 representa dropout=0.3.
        dropout = int(round(float(getattr(estimator, "dropout", .1)) * 10))
        suffix = "_physicalRMSE" if getattr(estimator, "monitor_original_rmse", False) else ""
        return f"LSTM_{'selected' if regularize_lstm else 'original'}_{units}_d{dropout:02d}{suffix}"
    return "SVM_original"


def _tabla(table, columns=None):
    return table.loc[:, columns].to_string(index=False, float_format=lambda x: f"{x:.3f}") if columns else table.to_string(index=False, float_format=lambda x: f"{x:.3f}")


def mostrar_resultados(resultados, top_stations=10):
    """Aquí se presentan las métricas principales, sin abrir archivos externos.

    MAE/RMSE: mg O2/L, menores son mejores. R²: adimensional, mayor es mejor.
    Prueba es el resultado principal; validación intervino en elegir modelos.
    Las tablas completas permanecen accesibles en resultados['metrics'].
    """
    metrics, diagnostics = resultados['metrics'], resultados['diagnostics']
    sections = ["RESULTADO PRINCIPAL: PRUEBA TEMPORAL (misma cohorte en todos los modelos)",
                "MAE y RMSE en mg O2/L: menor es mejor. R²: mayor es mejor; puede ser negativo.",
                _tabla(metrics['global'].query("split == 'test'"), ['model', 'N', 'MAE', 'RMSE', 'R2']),
                "\nREFERENCIAS SIMPLES EN PRUEBA (no son modelos ajustados con prueba)",
                _tabla(diagnostics['baselines'].query("split == 'test'")),
                "\nENTRENAMIENTO / VALIDACIÓN / PRUEBA (entrenamiento es desempeño aparente)",
                _tabla(pd.concat([resultados['train_metrics'], metrics['global'][['model','split','N','MAE','RMSE','R2']]], ignore_index=True)),
                "\nCOBERTURA Y FECHAS", _tabla(diagnostics['partitions']),
                "\nMÉTRICAS POR AÑO: PRUEBA", _tabla(metrics['year'].query("split == 'test'"), ['model','year','N','MAE','RMSE','R2']),
                "\nMÉTRICAS POR TEMPORADA: PRUEBA (calendario aproximado salvo configuración local)",
                _tabla(metrics['season'].query("split == 'test'"), ['model','season','N','MAE','RMSE','R2']),
                "\nERROR POR MAGNITUD DE DQO: PRUEBA (cuartiles fijados con entrenamiento)",
                _tabla(metrics['dqo_band'].query("split == 'test'"),
                       ['model','dqo_band','N','MAE','RMSE','R2','bias',
                        'median_absolute_error','underestimation_pct','overestimation_pct']),
                "\nESTACIONES CON MAYOR RMSE: PRUEBA (solo grupos con suficientes muestras)"]
    stations = metrics['station'].query("split == 'test' and not insufficient_samples")
    top = stations.sort_values('RMSE', ascending=False).groupby('model', sort=False).head(top_stations)
    sections.append(_tabla(top, ['model','station','N','period_start','period_end','n_years',
                                 'MAE','RMSE','R2','bias','median_absolute_error']))
    if not diagnostics['station_macro'].empty:
        global_test = metrics['global'].query("split == 'test'")[['model','MAE']].rename(columns={'MAE':'observation_weighted_MAE'})
        macro_test = diagnostics['station_macro'].query("split == 'test'").merge(global_test, on='model', how='left')
        sections.extend(["\nOBSERVACIONES VS ESTACIONES (macro sólo con evidencia suficiente)",
                         _tabla(macro_test)])
    sections.extend(["\nCONCENTRACIÓN DEL ERROR", _tabla(diagnostics['error_concentration']),
                     "\nESTACIONES VISTAS / SIN ETIQUETAS DE ENTRENAMIENTO", _tabla(diagnostics['station_generalization']),
                     "\nLECTURA DEL DIAGNÓSTICO", *diagnostics['observations']])
    if not diagnostics['model_difference_uncertainty'].empty:
        sections.extend(["\nINCERTIDUMBRE DE DIFERENCIAS: PRUEBA (IC 95%, bootstrap por estación; A-B < 0 favorece A)",
                         _tabla(diagnostics['model_difference_uncertainty'].query("split == 'test'"))])
    for crossing in ('hydro_zone_year', 'hydro_subzone_year', 'altitude_band_year'):
        table = metrics.get(crossing, pd.DataFrame())
        if not table.empty:
            dimensions = [column for column in ('hydro_zone','hydro_subzone','altitude_band') if column in table]
            shown = (table.query("split == 'test' and not insufficient_samples")
                     .sort_values(['model','MAE'], ascending=[True, False])
                     .groupby('model', sort=False).head(5))
            sections.extend([f"\nCRUCE {crossing}: CINCO GRUPOS CON MAYOR MAE POR MODELO",
                             _tabla(shown, ['model',*dimensions,'year','N','n_stations','MAE','RMSE','bias'])])
    error_analysis = resultados.get('error_analysis')
    if error_analysis:
        sections.append("\nMODELO AUXILIAR DEL ERROR (su R² no es el R² de DQO)")
        for model, info in error_analysis['models'].items():
            sections.append(f"{model}: estado={info['status']}; R² auxiliar={info.get('surrogate_test_R2')}; mejora MAE vs referencia={info.get('MAE_skill_against_validation_median')}")
        for key, table in error_analysis.get('tables', {}).items():
            if key.endswith('shap_error_global'):
                sections.append(_tabla(table.sort_values('mean_abs_shap', ascending=False)))
    if resultados.get('selection_trials') is not None:
        sections.extend(['\nBÚSQUEDA XGBOOST: SOLO VALIDACIÓN; selected INDICA EL GANADOR', _tabla(resultados['selection_trials'])])
    if 'sample_flow' in diagnostics:
        sections.extend(['\nDE FILAS DEL CSV A EJEMPLOS DE DQO', _tabla(diagnostics['sample_flow'])])
    regional = resultados.get('regional_metrics', {})
    within = regional.get('within_regions')
    if within is not None and not within.empty:
        sections.extend([
            '\nRESUMEN REGIONAL: R2 DENTRO DE REGIONES (ponderado por variabilidad)',
            _tabla(within.query("split == 'test'"),
                   ['model', 'grouping', 'N_included', 'N_excluded',
                    'regions_included', 'R2_within_regions'])
        ])
    for dimension, table in regional.items():
        if dimension != 'within_regions' and not table.empty:
            representative = (table.query("split == 'test' and representative")
                              .sort_values(['model', 'N'], ascending=[True, False])
                              .groupby('model', sort=False).head(10))
            sections.extend([
                f'\nREGIONES REPRESENTATIVAS: {dimension} '
                '(hasta 10 por modelo; R2_reported exige N>=20 y >=2 estaciones)',
                _tabla(representative,
                       ['model','region','N','n_stations','MAE','RMSE','R2_reported'])
            ])
    text = '\n\n'.join(sections)
    print(text, flush=True)
    return text


def ejecutar_evaluacion(archivo_csv=DEFAULT_CSV, target="DEMANDA QUIMICA DE OXIGENO", *,
    output_dir="outputs/evaluacion_espaciotemporal", models=MODEL_NAMES,
    train_end=None, validation_end=None, test_start=None, test_end=None, train_ratio=0.6, validation_ratio=0.2,
    sequence_length=5, season_config=None, coverage_threshold=0.7,
    censored_target_policy="exclude", random_state=42, epochs=60, batch_size=32,
    min_samples=5, max_error_samples=300, quick=False, no_plots=False,
    no_error_analysis=False, model_instances=None, export_csv=False,
    target_log=True, verbose=True, improve_xgboost=False, regularize_lstm=False,
    ablate_groups=(), allowed_chemical_features=None):
    """Devuelve resultados legibles en memoria; exportar CSV es optativo.

    Ajustes: train. Selección/early stopping: validation. Resultado: test.
    No se recortan los valores reales extremos de DQO ni se reajusta sobre test.
    Cada ejecución puede reutilizar la carpeta de figuras; las tablas retornadas
    siempre pertenecen a la ejecución actual (no se cargan resultados antiguos).
    """
    names = _normalise_models(models)
    if sequence_length < 2 or min_samples < 2 or epochs < 1 or batch_size < 1 or max_error_samples < 1:
        raise ValueError("sequence_length/min_samples >= 2; epochs/batch_size/max_error_samples > 0.")
    csv_path = Path(archivo_csv)
    if not csv_path.exists() and not csv_path.is_absolute():
        csv_path = Path(__file__).resolve().parent / csv_path
    manager = Data_Manage(csv_path, target, sequence_length=sequence_length,
                          random_state=random_state, transformar_target_log=target_log)
    prepared = manager.preparar_evaluacion(train_end=train_end, validation_end=validation_end,
        test_start=test_start, test_end=test_end,
        train_ratio=train_ratio, validation_ratio=validation_ratio, season_config=season_config,
        coverage_threshold=coverage_threshold, censored_target_policy=censored_target_policy,
        ablate_groups=ablate_groups, allowed_chemical_features=allowed_chemical_features)
    if verbose:
        print('Muestras elegibles:', {s: len(p.y) for s,p in prepared.partitions.items()}, flush=True)
        print('Cortes temporales:', prepared.split_config['train_end'], prepared.split_config['validation_end'], flush=True)
    estimators, blocks, train_metrics, training = dict(model_instances or {}), [], [], {}
    model_prepared, selection_trials, model_labels, model_configurations = {}, None, {}, {}
    for name in names:
        if verbose:
            print(f'Entrenando {name}...', flush=True)
        started = time.perf_counter()
        data = prepared
        if name == 'XGBoost' and improve_xgboost:
            if quick or name in estimators:
                raise ValueError('improve_xgboost no admite quick ni una instancia XGBoost externa.')
            from Model_Selection import seleccionar_xgboost
            data, estimator, selection_trials = seleccionar_xgboost(csv_path, target, prepared,
                season_config=season_config, random_state=random_state, verbose=verbose)
        else:
            estimator = estimators.get(name) or _build_model(name, random_state=random_state,
                quick=quick, sequence_length=sequence_length, epochs=epochs, batch_size=batch_size)
            if name == 'LSTM' and regularize_lstm:
                estimator.units, estimator.dropout, estimator.monitor_original_rmse = 32, 0.3, True
            estimator.fit(data, inverse_target=data.inverse_target)
        model_prepared[name] = data
        estimators[name] = estimator
        display_name = _nombre_salida_modelo(
            name, estimator, data, improve_xgboost=improve_xgboost,
            selection_trials=selection_trials, regularize_lstm=regularize_lstm,
            target_log=data.target_log, coverage_threshold=data.split_config["coverage_threshold"]
        )
        model_labels[name] = display_name
        training[name] = dict(estimator.training_report, elapsed_seconds=time.perf_counter()-started,
                              output_name=display_name)
        model_configurations[display_name] = {
            "model_family": name, "target_log1p": bool(data.target_log),
            "feature_count": len(data.feature_cols),
            "coverage_threshold": data.split_config["coverage_threshold"],
            "ablated_feature_groups": data.split_config.get("ablated_feature_groups", []),
            "chemical_availability_filter_applied": data.split_config.get("chemical_availability_filter_applied", False),
            "selected_chemical_features": data.quality_report.get("selected_chemical_features", []),
            "training_parameters": estimator.training_report.get("parameters", {}),
        }
        for split, part in data.partitions.items():
            prediction = np.asarray(data.inverse_target(estimator.predict(part)), dtype=float).reshape(-1)
            if len(prediction) != len(part.y) or not np.isfinite(prediction).all():
                raise ValueError(f'Predicciones inválidas: {name}/{split}')
            prediction = np.maximum(prediction, 0)
            truth = part.metadata.y_true.to_numpy(float)
            if split == 'train':
                from Performance_Diagnostics import resumen_metricas
                train_metrics.append(dict(model=display_name, split=split, **resumen_metricas(truth, prediction)))
            else:
                blocks.append(construir_predicciones(part.metadata, truth, prediction, display_name, split))
    predictions = pd.concat(blocks, ignore_index=True)
    validar_cohorte(predictions, modelos_esperados=list(model_labels.values()))
    tables = metricas_desagregadas(predictions, min_samples=min_samples)
    dqo_bands, dqo_band_boundaries = metricas_por_banda_dqo(
        predictions, prepared.partitions['train'].metadata.y_true.to_numpy(), min_samples=min_samples
    )
    tables['dqo_band'] = dqo_bands
    result = dict(predictions=predictions, metrics=tables, prepared=prepared, models=estimators,
                  model_prepared=model_prepared, selection_trials=selection_trials,
                  model_labels=model_labels, model_configurations=model_configurations,
                  train_metrics=pd.DataFrame(train_metrics), training=training,
                  quality_report=prepared.quality_report, split_config=prepared.split_config,
                  dqo_band_boundaries=dqo_band_boundaries,
                  diagnostics=diagnosticar_desempeno(
                      model_prepared[names[0]] if len(names) == 1 else prepared, predictions,
                      station_metrics=tables['station'], random_state=random_state),
                  error_analysis=None, plots=[], output_dir=str(Path(output_dir).resolve()))
    # Trazabilidad legible, sin manifiestos JSON.
    result['source_sha256'] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    from Regional_Evaluation import metricas_regionales
    result['regional_metrics'] = metricas_regionales(predictions)
    if export_csv:
        guardar_evaluacion(predictions, output_dir, min_samples=min_samples)
        dqo_bands.to_csv(Path(output_dir)/'metrics_dqo_band.csv', index=False)
        dqo_band_boundaries.to_csv(Path(output_dir)/'dqo_band_boundaries.csv', index=False)
        prepared.frame.to_csv(Path(output_dir)/'coverage.csv', index=False)
    if not no_error_analysis:
        from Error_Analysis import analizar_error
        result['error_analysis'] = analizar_error(predictions, output_dir,
            random_state=random_state, max_samples=max_error_samples, export_csv=export_csv)
    if not no_plots:
        from SpatioTemporal_Plots import generar_visualizaciones
        result['plots'] = generar_visualizaciones(predictions, tables, output_dir)
    if verbose:
        result['summary'] = mostrar_resultados(result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description='DQO: resultados legibles por modelo, estación y tiempo.')
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--output', default='outputs/evaluacion_espaciotemporal', help='Carpeta de figuras y CSV opcionales.')
    parser.add_argument('--models', nargs='+', default=list(MODEL_NAMES))
    parser.add_argument('--target', default='DEMANDA QUIMICA DE OXIGENO')
    parser.add_argument('--train-end')
    parser.add_argument('--validation-end')
    parser.add_argument('--test-end', help='Fin opcional de prueba; fechas posteriores quedan fuera de esta evaluación.')
    parser.add_argument('--test-start', help='Inicio opcional de prueba; el intervalo previo queda fuera de evaluación.')
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--validation-ratio', type=float, default=0.2)
    parser.add_argument('--sequence-length', type=int, default=5)
    parser.add_argument('--coverage-threshold', type=float, default=0.7)
    parser.add_argument('--censored-target-policy', choices=['exclude','half_limit','train_half_limit'], default='exclude')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--max-error-samples', type=int, default=300)
    parser.add_argument('--quick', action='store_true', help='Solo prueba de funcionamiento, no experimento definitivo.')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--no-error-analysis', action='store_true')
    parser.add_argument('--export-csv', action='store_true', help='Opcional: exportar las tablas que también se muestran en consola.')
    parser.add_argument('--raw-target', action='store_true', help='Experimento: entrenar sin log1p; conservar escalado train-only.')
    parser.add_argument('--improve-xgboost', action='store_true', help='Seleccionar transformación, cobertura y profundidad con validación temporal (12 candidatos).')
    parser.add_argument('--regularize-lstm', action='store_true', help='LSTM 32 unidades, dropout 0.3, parada por RMSE físico.')
    parser.add_argument('--ablate-groups', nargs='*', default=(), choices=[
        'historical_dqo', 'geography', 'time', 'contemporary_covariates'],
        help='Diagnóstico: elimina grupos completos de predictores antes del ajuste.')
    return parser


def main(argv=None):
    # Evita fallos de impresión de unidades/símbolos en consolas Windows antiguas.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    args = vars(build_parser().parse_args(argv))
    args['archivo_csv'], args['output_dir'] = args.pop('csv'), args.pop('output')
    args['target_log'] = not args.pop('raw_target')
    ejecutar_evaluacion(**args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
