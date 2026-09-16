"""Búsqueda acotada, declarada antes de consultar métricas de prueba."""
import numpy as np
import pandas as pd
from Data_Manage import Data_Manage
from Performance_Diagnostics import resumen_metricas
from XGBoost_Algorithm import XGBoost_Algorithm


def seleccionar_xgboost(csv_path, target, reference, *, season_config=None, random_state=42, verbose=True):
    """12 candidatos: log/raw x cobertura 70/40/15 % x profundidad 3/5.

    Selección por RMSE de validación en mg O2/L. No se predice prueba aquí.
    Se devuelve el preprocesamiento del ganador sin reajustarlo.
    """
    trials, best = [], None
    cuts = reference.split_config
    for target_log in (True, False):
        for coverage in (.7, .4, .15):
            data = Data_Manage(csv_path, target, sequence_length=cuts['sequence_length'],
                random_state=random_state, transformar_target_log=target_log).preparar_evaluacion(
                    train_end=cuts['train_end'], validation_end=cuts['validation_end'],
                    season_config=season_config, coverage_threshold=coverage,
                    censored_target_policy=cuts['censored_target_policy'])
            for split in ('train','validation','test'):
                pd.testing.assert_frame_equal(data.partitions[split].metadata, reference.partitions[split].metadata)
            valid = data.partitions['validation']
            for depth in (3, 5):
                estimator = XGBoost_Algorithm(random_state=random_state, max_depth=depth).fit(data)
                prediction = np.maximum(data.inverse_target(estimator.predict(valid)), 0)
                metrics = resumen_metricas(valid.metadata.y_true, prediction)
                trials.append(dict(target_log=target_log, coverage_threshold=coverage,
                    max_depth=depth, features=len(data.feature_cols), **metrics))
                if verbose:
                    print(f"XGBoost validación: log={target_log}, cobertura={coverage}, profundidad={depth}, RMSE={metrics['RMSE']:.3f}", flush=True)
                if best is None or metrics['RMSE'] < best[0]:
                    best = (metrics['RMSE'], data, estimator, len(trials)-1)
    table = pd.DataFrame(trials)
    table['selected'] = table.index == best[3]
    best[2].training_report['selection_trials'] = table.copy()
    best[2].training_report['preprocessing_selected'] = best[1].split_config.copy()
    best[2].training_report['selection_rule'] = 'minimum validation RMSE in mg O2/L; no test ranking'
    return best[1], best[2], table
