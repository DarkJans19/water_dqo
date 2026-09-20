"""XGBoost with chronological validation and shared observation IDs."""
from __future__ import annotations

import inspect
import numpy as np


class XGBoost_Algorithm:
    name = "XGBoost"

    def __init__(self, archivo_csv=None, target="DEMANDA QUIMICA DE OXIGENO", *, random_state=42, quick=False, max_depth=5,
                 high_dqo_weight=1.0, high_dqo_quantile=0.75):
        self.archivo = archivo_csv
        self.target = target
        self.random_state = random_state
        self.quick = quick
        self.max_depth = max_depth
        if high_dqo_weight < 1 or not 0 < high_dqo_quantile < 1:
            raise ValueError("high_dqo_weight debe ser >=1 y high_dqo_quantile estar en (0,1).")
        self.high_dqo_weight, self.high_dqo_quantile = high_dqo_weight, high_dqo_quantile
        self.model = None
        self.training_report = {}

    def fit(self, prepared, **_):
        """Fit on train; choose the boosting iteration on future validation."""
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost solicitado, pero xgboost no está disponible. Instale la dependencia o seleccione otros --models.") from exc
        train, valid = prepared.partitions["train"], prepared.partitions["validation"]
        if not len(train.y) or not len(valid.y):
            raise ValueError("XGBoost requiere entrenamiento y validación no vacíos.")
        rounds = 8 if self.quick else 40
        params = dict(
            n_estimators=50 if self.quick else 800, learning_rate=0.05,
            max_depth=3 if self.quick else self.max_depth, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=2.0,
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            n_jobs=2, random_state=self.random_state,
        )
        # Older releases take this in fit; xgboost >= 2.1 uses the constructor.
        fit_params = {}
        if "early_stopping_rounds" in inspect.signature(XGBRegressor.fit).parameters:
            fit_params["early_stopping_rounds"] = rounds
        else:
            params["early_stopping_rounds"] = rounds
        self.model = XGBRegressor(**params)
        threshold = float(np.quantile(train.metadata.y_true.to_numpy(float), self.high_dqo_quantile))
        sample_weight = np.where(train.metadata.y_true.to_numpy(float) > threshold,
                                 self.high_dqo_weight, 1.0)
        self.model.fit(train.X, train.y, sample_weight=sample_weight,
                       eval_set=[(train.X, train.y), (valid.X, valid.y)], verbose=False, **fit_params)
        self.training_report = {
            "parameters": params, "early_stopping_rounds": rounds,
            "best_iteration": int(self.model.best_iteration),
            "selection_partition": "validation",
            "selection_metric": "RMSE of scaled log1p(DQO)" if prepared.target_log else "RMSE of scaled DQO",
            "history": self.model.evals_result(),
            "sample_weighting": {"high_dqo_weight": self.high_dqo_weight,
                                 "training_quantile": self.high_dqo_quantile,
                                 "training_threshold": threshold,
                                 "high_training_N": int((sample_weight > 1).sum())},
        }
        return self

    def predict(self, partition):
        if self.model is None:
            raise RuntimeError("Debe entrenar XGBoost antes de predecir.")
        return np.asarray(self.model.predict(partition.X), dtype=float).reshape(-1)

    def ejecutar(self, **kwargs):
        """Compatibility entrypoint exporting the complete new evaluation."""
        if self.archivo is None:
            raise ValueError("Indique archivo_csv o use fit(prepared).")
        from Diagnosis_Algorithms import ejecutar_evaluacion
        return ejecutar_evaluacion(archivo_csv=self.archivo, target=self.target, models=[self.name], model_instances={self.name: self}, random_state=self.random_state, quick=self.quick, **kwargs)
