"""SVR selection on a future validation interval, without shuffled CV."""
from __future__ import annotations

from itertools import product
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVR


class SVM_Algorithm:
    name = "SVM"

    def __init__(self, archivo_csv=None, target="DEMANDA QUIMICA DE OXIGENO", *, random_state=42, quick=False):
        self.archivo = archivo_csv
        self.target = target
        self.random_state = random_state
        self.quick = quick
        self.model = None
        self.training_report = {}

    def fit(self, prepared, *, inverse_target=None, **_):
        train, valid = prepared.partitions["train"], prepared.partitions["validation"]
        if not len(train.y) or not len(valid.y):
            raise ValueError("SVM requiere entrenamiento y validación no vacíos.")
        candidates = [dict(C=10.0, epsilon=0.1, gamma="scale")]
        if not self.quick:
            candidates = [dict(C=c, epsilon=eps, gamma=gamma) for c, eps, gamma in product((1.0, 10.0, 50.0), (0.05, 0.2), ("scale", 0.01))]
        best_rmse, trials = float("inf"), []
        for parameters in candidates:
            model = SVR(kernel="rbf", cache_size=500, max_iter=100000, **parameters)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(train.X, train.y)
            prediction, truth = model.predict(valid.X), np.asarray(valid.y, dtype=float)
            if inverse_target is not None:
                truth = np.asarray(valid.metadata["y_true"], dtype=float)
                prediction = np.maximum(inverse_target(prediction), 0.0)
            rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
            converged = int(model.fit_status_) == 0
            trials.append({"parameters": parameters, "validation_rmse": rmse, "converged": converged, "warnings": [str(item.message) for item in recorded]})
            if converged and np.isfinite(rmse) and rmse < best_rmse:
                self.model, best_rmse = model, rmse
        if self.model is None:
            raise RuntimeError("Ningún candidato SVM convergió con predicciones finitas; revise datos o límite de iteraciones.")
        self.training_report = {
            "parameters": self.model.get_params(), "selection_partition": "validation",
            "selection_metric": "RMSE in original DQO units" if inverse_target else "RMSE of scaled log1p(DQO)",
            "validation_rmse": best_rmse, "trials": trials,
            "support_vectors": int(len(self.model.support_)), "refit_on_validation": False,
        }
        return self

    def predict(self, partition):
        if self.model is None:
            raise RuntimeError("Debe entrenar SVM antes de predecir.")
        return np.asarray(self.model.predict(partition.X), dtype=float).reshape(-1)

    def ejecutar(self, **kwargs):
        if self.archivo is None:
            raise ValueError("Indique archivo_csv o use fit(prepared).")
        from Diagnosis_Algorithms import ejecutar_evaluacion
        return ejecutar_evaluacion(archivo_csv=self.archivo, target=self.target, models=[self.name], model_instances={self.name: self}, random_state=self.random_state, quick=self.quick, **kwargs)
