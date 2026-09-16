"""Multivisit LSTM using chronological sequences from the same station."""
from __future__ import annotations

import numpy as np


class LSTM_Algorithm:
    name = "LSTM"

    def __init__(self, archivo_csv=None, target="DEMANDA QUIMICA DE OXIGENO", *, sequence_length=5, random_state=42, quick=False, epochs=60, batch_size=32, units=64, dropout=0.1, monitor_original_rmse=False):
        self.archivo = archivo_csv
        self.target = target
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.quick = quick
        self.epochs = epochs
        self.batch_size = batch_size
        if units < 1 or not 0 <= dropout < 1:
            raise ValueError('units debe ser positivo y dropout estar en [0,1).')
        self.units, self.dropout = units, dropout
        self.monitor_original_rmse = monitor_original_rmse
        self.model = None
        self.training_report = {}

    def fit(self, prepared, **_):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("LSTM solicitado, pero TensorFlow no está disponible. Instale tensorflow o seleccione --models XGBoost SVM.") from exc
        train, valid = prepared.partitions["train"], prepared.partitions["validation"]
        if not len(train.y) or not len(valid.y):
            raise ValueError("LSTM requiere entrenamiento y validación no vacíos.")
        if train.X_sequence.ndim != 3 or train.X_sequence.shape[1] < 2:
            raise ValueError("LSTM requiere secuencias reales de al menos dos visitas.")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.random_state)
        deterministic = True
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            deterministic = False
        units = 16 if self.quick else self.units
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=train.X_sequence.shape[1:]),
            tf.keras.layers.LSTM(units, dropout=self.dropout),
            tf.keras.layers.Dense(max(8, units // 2), activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), loss="mse")
        epochs = min(self.epochs, 3) if self.quick else self.epochs
        callbacks = []
        monitor = 'val_loss'
        if self.monitor_original_rmse:
            class PhysicalRMSE(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    scaled = np.asarray(self.model(valid.X_sequence, training=False)).ravel()
                    prediction = np.maximum(prepared.inverse_target(scaled),0)
                    rmse = float(np.sqrt(np.mean((valid.metadata.y_true.to_numpy()-prediction)**2)))
                    if not np.isfinite(rmse):
                        raise ValueError('RMSE físico de validación no finito.')
                    logs['val_rmse_original'] = rmse
            callbacks.append(PhysicalRMSE())
            monitor = 'val_rmse_original'
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, mode='min', patience=8, restore_best_weights=True))
        history = self.model.fit(
            train.X_sequence, train.y, validation_data=(valid.X_sequence, valid.y),
            epochs=epochs, batch_size=self.batch_size, shuffle=False,
            callbacks=callbacks, verbose=0,
        )
        self.training_report = {
            "selection_partition": "validation", "selection_metric": "RMSE in mg O2/L" if self.monitor_original_rmse else ("MSE of scaled log1p(DQO)" if prepared.target_log else "MSE of scaled DQO"),
            "history": {k: [float(x) for x in v] for k, v in history.history.items()},
            "best_epoch": int(np.argmin(history.history[monitor])) + 1,
            "epochs_run": len(history.history["loss"]),
            "parameters": {"units": units, "dropout": self.dropout, "batch_size": self.batch_size, "epochs": epochs, "monitor": monitor},
            "sequence_length": int(train.X_sequence.shape[1]), "shuffle": False,
            "deterministic_ops_enabled": deterministic, "refit_on_validation": False,
        }
        return self

    def predict(self, partition):
        if self.model is None:
            raise RuntimeError("Debe entrenar LSTM antes de predecir.")
        return np.asarray(self.model.predict(partition.X_sequence, verbose=0), dtype=float).reshape(-1)

    def ejecutar(self, epochs=None, batch_size=None, **kwargs):
        if self.archivo is None:
            raise ValueError("Indique archivo_csv o use fit(prepared).")
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        from Diagnosis_Algorithms import ejecutar_evaluacion
        return ejecutar_evaluacion(archivo_csv=self.archivo, target=self.target, models=[self.name], model_instances={self.name: self}, sequence_length=self.sequence_length, random_state=self.random_state, quick=self.quick, **kwargs)
