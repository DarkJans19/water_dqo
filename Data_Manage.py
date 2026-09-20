"""Datos IDEAM y preprocesamiento causal para estimar DQO en cada visita.

Se dispone de las covariables contemporáneas y los resultados de visitas
anteriores. No representa un pronóstico de horizonte fijo sin observaciones.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = [
    "sample_id", "station", "date", "latitude", "longitude", "elevation",
    "year", "month", "season", "season_source", "gap_days", "history_count",
    "target_censored", "y_true", "split", "eligible", "exclusion_reason",
    "hydro_zone", "hydro_subzone", "altitude_band",
]
DEFAULT_SEASONS = {
    1: "seca_1", 2: "seca_1", 3: "lluviosa_1", 4: "lluviosa_1",
    5: "lluviosa_1", 6: "seca_2", 7: "seca_2", 8: "seca_2",
    9: "lluviosa_2", 10: "lluviosa_2", 11: "lluviosa_2", 12: "seca_1",
}


@dataclass
class Partition:
    X: np.ndarray
    X_sequence: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


@dataclass
class PreparedData:
    partitions: dict[str, Partition]
    feature_cols: list[str]
    frame: pd.DataFrame
    quality_report: dict
    split_config: dict
    target_scaler: StandardScaler
    target_log: bool

    def inverse_target(self, y):
        values = self.target_scaler.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
        return np.expm1(values) if self.target_log else values


class Data_Manage:
    def __init__(self, archivo_csv, target="DEMANDA QUIMICA DE OXIGENO",
                 sequence_length=5, random_state=42, transformar_target_log=True,
                 split_estrategia="temporal", **legacy_options):
        if split_estrategia != "temporal":
            raise ValueError("La evaluación requiere split_estrategia='temporal'.")
        if not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length debe ser un entero positivo.")
        self.archivo, self.target = str(archivo_csv), target
        self.sequence_length, self.random_state = sequence_length, random_state
        self.transformar_target_log = transformar_target_log
        self.scaler, self.target_scaler = StandardScaler(), StandardScaler()
        self.prepared = None
        if legacy_options:
            warnings.warn("Opciones antiguas reemplazadas por preparar_evaluacion: "
                          + ", ".join(legacy_options), DeprecationWarning, stacklevel=2)

    @staticmethod
    def _normalizar_texto(texto):
        normalized = unicodedata.normalize("NFD", str(texto))
        return "".join(c for c in normalized if unicodedata.category(c) != "Mn").strip().upper()

    @staticmethod
    def _simplificar_texto(texto):
        text = Data_Manage._normalizar_texto(texto)
        return re.sub(r"\s+", " ", re.sub(r"\([^)]*\)|[^A-Z0-9 ]", " ", text)).strip()

    def _resolver_target(self, columnas):
        wanted = self._simplificar_texto(self.target)
        candidates = [c for c in columnas if self._simplificar_texto(c) == wanted]
        if len(candidates) != 1:
            raise ValueError(f"Objetivo ambiguo o ausente: {self.target!r}; candidatos: {candidates}")
        return candidates[0]

    @staticmethod
    def _parsear_resultado(serie):
        # En este CSV la coma es decimal ('<3,00'), nunca millares.
        raw = serie.astype("string").str.strip()
        numeric = raw.str.replace(r"^[<>]=?\s*", "", regex=True)
        mixed = numeric.str.contains(",", na=False) & numeric.str.contains(".", regex=False, na=False)
        values = pd.to_numeric(numeric.str.replace(",", ".", regex=False), errors="coerce").astype(float).mask(mixed)
        values = values.where(np.isfinite(values)).mask(raw.str.startswith(">", na=False))
        return values.mask(raw.str.startswith("<", na=False), values / 2)

    @staticmethod
    def _parsear_fechas(serie):
        raw = serie.astype("string").str.strip()
        months = {name: f"{i:02d}" for i, name in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
        numeric = raw.str.replace(r"\b[A-Za-z]{3}\b", lambda m: months.get(m.group(), m.group()), regex=True)
        dates = pd.to_datetime(numeric, format="%Y %m %d %I:%M:%S %p", errors="coerce")
        iso = dates.isna() & raw.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$", na=False)
        dates.loc[iso] = pd.to_datetime(raw.loc[iso], format="ISO8601", errors="coerce")
        return dates

    def _cargar_muestras(self, censored_target_policy):
        if censored_target_policy not in {"exclude", "half_limit", "train_half_limit"}:
            raise ValueError("Política esperada: exclude, half_limit o train_half_limit.")
        path = Path(self.archivo)
        if not path.exists() and not path.is_absolute():
            path = Path(__file__).resolve().parent / path
        df = pd.read_csv(path, dtype="string", encoding="utf-8-sig")
        df.columns = [self._normalizar_texto(c) for c in df.columns]
        required = {"NOMBRE DEL PUNTO DE MONITOREO", "FECHA", "PROPIEDAD OBSERVADA",
                    "RESULTADO", "UNIDAD DEL RESULTADO", "LATITUD", "LONGITUD", "ELEVACION (M.S.N.M.)"}
        if required - set(df.columns):
            raise ValueError(f"Faltan columnas IDEAM: {sorted(required - set(df.columns))}")
        report = {"source_rows": len(df), "source_stations": int(df["NOMBRE DEL PUNTO DE MONITOREO"].nunique()),
                  "exact_duplicate_rows": int(df.duplicated().sum()), "censored_target_policy": censored_target_policy}
        df = df.drop_duplicates().copy()
        df["station"] = df["NOMBRE DEL PUNTO DE MONITOREO"].str.strip().replace("", pd.NA)
        df["date"] = self._parsear_fechas(df["FECHA"])
        df["property"] = df["PROPIEDAD OBSERVADA"].map(self._normalizar_texto)
        df["unit"] = df["UNIDAD DEL RESULTADO"].fillna("UNKNOWN").str.strip().str.upper().str.replace(r"\s+", "", regex=True)
        df["censored"] = df["RESULTADO"].str.match(r"^\s*[<>]", na=False)
        df["value"] = self._parsear_resultado(df["RESULTADO"])
        report.update(invalid_dates=int(df.date.isna().sum()),
                      invalid_numeric_or_right_censored_results=int(df.value.isna().sum()),
                      missing_station_rows=int(df.station.isna().sum()))
        df = df.dropna(subset=["station", "date", "PROPIEDAD OBSERVADA"]).copy()
        target_col = self._resolver_target(df["property"].unique())
        target_mask = df["property"].eq(target_col)
        target_units = sorted(df.loc[target_mask, "unit"].unique().tolist())
        if target_units != ["MGO2/L"]:
            raise ValueError(f"DQO requiere mg O2/L; no se mezclan unidades implícitamente: {target_units}")
        report.update(target_property=target_col, target_unit="mg O2/L", target_rows=int(target_mask.sum()),
                      censored_target_rows=int(df.loc[target_mask, "censored"].sum()))
        keys = ["station", "date"]
        target = df.loc[target_mask].copy()
        target['right_censored'] = target.RESULTADO.str.match(r'^\s*>', na=False)
        target["value"] = target["value"].where(target.value >= 0)
        report["repeated_target_station_date_rows"] = int(target.duplicated(keys).sum())
        if "CODIGO__MUESTRA" in target:
            report["station_dates_with_multiple_sample_codes"] = int((target.groupby(keys)["CODIGO__MUESTRA"].nunique() > 1).sum())
        observed = target.groupby(keys).agg(y_true=("value", "median"), target_censored=("censored", "max"), right_censored=('right_censored','max'))
        observed.loc[observed.right_censored, 'y_true'] = np.nan
        observed = observed.drop(columns='right_censored')
        if censored_target_policy == "exclude":
            observed.loc[observed.target_censored, "y_true"] = np.nan
        geo_names = {"LATITUD": "latitude", "LONGITUD": "longitude", "ELEVACION (M.S.N.M.)": "elevation"}
        for original, canonical in geo_names.items():
            df[canonical] = pd.to_numeric(df[original].str.replace(",", ".", regex=False), errors="coerce").astype(float)
            df[canonical] = df[canonical].where(np.isfinite(df[canonical]))
        report["invalid_or_missing_coordinate_rows"] = int((~(df.latitude.between(-90, 90) & df.longitude.between(-180, 180))).sum())
        df.loc[~df.latitude.between(-90, 90), "latitude"] = np.nan
        df.loc[~df.longitude.between(-180, 180), "longitude"] = np.nan
        geo = df.groupby(keys)[list(geo_names.values())].median()
        report["station_dates_with_conflicting_geography"] = int((df.groupby(keys)[list(geo_names.values())].nunique() > 1).any(axis=1).sum())
        # Propiedad+unidad evita mezclar agua/sedimento y seleccionar unidades
        # usando frecuencias del futuro. La cobertura se ajusta después del corte.
        covariates = df.loc[~target_mask].copy()
        covariates["feature"] = "chem::" + covariates["property"] + " [" + covariates["unit"] + "]"
        chemistry = covariates.pivot_table(index=keys, columns="feature", values="value", aggfunc="median")
        censor = covariates.pivot_table(index=keys, columns="feature", values="censored", aggfunc="max")
        censor.columns = [f"{c}__censored" for c in censor.columns]
        frame = geo.join(observed).join(chemistry).join(censor).reset_index()
        for source, dest in [('ZONA HIDROGRAFICA - ZH', 'hydro_zone'),
                             ('SZH - CODIGO (#AREA#ZONA##SUBZONA)', 'hydro_subzone')]:
            if source in df:
                def unique_label(values):
                    labels = values.dropna().astype(str).str.strip().unique()
                    return labels[0] if len(labels) == 1 else ('sin_dato' if len(labels) == 0 else 'conflicto')
                labels = df.groupby(keys)[source].agg(unique_label).rename(dest).reset_index()
                frame = frame.merge(labels, on=keys, how='left', validate='one_to_one')
            else:
                frame[dest] = 'sin_dato'
        frame['altitude_band'] = pd.cut(frame.elevation, [-np.inf,500,1500,2500,np.inf],
            labels=['<500 m','500–1499 m','1500–2499 m','>=2500 m'], right=False).astype('string').fillna('sin_dato')
        frame["target_censored"] = frame.target_censored.fillna(False).astype(bool)
        frame = frame.sort_values(keys).reset_index(drop=True)
        grouped = frame.groupby("station", sort=False)
        frame["gap_days"] = grouped.date.diff().dt.total_seconds() / 86400
        frame["history_count"] = grouped.cumcount()
        frame["year"], frame["month"] = frame.date.dt.year, frame.date.dt.month
        frame["month_sin"] = np.sin(2 * np.pi * frame.month / 12)
        frame["month_cos"] = np.cos(2 * np.pi * frame.month / 12)
        for lag in (1, 2, 3):
            frame[f"dqo_lag_{lag}"] = grouped.y_true.transform(lambda s: s.ffill().shift(lag))
        previous_dates = frame.date.where(frame.y_true.notna()).groupby(frame.station).transform(lambda s: s.ffill().shift())
        frame["days_since_previous_dqo"] = (frame.date - previous_dates).dt.total_seconds() / 86400
        frame["sample_id"] = [hashlib.sha256(f"{s}|{d.isoformat()}".encode()).hexdigest()[:20] for s, d in zip(frame.station, frame.date)]
        report.update(visits=len(frame), visits_with_usable_target=int(frame.y_true.notna().sum()),
                      gaps_over_180_days=int((frame.gap_days > 180).sum()),
                      max_gap_days=float(frame.gap_days.max()) if frame.gap_days.notna().any() else None,
                      duplicate_policy="median per station, timestamp, property and unit; no target clipping",
                      prediction_task="contemporaneous DQO estimation; previous results available at next visit")
        return frame, chemistry.columns.tolist(), report

    @staticmethod
    def _agregar_estacionalidad(frame, season_config=None):
        frame = frame.copy()
        if season_config is None:
            frame["season"] = frame.month.map(DEFAULT_SEASONS)
            frame["season_source"] = "calendar_proxy_bimodal_not_validated"
            return frame
        if isinstance(season_config, (str, Path)):
            with open(season_config, encoding="utf-8") as stream:
                season_config = json.load(stream)
        if not isinstance(season_config, dict):
            raise ValueError("season_config debe ser un objeto JSON.")
        source = season_config.get("source", "user_calendar")
        calendars, default = dict(season_config.get("stations", {})), season_config.get("default")
        for mapping in [*calendars.values(), *([default] if default is not None else [])]:
            if not isinstance(mapping, dict) or set(map(str, mapping)) != {str(i) for i in range(1, 13)}:
                raise ValueError("Cada calendario debe definir los 12 meses (1..12).")
            if any(not isinstance(v, str) or not v.strip() for v in mapping.values()):
                raise ValueError("Las etiquetas de estacionalidad deben ser textos no vacíos.")
        seasons, sources = [], []
        for station, month in zip(frame.station, frame.month):
            mapping = calendars.get(station, default)
            seasons.append(str(mapping.get(str(month), mapping.get(month))) if mapping is not None else "sin_clasificar")
            sources.append(str(source) if mapping is not None else "no_calendar_for_station")
        frame["season"], frame["season_source"] = seasons, sources
        return frame

    def preparar_evaluacion(self, train_end=None, validation_end=None, test_start=None, test_end=None, train_ratio=0.6,
                           validation_ratio=0.2, season_config=None, coverage_threshold=0.7,
                           censored_target_policy="exclude", ablate_groups=(),
                           validation_stations=None, test_stations=None,
                           allowed_chemical_features=None):
        if not 0 <= coverage_threshold <= 1:
            raise ValueError("coverage_threshold debe estar entre 0 y 1.")
        frame, chemical_features, report = self._cargar_muestras(censored_target_policy)
        frame = self._agregar_estacionalidad(frame, season_config)
        station_split = validation_stations is not None or test_stations is not None
        if station_split and (validation_stations is None or test_stations is None):
            raise ValueError("Leave-station-out requiere validation_stations y test_stations.")
        if station_split and (train_end is not None or validation_end is not None or test_start is not None or test_end is not None):
            raise ValueError("No mezcle cortes temporales con leave-station-out.")
        if not station_split and (train_end is None) != (validation_end is None):
            raise ValueError("Especifique ambos cortes: train_end y validation_end.")
        if station_split:
            validation_stations, test_stations = set(validation_stations), set(test_stations)
            if not validation_stations or not test_stations or validation_stations & test_stations:
                raise ValueError("Las estaciones de validación/prueba deben ser no vacías y disjuntas.")
            known = set(frame.station)
            unknown = (validation_stations | test_stations) - known
            if unknown:
                raise ValueError(f"Estaciones desconocidas en el split: {sorted(unknown)[:5]}")
            frame["split"] = np.select(
                [frame.station.isin(validation_stations), frame.station.isin(test_stations)],
                ["validation", "test"], default="train"
            )
            train_end = validation_end = test_end = None
        elif train_end is None:
            if not (0 < train_ratio < 1 and 0 < validation_ratio < 1 and train_ratio + validation_ratio < 1):
                raise ValueError("Las proporciones deben ser positivas y sumar menos de 1.")
            date_mask = frame.y_true.notna()
            if censored_target_policy == 'train_half_limit':
                date_mask &= ~frame.target_censored
            days = np.sort(frame.loc[date_mask, "date"].dt.normalize().unique())
            train_n, validation_n = int(len(days) * train_ratio), int(len(days) * (train_ratio + validation_ratio))
            if not 0 < train_n < validation_n < len(days):
                raise ValueError("No hay fechas suficientes para tres particiones temporales.")
            train_end, validation_end = pd.Timestamp(days[train_n - 1]), pd.Timestamp(days[validation_n - 1])
        else:
            train_end, validation_end = pd.Timestamp(train_end).normalize(), pd.Timestamp(validation_end).normalize()
        if not station_split:
            test_start = pd.Timestamp(test_start).normalize() if test_start is not None else None
            test_end = pd.Timestamp(test_end).normalize() if test_end is not None else None
            if (pd.isna(train_end) or pd.isna(validation_end) or train_end >= validation_end
                    or (test_start is not None and (pd.isna(test_start) or validation_end >= test_start))
                    or (test_end is not None and (pd.isna(test_end) or validation_end >= test_end))):
                raise ValueError("Los cortes deben satisfacer train_end < validation_end < test_start/test_end.")
            if test_start is not None and test_end is not None and test_start > test_end:
                raise ValueError("test_start debe ser anterior o igual a test_end.")
            dates = frame.date.dt.normalize()
            if test_start is None and test_end is None:
                frame["split"] = np.select([dates <= train_end, dates <= validation_end], ["train", "validation"], default="test")
            else:
                test_mask = dates >= (test_start if test_start is not None else validation_end + pd.Timedelta(days=1))
                if test_end is not None:
                    test_mask &= dates <= test_end
                frame["split"] = np.select([dates <= train_end, dates <= validation_end, test_mask],
                                            ["train", "validation", "test"], default="outside_evaluation")
        if censored_target_policy == 'train_half_limit':
            # Se rescatan etiquetas de entrenamiento, nunca verdad de evaluación.
            frame.loc[frame.target_censored & frame.split.ne('train'), 'y_true'] = np.nan
            for lag in (1,2,3):
                frame[f'dqo_lag_{lag}'] = frame.groupby('station').y_true.transform(lambda s:s.ffill().shift(lag))
            previous = frame.date.where(frame.y_true.notna()).groupby(frame.station).transform(lambda s:s.ffill().shift())
            frame['days_since_previous_dqo'] = (frame.date-previous).dt.total_seconds()/86400
            report['visits_with_usable_target'] = int(frame.y_true.notna().sum())
        frame["eligible"] = frame.y_true.notna() & (frame.history_count >= self.sequence_length - 1)
        frame["exclusion_reason"] = np.select(
            [frame.target_censored & frame.y_true.isna(), frame.y_true.isna(), frame.history_count < self.sequence_length - 1],
            ["censored_target", "missing_target", "insufficient_history"], default="")
        for split in ("train", "validation", "test"):
            if not (frame.split.eq(split) & frame.eligible).any():
                raise ValueError(f"Partición {split} sin muestras elegibles; revise cortes/sequence_length/censura.")
        train = frame.split.eq("train")
        coverage = frame.loc[train, chemical_features].notna().mean()
        selected = coverage[(coverage >= coverage_threshold) & (coverage > 0)].index.tolist()
        selected_before_availability = list(selected)
        if allowed_chemical_features is not None:
            allowed = set(allowed_chemical_features)
            unknown = allowed - set(chemical_features)
            if unknown:
                raise ValueError(f"Predictores químicos desconocidos: {sorted(unknown)}")
            selected = [feature for feature in selected if feature in allowed]
        if isinstance(ablate_groups, str):
            ablate_groups = (ablate_groups,)
        ablate_groups = tuple(ablate_groups)
        allowed_ablation_groups = {"historical_dqo", "geography", "time", "contemporary_covariates"}
        unknown = set(ablate_groups) - allowed_ablation_groups
        if unknown:
            raise ValueError(f"Grupos de ablación desconocidos: {sorted(unknown)}")
        feature_groups = {
            "geography": ["latitude", "longitude", "elevation"],
            "time": ["year", "month_sin", "month_cos", "gap_days", "history_count"],
            "historical_dqo": ["days_since_previous_dqo", "dqo_lag_1", "dqo_lag_2", "dqo_lag_3"],
            "contemporary_covariates": selected + [f"{c}__censored" for c in selected],
        }
        base_features = [feature for group, features in feature_groups.items()
                         if group not in ablate_groups for feature in features]
        if not base_features:
            raise ValueError("La ablación no puede eliminar todos los predictores.")
        raw = frame[base_features].astype(float).replace([np.inf, -np.inf], np.nan)
        self.imputation_values_ = raw.loc[train].median().fillna(0.0)
        predictors = pd.concat([raw.fillna(self.imputation_values_), raw.isna().astype(float).add_suffix("__missing")], axis=1)
        self.feature_cols = predictors.columns.tolist()
        self.scaler.fit(predictors.loc[train])
        values = self.scaler.transform(predictors).astype(np.float32)
        fit_labels = frame.loc[train & frame.eligible, "y_true"].to_numpy(float)
        self.target_scaler.fit((np.log1p(fit_labels) if self.transformar_target_log else fit_labels).reshape(-1, 1))
        endpoints, contexts = [], []
        for _, group in frame.groupby("station", sort=False):
            indices = group.index.to_numpy()
            for offset in range(self.sequence_length - 1, len(indices)):
                endpoint = indices[offset]
                if frame.at[endpoint, "eligible"]:
                    endpoints.append(endpoint)
                    contexts.append(indices[offset - self.sequence_length + 1:offset + 1])
        order = sorted(range(len(endpoints)), key=lambda i: (frame.at[endpoints[i], "date"], frame.at[endpoints[i], "station"]))
        endpoints, contexts = np.asarray(endpoints)[order], np.asarray(contexts)[order]
        partitions = {}
        for split in ("train", "validation", "test"):
            mask = frame.loc[endpoints, "split"].eq(split).to_numpy()
            selected_endpoints, selected_contexts = endpoints[mask], contexts[mask]
            metadata = frame.loc[selected_endpoints, METADATA_COLUMNS].reset_index(drop=True).copy()
            target_values = metadata.y_true.to_numpy(float)
            if self.transformar_target_log:
                target_values = np.log1p(target_values)
            partitions[split] = Partition(values[selected_endpoints], values[selected_contexts],
                self.target_scaler.transform(target_values.reshape(-1, 1)).ravel().astype(np.float32), metadata)
        self.sequence_indices_, self.sequence_endpoints_ = contexts, endpoints
        split_config = {"strategy": "leave_station_out" if station_split else "global_chronological_holdout",
                        "train_end": train_end.isoformat() if train_end is not None else None,
                        "validation_end": validation_end.isoformat() if validation_end is not None else None,
                        "sequence_length": self.sequence_length,
                        "test_start": test_start.isoformat() if test_start is not None else None,
                        "test_end": test_end.isoformat() if test_end is not None else None,
                        "same_date_kept_together": True, "target_log1p": self.transformar_target_log,
                        "coverage_threshold": coverage_threshold, "censored_target_policy": censored_target_policy,
                        "ablated_feature_groups": list(ablate_groups),
                        "chemical_availability_filter_applied": allowed_chemical_features is not None,
                        "allowed_chemical_features": sorted(allowed_chemical_features) if allowed_chemical_features is not None else None,
                        "test_history_policy": "rolling_observed_past_results; no parameter refit",
                        "validation_station_count": len(validation_stations) if station_split else None,
                        "test_station_count": len(test_stations) if station_split else None}
        report.update(selected_chemical_features=selected,
                      selected_chemical_features_before_availability=selected_before_availability,
                      excluded_by_availability=sorted(set(selected_before_availability)-set(selected)),
                      chemical_availability_filter_applied=allowed_chemical_features is not None,
                      feature_count=len(self.feature_cols),
                      feature_groups=feature_groups, ablated_feature_groups=list(ablate_groups),
                      censored_training_labels_used=int(partitions['train'].metadata.target_censored.sum()),
                      censored_evaluation_labels_used=int(sum(partitions[s].metadata.target_censored.sum() for s in ('validation','test'))),
                      train_coverage={str(k): float(v) for k, v in coverage.items()},
                      features_all_missing_in_train=raw.columns[raw.loc[train].isna().all()].tolist(),
                      season_sources=sorted(frame.season_source.unique().tolist()),
                      eligible_counts={s: len(p.y) for s, p in partitions.items()}, excluded_visits=int((~frame.eligible).sum()),
                      target_min=float(frame.y_true.min()), target_max=float(frame.y_true.max()),
                      warnings=["Climate seasons are calendar labels, not observed rainfall.",
                                "Spatial metrics describe monitored stations, not transfer to unseen stations.",
                                "Validation residuals may be optimistic after model selection; error attribution is post hoc."])
        self.prepared = PreparedData(partitions, self.feature_cols, frame, report, split_config,
                                     self.target_scaler, self.transformar_target_log)
        return self.prepared

    def desescalar_target(self, y):
        if self.prepared is None:
            raise RuntimeError("Primero debe preparar los datos.")
        return self.prepared.inverse_target(y)

    inverse_target = desescalar_target

    def preparar_datos_supervisado(self, train_ratio=0.6, escalar=True):
        if not escalar:
            raise ValueError("La API de compatibilidad requiere escalar=True; use preparar_evaluacion.")
        data = self.preparar_evaluacion(train_ratio=train_ratio, validation_ratio=(1 - train_ratio) / 2)
        train, test = data.partitions["train"], data.partitions["test"]
        return train.X, test.X, train.y, test.y

    def preparar_datos_secuenciales(self, sequence_length=None, train_ratio=0.6, escalar=True):
        if sequence_length is not None:
            self.sequence_length = sequence_length
        self.preparar_datos_supervisado(train_ratio=train_ratio, escalar=escalar)
        train, test = self.prepared.partitions["train"], self.prepared.partitions["test"]
        return train.X_sequence, test.X_sequence, train.y, test.y

    def obtener_dataset_tabular(self):
        prepared = self.prepared or self.preparar_evaluacion()
        # Includes raw chemical columns; transformed arrays are in partitions.
        return prepared.frame.copy(), prepared.feature_cols, "y_true"

    procesar = obtener_dataset_tabular
