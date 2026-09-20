"""Auditoría operacional de covariables contemporáneas de DQO."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from Data_Manage import Data_Manage
from Diagnosis_Algorithms import DEFAULT_CSV
from Performance_Diagnostics import resumen_metricas
from SpatioTemporal_Evaluation import construir_predicciones, metricas_por_banda_dqo
from XGBoost_Algorithm import XGBoost_Algorithm


FIELD_CANDIDATE_TOKENS = (
    "CONDUCTIVIDAD ELECTRICA", "OXIGENO DISUELTO", "PH [", "TEMPERATURA", "TURBIDEZ",
)


def _base_data(csv):
    return Data_Manage(csv, sequence_length=5, transformar_target_log=False).preparar_evaluacion(
        train_end="2015-08-11", validation_end="2020-02-06",
        coverage_threshold=.4, censored_target_policy="exclude",
    )


def auditar_disponibilidad(archivo_csv=DEFAULT_CSV, manifest_path=None):
    """Liste cobertura y estado; no infiera disponibilidad real desde el nombre."""
    data = _base_data(archivo_csv)
    coverage = data.quality_report["train_coverage"]
    rows = []
    for feature in data.quality_report["selected_chemical_features_before_availability"]:
        rows.append({
            "feature": feature,
            "train_coverage": coverage[feature],
            "field_candidate_by_name_only": any(token in feature for token in FIELD_CANDIDATE_TOKENS),
            "availability_status": "unverified",
            "available_before_dqo": pd.NA,
            "evidence_source": "",
            "warning": "El nombre no demuestra método, hora de disponibilidad ni ausencia de fuga operacional.",
        })
    audit = pd.DataFrame(rows)
    if manifest_path is not None:
        manifest = pd.read_csv(manifest_path, dtype="string").fillna("")
        required = {"feature", "available_before_dqo", "evidence_source"}
        missing = required - set(manifest)
        if missing:
            raise ValueError(f"Faltan columnas del manifiesto: {sorted(missing)}")
        if manifest.feature.duplicated().any():
            raise ValueError("El manifiesto contiene predictores duplicados.")
        audit = audit.drop(columns=["available_before_dqo", "evidence_source"]).merge(
            manifest[["feature", "available_before_dqo", "evidence_source"]],
            on="feature", how="left", validate="one_to_one",
        )
        normalized = audit.available_before_dqo.fillna("").str.strip().str.lower()
        audit["availability_status"] = np.select(
            [normalized.isin(["true", "si", "sí", "1"]), normalized.isin(["false", "no", "0"])],
            ["verified_available", "verified_unavailable"], default="unverified",
        )
        claimed = audit.availability_status.eq("verified_available")
        if (claimed & audit.evidence_source.fillna("").str.strip().eq("")).any():
            raise ValueError("Todo predictor marcado disponible requiere evidence_source.")
    return audit


def predictores_operacionales_verificados(manifest_path, archivo_csv=DEFAULT_CSV):
    audit = auditar_disponibilidad(archivo_csv, manifest_path)
    unresolved = audit.loc[audit.availability_status.eq("unverified"), "feature"]
    if len(unresolved):
        raise ValueError(f"Quedan {len(unresolved)} predictores sin verificar; no se autoriza modelo operacional.")
    return audit.loc[audit.availability_status.eq("verified_available"), "feature"].tolist()


def ejecutar_sensibilidad_candidatos_campo(archivo_csv=DEFAULT_CSV, random_state=42, verbose=True):
    """Cuantifique el costo de usar sólo candidatos de campo; no los certifique."""
    baseline = _base_data(archivo_csv)
    candidates = [feature for feature in baseline.quality_report["selected_chemical_features"]
                  if any(token in feature for token in FIELD_CANDIDATE_TOKENS)]
    scenarios = {
        "XGBoost_all_contemporaneous_research": baseline,
        "XGBoost_field_candidates_unverified": Data_Manage(
            archivo_csv, sequence_length=5, transformar_target_log=False
        ).preparar_evaluacion(
            train_end="2015-08-11", validation_end="2020-02-06",
            coverage_threshold=.4, censored_target_policy="exclude",
            allowed_chemical_features=candidates,
        ),
    }
    rows, blocks = [], []
    for name, data in scenarios.items():
        if verbose:
            print(f"Entrenando {name} ({len(data.quality_report['selected_chemical_features'])} químicas)", flush=True)
        model = XGBoost_Algorithm(random_state=random_state, max_depth=5).fit(data)
        for split in ("validation", "test"):
            part = data.partitions[split]
            prediction = np.maximum(data.inverse_target(model.predict(part)), 0)
            rows.append({"model": name, "split": split,
                         "chemical_features": len(data.quality_report["selected_chemical_features"]),
                         **resumen_metricas(part.metadata.y_true, prediction)})
            blocks.append(construir_predicciones(part.metadata, part.metadata.y_true,
                                                  prediction, name, split))
    predictions = pd.concat(blocks, ignore_index=True)
    bands, boundaries = metricas_por_banda_dqo(
        predictions, baseline.partitions["train"].metadata.y_true
    )
    return {"audit": auditar_disponibilidad(archivo_csv), "candidate_features": candidates,
            "global": pd.DataFrame(rows), "dqo_band": bands,
            "dqo_band_boundaries": boundaries,
            "interpretation": "field candidates are name-based sensitivity only, not verified availability"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="DQO: auditoría de disponibilidad de predictores.")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--manifest")
    parser.add_argument("--run-sensitivity", action="store_true")
    args = parser.parse_args(argv)
    audit = auditar_disponibilidad(args.csv, args.manifest)
    print(audit.to_string(index=False))
    if args.run_sensitivity:
        result = ejecutar_sensibilidad_candidatos_campo(args.csv)
        print("\nSENSIBILIDAD; NO CERTIFICA DISPONIBILIDAD")
        print(result["global"].to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
