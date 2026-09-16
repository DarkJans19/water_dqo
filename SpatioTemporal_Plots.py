"""Figuras comparables por modelo sin descargas cartográficas ni interfaz gráfica."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


def _evaluation_rows(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if frame.empty or "split" not in frame:
        return frame.copy()
    return frame.loc[frame["split"].eq(split)].copy()


def _save(fig, path: Path, paths: list[str]):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path))


def generar_visualizaciones(
    predictions: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[str]:
    """Comparar modelos en prueba (o validación si no hay prueba).

    Los mapas muestran coordenadas observadas con escalas comunes entre modelos.
    No interpolan errores a zonas sin observaciones. Las celdas sin evaluación
    permanecen vacías. Las métricas provienen de las mismas tablas exportadas.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if predictions.empty:
        return paths
    split = "test" if predictions["split"].eq("test").any() else "validation"
    rows = _evaluation_rows(predictions, split)
    models = sorted(rows["model"].dropna().astype(str).unique())
    if not models:
        return paths
    station = _evaluation_rows(metric_tables.get("station", pd.DataFrame()), split)
    if 'R2_reported' in station:
        station = station.copy()
        station['R2'] = station['R2_reported']
    year = _evaluation_rows(metric_tables.get("year", pd.DataFrame()), split)
    season = _evaluation_rows(metric_tables.get("season", pd.DataFrame()), split)
    station_year = _evaluation_rows(metric_tables.get("station_year", pd.DataFrame()), split)
    colors = {name: plt.get_cmap("tab10")(index % 10) for index, name in enumerate(models)}
    for metric in ("MAE", "RMSE", "R2"):
        if not station.empty and {"longitude", "latitude", metric}.issubset(station):
            located = station.replace([np.inf, -np.inf], np.nan).dropna(subset=["longitude", "latitude", metric])
            if not located.empty:
                low = min(0.0, float(located[metric].min())) if metric == "R2" else 0.0
                high = 1.0 if metric == "R2" else max(float(located[metric].max()), 1e-12)
                norm = Normalize(vmin=low, vmax=high)
                fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), squeeze=False, sharex=True, sharey=True, layout="constrained")
                for model_name, ax in zip(models, axes[0]):
                    part = located.loc[located["model"].eq(model_name)]
                    sizes = 25 + 45 * np.sqrt(part["N"] / max(float(located["N"].max()), 1)) if "N" in part else 40
                    ax.scatter(part["longitude"], part["latitude"], c=part[metric], cmap="viridis" if metric != "R2" else "viridis_r", norm=norm, s=sizes, edgecolors="black", linewidths=0.3)
                    ax.set(title=f"{model_name} ({len(part)} estaciones)", xlabel="Longitud", ylabel="Latitud")
                    ax.grid(alpha=0.2)
                    if part.empty:
                        ax.text(0.5, 0.5, "Sin coordenadas/métrica válidas", ha="center", transform=ax.transAxes)
                fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis" if metric != "R2" else "viridis_r"), ax=axes[0].tolist(), label=metric)
                fig.suptitle(f"{metric} por estación · {split} · tamaño proporcional a √N")
                _save(fig, directory / f"map_station_{metric}.png", paths)
        if not year.empty and metric in year:
            fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
            for model_name in models:
                part = year.loc[year["model"].eq(model_name)].sort_values("year")
                ax.plot(part["year"], part[metric], marker="o", label=model_name, color=colors[model_name])
            ax.set(title=f"{metric} anual · {split}", xlabel="Año", ylabel=metric)
            ax.set_xticks(sorted(year["year"].dropna().unique()))
            ax.grid(alpha=0.25)
            ax.legend()
            _save(fig, directory / f"year_{metric}.png", paths)
        if not season.empty and metric in season:
            labels = sorted(season["season"].dropna().astype(str).unique())
            if labels:
                fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 4.5), layout="constrained")
                width = 0.8 / len(models)
                for index, model_name in enumerate(models):
                    part = season.loc[season["model"].eq(model_name)].set_index("season")
                    values = part[metric].reindex(labels)
                    positions = np.arange(len(labels)) + (index - (len(models) - 1) / 2) * width
                    ax.bar(positions, values, width=width, label=model_name, color=colors[model_name])
                ax.set_xticks(np.arange(len(labels)), labels, rotation=15)
                ax.set(title=f"{metric} por temporada · {split} · calendario configurado", ylabel=metric)
                ax.grid(axis="y", alpha=0.25)
                ax.legend()
                _save(fig, directory / f"season_{metric}.png", paths)
    if not station_year.empty and "MAE" in station_year:
        stations = sorted(station_year["station"].dropna().astype(str).unique())
        years = sorted(station_year["year"].dropna().unique())
        valid_mae = pd.to_numeric(station_year["MAE"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if stations and years and not valid_mae.empty:
            norm = Normalize(vmin=0, vmax=max(float(valid_mae.max()), 1e-12))
            # Paginación mantiene legibles todas las estaciones sin descartarlas.
            for page_start in range(0, len(stations), 40):
                page = stations[page_start:page_start + 40]
                fig, axes = plt.subplots(1, len(models), figsize=(max(6, len(years) * 0.55) * len(models), max(4, len(page) * 0.26)), squeeze=False, sharey=True, layout="constrained")
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("#eeeeee")
                for model_name, ax in zip(models, axes[0]):
                    part = station_year.loc[station_year["model"].eq(model_name)].copy()
                    part["station"] = part["station"].astype(str)
                    matrix = part.pivot_table(index="station", columns="year", values="MAE", aggfunc="first").reindex(index=page, columns=years)
                    ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), aspect="auto", cmap=cmap, norm=norm)
                    ax.set_xticks(np.arange(len(years)), [str(int(y)) for y in years], rotation=45)
                    ax.set_yticks(np.arange(len(page)), page, fontsize=7)
                    ax.tick_params(axis="y", labelleft=ax is axes[0][0])
                    ax.set(title=model_name, xlabel="Año", ylabel="Estación" if ax is axes[0][0] else "")
                fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0].tolist(), label="MAE")
                fig.suptitle(f"MAE estación × año · {split} · gris = sin evaluación")
                _save(fig, directory / f"station_year_MAE_{page_start // 40 + 1:02d}.png", paths)
    for factor, label in (("gap_days", "Días desde el registro anterior"), ("elevation", "Elevación (m)")):
        if factor not in rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
        plotted = False
        for model_name in models:
            part = rows.loc[rows["model"].eq(model_name), [factor, "absolute_error"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if part.empty:
                continue
            # Muestreo reproducible solo para legibilidad; tablas conservan todas las filas.
            if len(part) > 3000:
                part = part.sample(3000, random_state=42)
            ax.scatter(part[factor], part["absolute_error"], alpha=0.3, s=12, label=model_name, color=colors[model_name])
            plotted = True
        if plotted:
            ax.set(title=f"Error absoluto frente a {label.lower()} · {split}", xlabel=label, ylabel="Error absoluto DQO")
            ax.grid(alpha=0.25)
            ax.legend()
            _save(fig, directory / f"error_vs_{factor}.png", paths)
        else:
            plt.close(fig)
    return paths
