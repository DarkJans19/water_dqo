"""Métricas regionales descriptivas; no alteran entrenamiento ni error global."""
import numpy as np
import pandas as pd
from Performance_Diagnostics import resumen_metricas


def metricas_regionales(predictions, min_samples=20, min_stations=2):
    tables, weighted = {}, []
    for dimension in ('hydro_zone','hydro_subzone','altitude_band'):
        if dimension not in predictions:
            continue
        rows = []
        for (model,split,region), group in predictions.groupby(['model','split',dimension],dropna=False):
            m = resumen_metricas(group.y_true,group.y_pred)
            sst = float(np.square(group.y_true-group.y_true.mean()).sum())
            sse = float(np.square(group.y_pred-group.y_true).sum())
            nstations = group.station.nunique()
            reliable = m['N'] >= min_samples and nstations >= min_stations and sst > 0 and region not in ('sin_dato','conflicto') and pd.notna(region)
            rows.append(dict(model=model,split=split,region=region,**m,n_stations=nstations,
                             R2_reported=m['R2'] if reliable else np.nan, representative=reliable, SST=sst,SSE=sse))
        table=pd.DataFrame(rows)
        tables[dimension]=table
        for (model,split),g in table.groupby(['model','split']):
            usable=g[g.representative]
            # Equivale a ponderar R² regional por SST=N*varianza poblacional.
            denom=usable.SST.sum()
            weighted.append(dict(model=model,split=split,grouping=dimension,
                R2_within_regions=1-usable.SSE.sum()/denom if denom>0 else np.nan,
                N_included=int(usable.N.sum()),N_excluded=int(g.N.sum()-usable.N.sum()),
                regions_included=len(usable),min_samples=min_samples,min_stations=min_stations))
    tables['within_regions']=pd.DataFrame(weighted)
    return tables
