"""Comparación controlada de rescate de censura y regularización LSTM.

Ejecutar: python Experiment_Proposals.py. Sin CSV/JSON; devuelve DataFrames.
Prueba ya inspeccionada en el desarrollo: confirmar después con nuevos períodos.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from Data_Manage import Data_Manage
from XGBoost_Algorithm import XGBoost_Algorithm
from SVM_Algorithm import SVM_Algorithm
from LSTM_Algorithm import LSTM_Algorithm
from Performance_Diagnostics import resumen_metricas
from SpatioTemporal_Evaluation import construir_predicciones, metricas_desagregadas, validar_cohorte
from Regional_Evaluation import metricas_regionales


def ejecutar_experimento(csv='Data_historica_de_calidad_de_agua_20260223.csv', epochs=60, seed=42):
    cuts=dict(train_end='2015-08-11',validation_end='2020-02-06')
    candidates, trials, counts = {}, [], []
    reference=None
    for policy in ('exclude','train_half_limit'):
        data=Data_Manage(csv,sequence_length=5).preparar_evaluacion(**cuts,censored_target_policy=policy)
        data_xgb=Data_Manage(csv,sequence_length=5,transformar_target_log=False).preparar_evaluacion(
            **cuts,censored_target_policy=policy,coverage_threshold=.4)
        if reference is None:
            reference=data
        for d in (data,data_xgb):
            for split in ('validation','test'):
                pd.testing.assert_frame_equal(d.partitions[split].metadata,reference.partitions[split].metadata)
        counts.append(dict(policy=policy,train_N=len(data.partitions['train'].y),
            train_censored_N=int(data.partitions['train'].metadata.target_censored.sum()),
            validation_N=len(data.partitions['validation'].y),test_N=len(data.partitions['test'].y)))
        configs=[('XGBoost',XGBoost_Algorithm(random_state=seed),data_xgb),
                 ('SVM',SVM_Algorithm(random_state=seed),data)]
        for name,units,dropout,physical in [('LSTM_original',64,.1,False),
            ('LSTM_RMSE',64,.1,True),('LSTM_32_d02',32,.2,True),('LSTM_32_d03',32,.3,True)]:
            configs.append((name,LSTM_Algorithm(units=units,dropout=dropout,
                monitor_original_rmse=physical,epochs=epochs,random_state=seed),data))
        for name,model,prepared in configs:
            key=f'{name}__{policy}'
            print(f'Entrenando {key}...',flush=True)
            model.fit(prepared,inverse_target=prepared.inverse_target)
            valid=prepared.partitions['validation']
            prediction=np.maximum(prepared.inverse_target(model.predict(valid)),0)
            metrics=resumen_metricas(valid.metadata.y_true,prediction)
            trials.append(dict(candidate=key,model=name,policy=policy,seed=seed,**metrics,
                best_epoch=model.training_report.get('best_epoch'),epochs_run=model.training_report.get('epochs_run')))
            candidates[key]=(model,prepared)
            print(f"Validación RMSE={metrics['RMSE']:.3f}; R²={metrics['R2']:.3f}",flush=True)
    validation=pd.DataFrame(trials)
    # Congelar selección ANTES de pedir cualquier predicción de prueba.
    validation['selected_within_family']=False
    for policy in ('exclude','train_half_limit'):
        for family in ('XGBoost','SVM','LSTM'):
            group=validation[validation.policy.eq(policy)&validation.model.str.startswith(family)]
            validation.loc[group.RMSE.idxmin(),'selected_within_family']=True
    blocks, comparisons=[] , []
    for row in validation.to_dict('records'):
        model,data=candidates[row['candidate']]
        part=data.partitions['test']
        pred=np.maximum(data.inverse_target(model.predict(part)),0)
        block=construir_predicciones(part.metadata,part.metadata.y_true.to_numpy(),pred,row['candidate'],'test')
        blocks.append(block)
        comparisons.append(dict(candidate=row['candidate'],policy=row['policy'],
            selected_within_family=row['selected_within_family'],**resumen_metricas(part.metadata.y_true,pred)))
    predictions=pd.concat(blocks,ignore_index=True)
    validar_cohorte(predictions)
    result=dict(counts=pd.DataFrame(counts),validation=validation,test=pd.DataFrame(comparisons),
        predictions=predictions,metrics=metricas_desagregadas(predictions),
        regional_metrics=metricas_regionales(predictions),training={k:v[0].training_report for k,v in candidates.items()})
    return result


def escribir_informe(result,path='HISTORIAL_EXPERIMENTO_TEMPORAL.md'):
    pieces=['# Censura, regiones y regularización: experimento controlado',
        'Semilla 42, mismos cortes y 974 etiquetas exactas de prueba. Cada límite se divide por dos solo en entrenamiento. No se evalúa contra etiquetas inventadas. La política también afecta la historia de entrenamiento, no solo el número de ejemplos.',
        'XGBoost conserva la configuración previamente seleccionada (sin log, cobertura 40 %, profundidad 5). SVM conserva su búsqueda temporal. Se prueban cuatro LSTM por política; selección exclusivamente por RMSE de validación. Todos los resultados de prueba se muestran, incluso si empeoran.',
        'Agrupar regiones no mejora las predicciones: cambia la escala del diagnóstico. R2_reported regional requiere N>=20, >=2 estaciones y DQO no constante. Son umbrales descriptivos, no garantías de representatividad estadística.',
        'R2_within_regions = 1 - suma(SSE regional)/suma(SST regional), sobre regiones elegibles; se informa cuántas muestras excluye. No equivale al R² global convencional y no debe sustituirlo. Los códigos de zona/subzona provienen del CSV, no de cuencas inferidas.',
        'Prueba ya examinada durante desarrollo; una semilla y un corte temporal no demuestran robustez. Se requiere confirmación con otros períodos y semillas antes de adoptar cambios.',
    ]
    for name in ('counts','validation','test'):
        pieces.extend([f'## {name}','```text',result[name].to_string(index=False,float_format=lambda x:f'{x:.4f}'),'```'])
    chosen=result['validation'].query('selected_within_family').candidate.tolist()
    for name,table in result['regional_metrics'].items():
        shown=table[table.model.isin(chosen)].drop(columns=['SSE','SST','R2'],errors='ignore')
        pieces.extend([f'## Regiones: {name}','```text',shown.to_string(index=False,float_format=lambda x:f'{x:.4f}'),'```'])
    Path(path).write_text('\n\n'.join(pieces),encoding='utf-8')


def leer_tabla_informe(section, path='HISTORIAL_DESARROLLO.md'):
    """Reabrir tablas numéricas del informe legible sin exportaciones CSV/JSON.

    Para regiones con nombres que contienen espacios, consultar el informe o
    los DataFrames devueltos por ejecutar_experimento, no este lector compacto.
    """
    import io
    if section not in ('counts','validation','test','Regiones: within_regions'):
        raise ValueError('Sección numérica desconocida.')
    body=Path(path).read_text(encoding='utf-8').split(f'## {section}\n',1)[1]
    block=body.split('```text',1)[1].split('```',1)[0].strip()
    return pd.read_csv(io.StringIO(block),sep=r'\s+')


if __name__=='__main__':
    results=ejecutar_experimento()
    print(results['counts'].to_string(index=False))
    print(results['test'].to_string(index=False))
    escribir_informe(results)
