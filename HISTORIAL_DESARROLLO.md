# Historial de desarrollo y experimentos

Este es el único documento de historial del proyecto. El flujo vigente está en
`README.md` y el análisis visual en `Analisis_DQO.ipynb`.

## Protocolo estable

- 134.261 filas largas se convierten en 6.835 combinaciones estación-fecha.
- Cortes: train hasta 2015-08-11, validación hasta 2020-02-06 y test posterior.
- Cohorte exacta: 2.679 train, 877 validación y 974 test.
- El preprocesamiento se ajusta solo con train; validación selecciona; test informa.
- Test ya fue inspeccionado durante el desarrollo. Nuevas mejoras requieren otro
  período o varios orígenes temporales para considerarse confirmadas.

## Resultados y decisiones

| Modelo/configuración | MAE test | RMSE test | R² test | Decisión |
|---|---:|---:|---:|---|
| XGBoost original | 18,499 | 45,754 | 0,481 | Reemplazado |
| XGBoost sin log, cobertura 40 %, profundidad 5 | 17,690 | 39,576 | 0,612 | Conservado |
| LSTM original, 64 unidades/dropout 0,1 | 19,767 | 45,005 | 0,498 | Referencia |
| LSTM 32 unidades/dropout 0,3/RMSE físico | 18,751 | 42,774 | 0,546 | Conservado |
| SVM | 22,649 | 51,709 | 0,337 | Conservado para comparación |

XGBoost se seleccionó entre 12 candidatos usando exclusivamente RMSE de
validación. La LSTM regularizada ganó entre cuatro configuraciones de su familia
en validación. Que otra configuración tuviera mejor test no cambió la selección.

## Intento que no aportó: mitad del límite de detección

`train_half_limit` aproxima cada etiqueta `<L` por `L/2` únicamente en train.
Rescató 718 ejemplos y aumentó train de 2.679 a 3.397, manteniendo las mismas
877 etiquetas exactas de validación y 974 de test.

| Modelo | RMSE validación sin rescate | Con rescate | RMSE test sin rescate | Con rescate |
|---|---:|---:|---:|---:|
| XGBoost | 35,451 | 36,391 | 39,576 | 40,936 |
| SVM | 49,958 | 51,373 | 51,709 | 53,280 |
| LSTM 32/dropout 0,3 | 37,716 | 38,319 | 42,774 | 41,294 |

La solución **no aportó aprendizaje neto**: empeoró validación en los tres
modelos, además de empeorar test en XGBoost y SVM. La aparente mejora de test en
LSTM no se adopta porque su validación fue peor; hacerlo seleccionaría usando
test. La opción permanece reproducible para análisis de sensibilidad, pero el
flujo recomendado excluye censura. Una continuación razonable es una pérdida
censurada/Tobit o de intervalo, evaluada con múltiples cortes.

## Intento que no aportó: pesos y perfil regional

Se comprobó la propuesta de priorizar la subzona `2,120`, que contiene sectores
del río Bogotá y Soacha. No se eligió por los datos de prueba: en entrenamiento
tenía 215 ejemplos de nueve estaciones y una DQO media de 98,30 mg O2/L; en
validación tenía 83 ejemplos. El ID de estación ya estaba excluido de los
predictores antes del experimento.

Se compararon en las mismas 877 observaciones de validación: pesos 2, 3 y 5 para
la subzona `2,120`; codificación de zona, subzona y banda altitudinal agrupando
categorías con menos de 20 ejemplos de entrenamiento; y cuatro configuraciones
de XGBoost con menor profundidad y penalizaciones L1/L2 más fuertes.

| Variante XGBoost | RMSE global validación | RMSE subzona 2,120 |
|---|---:|---:|
| Modelo conservado | **35,451** | **77,830** |
| Peso 2 | 36,840 | 84,189 |
| Peso 3 | 37,911 | 86,772 |
| Peso 5 | 37,844 | 88,471 |
| Perfil regional como predictor | 37,355 | 84,443 |
| Mejor regularización adicional (profundidad 4) | 36,570 | 81,847 |

El modelo conservado ganó tanto con RMSE global como con un RMSE de validación
que ponderaba por cinco la subzona prioritaria. Por ello no se consultó prueba
para ordenar las alternativas y no se incorporaron al pipeline. Los pesos no
crean información nueva: cambian el costo de los errores y, en este caso,
aumentaron incluso el error de la región priorizada. La regionalización continúa
en la evaluación, donde sí ayuda a localizar el problema sin degradar el modelo.

La mitigación LSTM ya adoptada permanece: 32 unidades, dropout 0,3, parada
temprana sobre RMSE físico y restauración del mejor peso. Aumentar dropout sin
evidencia adicional no se adopta, porque también puede provocar subajuste.

## Evaluación regional

Se añadieron zona y subzona hidrográfica informadas por el CSV y bandas de
elevación. `R2_reported` exige al menos 20 casos, dos estaciones y objetivo no
constante. `R2_within_regions = 1 - suma(SSE) / suma(SST)` pondera por varianza y
reporta exclusiones; no sustituye al R² global. Agrupar mejora el diagnóstico,
no las predicciones.

| Modelo | Agrupación | R² dentro de regiones | N incluido | N excluido |
|---|---|---:|---:|---:|
| XGBoost seleccionado | Zona | 0,609 | 856 | 118 |
| XGBoost seleccionado | Subzona | 0,528 | 562 | 412 |
| XGBoost seleccionado | Altitud | 0,608 | 974 | 0 |
| LSTM regularizada | Zona | 0,539 | 856 | 118 |
| LSTM regularizada | Subzona | 0,447 | 562 | 412 |
| LSTM regularizada | Altitud | 0,542 | 974 | 0 |

## SHAP y análisis del error

SHAP explica un Random Forest auxiliar del error absoluto, no directamente los
predictores DQO. Los auxiliares no mejoraron en MAE una referencia constante de
validación y se marcan con baja capacidad predictiva. Sus importancias son
exploratorias y no prueban causalidad.

## Próximos intentos

1. Orígenes temporales móviles y varias semillas.
2. Pérdida censurada/Tobit en lugar de sustitución fija.
3. Calendarios climáticos regionales o precipitación observada.
4. Leave-station-out para transferencia espacial.
5. Ablación de DQO histórica y covariables contemporáneas.
6. Calibración para picos, sin recortarlos ni elegir por test.

## Tablas compactas usadas por el notebook

## counts

```text
          policy  train_N  train_censored_N  validation_N  test_N
         exclude     2679                 0           877     974
train_half_limit     3397               718           877     974
```

## validation

```text
                      candidate         model           policy  seed   N     MAE    RMSE     R2  best_epoch  epochs_run  selected_within_family
               XGBoost__exclude       XGBoost          exclude    42 877 17.6468 35.4512 0.7016         NaN         NaN                    True
                   SVM__exclude           SVM          exclude    42 877 21.4770 49.9583 0.4074         NaN         NaN                    True
         LSTM_original__exclude LSTM_original          exclude    42 877 19.3778 42.7613 0.5658      7.0000     15.0000                   False
             LSTM_RMSE__exclude     LSTM_RMSE          exclude    42 877 19.4732 40.8671 0.6034     23.0000     31.0000                   False
           LSTM_32_d02__exclude   LSTM_32_d02          exclude    42 877 18.4846 38.2398 0.6528     12.0000     20.0000                   False
           LSTM_32_d03__exclude   LSTM_32_d03          exclude    42 877 18.6137 37.7156 0.6622     12.0000     20.0000                    True
      XGBoost__train_half_limit       XGBoost train_half_limit    42 877 17.2403 36.3905 0.6855         NaN         NaN                    True
          SVM__train_half_limit           SVM train_half_limit    42 877 22.0797 51.3726 0.3733         NaN         NaN                    True
LSTM_original__train_half_limit LSTM_original train_half_limit    42 877 20.9708 44.3931 0.5320      9.0000     17.0000                   False
    LSTM_RMSE__train_half_limit     LSTM_RMSE train_half_limit    42 877 20.7236 43.4188 0.5524      8.0000     16.0000                   False
  LSTM_32_d02__train_half_limit   LSTM_32_d02 train_half_limit    42 877 19.2541 38.8655 0.6413      8.0000     16.0000                   False
  LSTM_32_d03__train_half_limit   LSTM_32_d03 train_half_limit    42 877 19.2706 38.3193 0.6513      6.0000     14.0000                    True
```

## test

```text
                      candidate           policy  selected_within_family   N     MAE    RMSE     R2
               XGBoost__exclude          exclude                    True 974 17.6902 39.5762 0.6115
                   SVM__exclude          exclude                    True 974 22.6490 51.7085 0.3369
         LSTM_original__exclude          exclude                   False 974 19.7672 45.0055 0.4977
             LSTM_RMSE__exclude          exclude                   False 974 19.7174 41.7690 0.5673
           LSTM_32_d02__exclude          exclude                   False 974 18.6383 41.8263 0.5661
           LSTM_32_d03__exclude          exclude                    True 974 18.7507 42.7737 0.5462
      XGBoost__train_half_limit train_half_limit                    True 974 17.7237 40.9357 0.5844
          SVM__train_half_limit train_half_limit                    True 974 22.8780 53.2800 0.2960
LSTM_original__train_half_limit train_half_limit                   False 974 20.3378 44.9714 0.4984
    LSTM_RMSE__train_half_limit train_half_limit                   False 974 20.3665 45.0021 0.4977
  LSTM_32_d02__train_half_limit train_half_limit                   False 974 18.9887 41.9319 0.5639
  LSTM_32_d03__train_half_limit train_half_limit                    True 974 18.6114 41.2944 0.5771
```

## Regiones: within_regions

```text
                  model split      grouping  R2_within_regions  N_included  N_excluded  regions_included  min_samples  min_stations
 LSTM_32_d03__exclude  test    hydro_zone             0.5388         856         118                 6           20             2
         SVM__exclude  test    hydro_zone             0.3170         856         118                 6           20             2
     XGBoost__exclude  test    hydro_zone             0.6090         856         118                 6           20             2
 LSTM_32_d03__exclude  test hydro_subzone             0.4469         562         412                16           20             2
         SVM__exclude  test hydro_subzone             0.1385         562         412                16           20             2
     XGBoost__exclude  test hydro_subzone             0.5282         562         412                16           20             2
 LSTM_32_d03__exclude  test altitude_band             0.5421         974           0                 2           20             2
         SVM__exclude  test altitude_band             0.3308         974           0                 2           20             2
     XGBoost__exclude  test altitude_band             0.6080         974           0                 2           20             2
```
