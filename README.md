# Evaluación espacio-temporal de DQO

El proyecto estima la Demanda Química de Oxígeno (DQO) en visitas de monitoreo
del IDEAM y compara el error de XGBoost, LSTM y SVM por estación, año y temporada.
El resultado principal es una tabla de **MAE, RMSE y R² en prueba**, impresa en
`Diagnosis_Algorithms.py`, accesible como DataFrame y presentada con figuras en
`Analisis_DQO.ipynb`. No se necesita dashboard ni archivos JSON.

## Cómo ejecutarlo y dónde mirar

### Experimento de censura, agrupación regional y regularización LSTM

`python Experiment_Proposals.py` ejecuta la comparación completa (12 entrenamientos)
y escribe un informe temporal que se contrasta con `HISTORIAL_DESARROLLO.md`.
También devuelve tablas en memoria mediante `ejecutar_experimento()`. El notebook
incluye una sección que muestra los resultados de ese informe y cómo repetirlos.

- **Censura:** `--censored-target-policy train_half_limit` usa la mitad del límite
  de cada registro `<L` solo como etiqueta de entrenamiento. `<10` pasa a 5 y
  `<20` a 10. `>L` no se rescata. Se reconstruyen las variables históricas para
  que las etiquetas censuradas futuras no se conviertan en DQO exacta. Con estos
  cortes se recuperan 718 etiquetas elegibles: train pasa de 2.679 a 3.397.
  Validación y prueba conservan las mismas 877 y 974 etiquetas exactas. No se
  recuperan las 1.566 censuradas para train porque muchas están fuera de su
  período o carecen de historial. La política `half_limit` anterior permanece
  como sensibilidad con etiquetas aproximadas también en evaluación; **no** se
  usa para comparar las mejoras de este experimento.
- **Regiones:** se conservan los campos de zona y código de subzona hidrográfica
  del CSV, además de bandas de elevación fijas (<500, 500–1499, 1500–2499, >=2500 m).
  `resultados['regional_metrics']` contiene los nuevos desgloses. `R2_reported`
  exige al menos 20 casos, dos estaciones y variación del objetivo en una región
  conocida. Son reglas descriptivas de soporte, no una prueba de representatividad.
  En las tablas locales `R2_reported` queda vacío por debajo de `min_samples` y
  los mapas no muestran ese R². `R2` conserva el cálculo matemático para auditoría.
- **R² regional ponderado:** se informa por separado `R2_within_regions`, ponderado
  por SST (= N × varianza poblacional), con N incluido/excluido. No es el R² global
  habitual: tiene otra referencia y puede excluir regiones con poco soporte.
  Agrupar regiones mejora el diagnóstico, no las predicciones.
- **LSTM:** `--regularize-lstm` configura 32 unidades, dropout 0,3 y parada temprana
  sobre RMSE en mg O2/L, con paciencia de ocho épocas y restauración de pesos.
  No se detiene ante la primera oscilación. Antes ya había dropout 0,1 y parada
  sobre pérdida del objetivo transformado. Se compararon 64/0,1 con ambos monitores
  y 32/0,2 y 32/0,3 con RMSE físico, bajo ambas políticas de censura.

La sustitución por mitad del límite es una hipótesis que puede introducir sesgo,
no un valor medido. Véanse las [consideraciones de EPA sobre no detectados](https://www.epa.gov/risk/regional-guidance-handling-chemical-concentration-data-near-detection-limit-risk-assessments)
y sus [limitaciones estadísticas](https://archive.epa.gov/epawaste/hazard/web/pdf/unified-guid-2.pdf).
R² con dos valores diferentes sí está definido matemáticamente, aunque resulte
inestable; [scikit-learn documenta su indefinición con menos de dos muestras](https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.r2_score.html).
La restauración de pesos y el monitor se implementan según
[EarlyStopping de TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).

**Resultado:** LSTM 32/0,3 sin rescate fue la mejor LSTM en validación entre ambas
políticas; en prueba bajó RMSE de 45,005 a 42,774 y subió R² de 0,498 a 0,546.
La sustitución empeoró XGBoost y SVM. Combinada con LSTM 32/0,3 dio mejor prueba
(RMSE 41,294) pero peor validación que no rescatar; no se adopta solo por esa
cifra de prueba. Se conservan ambos experimentos, con semilla 42 y un único corte.
Son resultados de desarrollo, no confirmación independiente ni garantía de
generalización. Las opciones originales permanecen reproducibles.

```powershell
# Comparar las tres propuestas y ver todas las variantes:
.\.venv\Scripts\python.exe -X utf8 Experiment_Proposals.py
# Usar la LSTM regularizada junto a la búsqueda XGBoost ya implementada:
.\.venv\Scripts\python.exe -X utf8 Diagnosis_Algorithms.py --improve-xgboost --regularize-lstm
# Probar específicamente el rescate, sin cambiar la verdad de evaluación:
.\.venv\Scripts\python.exe -X utf8 Diagnosis_Algorithms.py --censored-target-policy train_half_limit
```

**Actualización: mejora de XGBoost.** `--improve-xgboost` compara doce candidatos
usando exclusivamente validación: con/sin logaritmo, cobertura mínima de 70/40/15 %
y profundidad de árbol 3/5. Selecciona el menor RMSE en mg O2/L y conserva sus
transformaciones. Después evalúa al ganador en las mismas 974 observaciones.
No aumenta artificialmente el conjunto ni incorpora etiquetas de prueba al ajuste.
La configuración original sigue siendo reproducible sin esa opción.

```powershell
.\.venv\Scripts\python.exe -X utf8 Diagnosis_Algorithms.py --improve-xgboost
```

En esta ejecución ganó DQO sin logaritmo, cobertura mínima 40 % y profundidad 5:
MAE 17,690, RMSE 39,576 y R² 0,612 en prueba. El resultado anterior de XGBoost era
MAE 18,499, RMSE 45,754 y R² 0,481. Ver `HISTORIAL_DESARROLLO.md` y la sección final del
notebook. **Prueba ya fue inspeccionada durante el desarrollo**: aunque la búsqueda
no la usa para ordenar candidatos, esta mejora aún requiere confirmación independiente.
Cuando se combina con LSTM/SVM, estos conservan su configuración; el preprocesamiento
de cada modelo está en `resultados['model_prepared']` y los candidatos en
`resultados['selection_trials']`. `prepared` conserva la preparación de referencia.

### Por qué no hay 134 mil muestras de prueba

El CSV ya contiene exactamente **6.835 combinaciones distintas de estación y
fecha/hora**, antes de filtrar etiquetas. Cada combinación reúne, en promedio,
19,64 filas de propiedades. Aquí se llama "visita" a esa unidad de agrupación;
no implica haber comprobado una visita física de personal.

De esas 6.835 combinaciones, 6.680 contienen algún registro de DQO y 155 no.
Con la política de censura quedan 5.114 etiquetas utilizables. Otras 584 no tienen
historial suficiente para terminar una secuencia de cinco visitas; los registros
previos pueden seguir usándose como contexto. Quedan **4.530 ejemplos**, separados
en 2.679 de entrenamiento, 877 de validación y 974 de prueba. Los 974 no son todo
lo que usa el proyecto: son solo los ejemplos reservados para medir desempeño.
`resultados['diagnostics']['sample_flow']` y la consola muestran este conteo.

Desde la raíz del proyecto, en PowerShell:

```powershell
.\.venv\Scripts\python.exe -X utf8 Diagnosis_Algorithms.py
```

La consola presenta las métricas principales, referencias simples, diferencias
entre entrenamiento/validación/prueba, resultados por año/temporada y estaciones
con mayor RMSE. También explica el tamaño efectivo de los datos y la concentración
del error. Los PNG se guardan en `outputs/evaluacion_espaciotemporal`.

```python
from Diagnosis_Algorithms import ejecutar_evaluacion

resultados = ejecutar_evaluacion(export_csv=False)
principal = resultados['metrics']['global'].query("split == 'test'")
print(principal[['model', 'N', 'MAE', 'RMSE', 'R2']].to_string(index=False))

# Son DataFrames completos, no rutas a archivos.
por_estacion = resultados['metrics']['station']
por_anio = resultados['metrics']['year']
por_temporada = resultados['metrics']['season']
predicciones = resultados['predictions']
```

`mostrar_resultados()` es la función de presentación en `Diagnosis_Algorithms.py`.
`ejecutar_evaluacion()` devuelve datos, modelos, métricas y diagnóstico en memoria.
Los CSV **no se exportan por defecto**; `--export-csv` habilita su exportación y
mantiene la impresión de resultados. `--no-plots --no-error-analysis` permite
ejecutar solamente entrenamiento y evaluación. `--quick` es una comprobación
rápida de funcionamiento, no una evaluación definitiva.

`Analisis_DQO.ipynb` incluye salidas ya calculadas. Para volver a ejecutar sus
celdas, seleccione el intérprete `.venv` en Jupyter/VS Code. Si necesita instalar
un kernel: `python -m pip install -e ".[notebook]"` con ese intérprete. Las salidas
incluidas se ejecutaron con Python del entorno, sin servidor Jupyter; no se
requiere instalar Jupyter para ejecutar el script principal.

`HISTORIAL_DESARROLLO.md` conserva decisiones, resultados e intentos descartados.
El repositorio no conserva CSV, JSON ni figuras generadas; el notebook incorpora
las tablas y gráficos necesarios. Cada ejecución puede recrear `outputs/`, que
está ignorado por Git.

## 1. Qué representan los datos

El CSV contiene 134.261 filas y 243 estaciones. Está en formato **largo**: una
fila representa una propiedad medida, no un ejemplo independiente de DQO.
Por ejemplo, pH, temperatura y DQO de una visita ocupan distintas filas.

`Data_Manage._cargar_muestras()` convierte este formato en una tabla **ancha**,
con una fila por estación y fecha/hora. El código de muestra no es la clave de
la visita: réplicas coincidentes se agregan por mediana. Hay nueve combinaciones
estación-fecha con códigos de muestra de DQO diferentes; el informe registra esta
decisión para que se revise si las réplicas debieran estudiarse separadamente.

Las tablas relevantes son:

| Tabla/objeto | Una fila representa | Contenido |
|---|---|---|
| CSV original | Una propiedad de una visita | Resultado, unidad, estación, fecha y ubicación |
| `prepared.frame` | Una visita | Química, DQO, coordenadas, historia, calendario, partición y elegibilidad |
| `partition.metadata` | Una visita evaluable | Identidad y contexto de cada fila del modelo |
| `partition.X` | Una visita evaluable | Predictores tabulares imputados/escalados |
| `partition.X_sequence` | Una ventana terminada en esa visita | Cinco visitas de la misma estación |
| `predictions` | Un modelo sobre una visita | DQO real, estimada, residuo y metadatos |
| `metrics['station_year']`, etc. | Un grupo de evaluación | N, MAE, RMSE, R² y banderas de calidad |

`sample_id` se deriva de estación y fecha. Permanece junto a los datos desde la
preparación hasta la evaluación para evitar asociar una predicción con otra
estación o fecha al ordenar y transformar arrays.

## 2. Limpieza: qué cambia y qué se conserva

1. Se normalizan nombres y se eliminan 45 duplicados exactos. El CSV original
   permanece intacto.
2. Se interpretan fechas IDEAM y fechas ISO explícitas; fechas inválidas no
   entran al modelo. No se adivina el orden día/mes de formatos ambiguos.
3. Se convierten resultados numéricos: la coma de `<3,00` es decimal. Mezclas
   ambiguas de coma y punto se consideran ausentes.
4. No se mezclan propiedades con unidades diferentes. Cada par propiedad/unidad
   tiene una columna. DQO debe estar en mg O2/L; no se convierten unidades por
   suposición ni se selecciona una unidad según frecuencias de prueba.
5. DQO negativa se considera inválida. **No se recortan ni eliminan los valores
   altos de DQO** para mejorar artificialmente el error.
6. Hay 6.694 filas originales de DQO, 6.693 después de quitar un duplicado exacto;
   1.566 están censuradas. `<10` no significa 10 ni un valor conocido. La política
   predeterminada excluye como etiquetas las visitas con censura. `half_limit`
   aproxima los límites inferiores por la mitad; los resultados `>x` quedan
   ausentes. Esta alternativa debe informarse y compararse, no tratarse como verdad.
7. En covariables, `<x` se aproxima por x/2 y se conserva su indicador de censura.
   Es una aproximación explícita, no una recuperación del valor real.
8. Ubicación/elevación se agregan dentro de la visita, sin usar ubicación futura
   para completar el pasado. Valores geográficos inválidos quedan ausentes.

La tabla resultante tiene 6.835 visitas, 5.114 con DQO utilizable bajo la política
predeterminada. Las visitas sin etiqueta pueden proporcionar contexto histórico,
pero no producen una observación evaluada. Excluir censura limita el alcance del
resultado a DQO cuantificada y puede desplazar la distribución hacia valores mayores.

## 3. Variables y clasificación espacio-temporal

Se conservan estación, fecha, latitud, longitud y elevación como metadatos.
Se agregan año, mes, seno/coseno del mes, tiempo desde la visita previa,
cantidad de visitas anteriores y antigüedad de la última DQO conocida.

`dqo_lag_1..3` desplazan la serie histórica de DQO rellenada solo hacia adelante.
Por ello pueden repetir una misma DQO cuando faltan etiquetas: son valores
disponibles en visitas previas, no necesariamente tres análisis independientes.
No se usa la DQO de la visita actual ni su indicador de censura como predictor.

La temporada por defecto es un **proxy bimodal no validado por estación**:
diciembre-febrero `seca_1`, marzo-mayo `lluviosa_1`, junio-agosto `seca_2`,
septiembre-noviembre `lluviosa_2`. El año es calendario. No representa un año
hidrológico ni lluvia medida. Colombia presenta variabilidad regional del régimen
pluviométrico; véase la [regionalización del IDEAM](https://www.ideam.gov.co/documents/21021/21141/Regionalizacion%2Bde%2Bla%2BPrecipitacion%2BMedia%2BMensual/1239c8b3-299d-4099-bf52-55a414557119).

Se pueden pasar calendarios locales directamente en Python, sin archivos JSON:

```python
calendario = {
    'source': 'Calendario local pendiente de validar',
    'stations': {
        'NOMBRE EXACTO DE LA ESTACION': {
            '1': 'seca', '2': 'seca', '3': 'transicion', '4': 'lluviosa',
            '5': 'lluviosa', '6': 'lluviosa', '7': 'lluviosa', '8': 'lluviosa',
            '9': 'lluviosa', '10': 'lluviosa', '11': 'transicion', '12': 'seca',
        }
    }
}
# Ejemplo de estructura, no calendario validado para una región:
# resultados = ejecutar_evaluacion(season_config=calendario)
```

Si se proporciona configuración sin calendario para una estación y sin
`default`, esa estación se marca `sin_clasificar`. No se inventa su clima.

## 4. Separación y prevención de fuga de información

Se ordenan globalmente las fechas, conservando todas las visitas del mismo día
en una partición. Por defecto, 60 % de fechas únicas con etiqueta se asigna a
entrenamiento, 20 % a validación y 20 % a prueba. No son porcentajes de filas.
En estos datos los cortes son 2015-08-11 y 2020-02-06.

Solo con entrenamiento se calculan cobertura de covariables, medianas para
faltantes, medias/desviaciones del escalado y escalado del objetivo. Es el
principio recomendado para evitar fuga en el
[preprocesamiento de scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).
Se seleccionan nueve variables químicas con cobertura >=70 %, se agregan sus
indicadores de censura y los indicadores de ausencia; en total hay 60 predictores.

Se crean secuencias de cinco visitas de una estación, incluida la actual. Las
primeras cuatro visitas no pueden ser extremos evaluables. XGBoost y SVM usan
la última fila transformada (que ya contiene lags); LSTM usa toda la ventana.
Los tres se evalúan en la misma cohorte: 2.679 ejemplos de entrenamiento,
877 de validación y 974 de prueba.

**Supuesto operativo:** antes de una visita se conocen los resultados de visitas
anteriores. Una ventana de prueba puede incluir visitas previas de validación
o prueba; eso es contexto pasado, no reajuste del modelo. Si los resultados de
laboratorio llegan con retraso, haría falta modelar su fecha de disponibilidad.
Este protocolo estima DQO en una visita con covariables contemporáneas; no es
un pronóstico futuro sin nuevas observaciones.

## 5. Entrenamiento

El objetivo se transforma por defecto con `log1p` y luego se estandariza con
parámetros de entrenamiento. Se invierten ambas operaciones antes de calcular
MAE/RMSE/R². Las predicciones negativas se proyectan a cero.

- **XGBoost:** árboles de profundidad 5, hasta 800 iteraciones, regularización y
  muestreo parcial. La parada temprana usa RMSE del objetivo transformado en
  validación. No se buscan automáticamente todas sus configuraciones.
- **SVM/SVR:** kernel RBF; 12 combinaciones de C, epsilon y gamma. Cada candidato
  se entrena con train y se compara mediante RMSE en mg O2/L en validación.
  Se verifica convergencia. No se usa validación cruzada aleatoria.
- **LSTM:** secuencias reales, 64 unidades, dropout, capa densa y pérdida MSE
  del objetivo transformado. Máximo 60 épocas y parada tras ocho épocas sin
  mejora de validación; se restauran los mejores pesos. No se mezclan filas.

Los modelos seleccionados no se reajustan con prueba. `--raw-target` permite
investigar el entrenamiento sin logaritmo manteniendo el escalado; el notebook
compara esa decisión solo en validación, sin sustituir el experimento principal.

## 6. Evaluación y explicación de resultados

MAE es la media de los errores absolutos. RMSE es la raíz de la media de errores
cuadráticos y penaliza mucho más los errores grandes. R² compara la suma del
error cuadrático con la variabilidad de la DQO dentro del conjunto evaluado.
R²=0 corresponde al error de predecir la media **de ese conjunto**; esa media
de prueba no se usa como predictor entrenado. Una referencia utilizable es la
media de entrenamiento, también incluida en los resultados.

Las métricas se recalculan para cada modelo y partición en ocho niveles:
global, estación, año, temporada, año-temporada, estación-año, estación-temporada
y estación-año-temporada. No se mezclan validación y prueba ni se promedian R²
locales para obtener el global. R² es NaN si hay una sola observación o DQO
constante. Se conserva N y una advertencia cuando N<5.

`Performance_Diagnostics.py` añade referencias simples, concentración del error,
sesgo en concentraciones altas, cobertura por partición y resultados separados
para estaciones con/sin etiquetas de entrenamiento. No son pruebas causales.

`Error_Analysis.py` entrena un Random Forest auxiliar para predecir **error
absoluto**, usando residuos de validación y factores geográficos/temporales.
Se evalúa en prueba y se compara con la mediana del error de validación.
SHAP explica este modelo auxiliar, no directamente la predicción DQO; consulte
la [documentación de TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
Si SHAP falla, se comunica y usa importancia por permutación. Las tablas se
devuelven en `resultados['error_analysis']['tables']`; exportarlas es opcional.
Un auxiliar sin capacidad predictiva suficiente se marca `low_predictive_skill`:
sus importancias no justifican afirmar causas del error.

## 7. Qué podemos concluir y qué falta probar

Los resultados concretos y experimentos de esta iteración están en
`HISTORIAL_DESARROLLO.md` y en el notebook. Es necesario distinguir:

- **Cantidad:** hay miles de etiquetas útiles globalmente, pero pocas por
  estación y período. Las 134 mil filas no son 134 mil ejemplos DQO.
- **Irregularidad:** la mediana entre visitas es 117 días; 2.004 intervalos
  superan 180 días. Cinco visitas no equivalen a cinco pasos de duración fija.
- **Cola de la distribución:** pocos errores grandes dominan RMSE. Los modelos
  subestiman varios picos; quitar esos casos ocultaría una debilidad importante.
- **Objetivo de entrenamiento:** minimizar pérdida en log no equivale a minimizar
  RMSE en unidades físicas. Se prueba esa hipótesis en validación.
- **Cobertura:** seleccionar solo variables frecuentes puede excluir variables
  informativas como DBO5. Reducir el umbral también agrega ausencia y ruido;
  hace falta comprobarlo, no asumir que más columnas siempre mejora.
- **Generalización:** prueba incluye estaciones sin etiquetas de entrenamiento.
  Se informa por separado, sin inferir automáticamente que expliquen todo el error.
- **Sobreajuste:** comparar train/validation/test y curvas ayuda a detectarlo.
  Aumentar épocas no lo corrige por sí solo.

La separación cronológica corresponde al uso declarado y es más exigente que
mezclar pasado y futuro. Un desempeño menor frente al código antiguo no demuestra
que la partición nueva esté mal: antes había imputación/recorte sobre toda la
serie y particiones aleatorias. No se han reejecutado aquí aquellos resultados
antiguos como una comparación experimental controlada.

## 8. CRISP-ML(Q) y límites

Se implementan comprensión del objetivo, revisión de calidad, preparación sin
fuga temporal, selección de modelos, evaluación desagregada y trazabilidad del
experimento. Los reportes y tablas permiten revisión humana. No se ha desplegado
un servicio ni implementado monitoreo operativo automático.

Próximas validaciones: cortes temporales sucesivos, calendarios locales,
tratamiento específico de censura, ablaciones de variables/ventanas y evaluación
reservando estaciones completas si interesa transferencia espacial. Ajustes
decididos después de mirar prueba deben confirmarse en otro período reservado.

## Verificación antes de cada commit

La carpeta de pruebas unitarias se retiró por decisión del proyecto. Antes de
confirmar cambios se debe ejecutar al menos la compilación/importación del código,
la ayuda del CLI y una corrida rápida:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -X utf8 Diagnosis_Algorithms.py --help
.\.venv\Scripts\python.exe -X utf8 Diagnosis_Algorithms.py --quick --models XGBoost --no-plots --no-error-analysis
```

Para resultados científicos debe ejecutarse sin `--quick`. El historial debe
registrar configuración, cortes, semilla y métricas de validación/prueba.
