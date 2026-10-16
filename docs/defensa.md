# Guía para la defensa

Todo lo necesario para presentar el TPO en 15 minutos frente a la "Gerencia Comercial y Técnica", según el
documento *Ciencia de Datos – Trabajo Práctico Obligatorio* (docente: Santiago Martín).

## 1. Cumplimiento de la consigna, punto por punto

### Lineamientos generales

| Requisito | Cómo lo cumplimos | Dónde está |
|---|---|---|
| Grupo de 5 a 6, cada uno con un rol propio | Seis roles definidos, con alternativas para 4 o 5 integrantes (sección 2) | este documento |
| Dominio conocido o de interés | Red de cajeros: un integrante trabaja en Red Link (solo usamos datos públicos) | `README.md` |
| Fuentes identificadas y justificadas | Seis fuentes de cuatro organismos, con qué aporta cada una y su calidad medida | `notebooks/01_calidad_y_eda.ipynb` |
| Primero el problema, después los datos | La hipótesis se definió antes de buscar datos; los datos se eligieron para responderla | `docs/informe.md` §1 |

### La hipótesis en una frase

> En CABA hay zonas con mucha demanda potencial y pocos cajeros Link; detectarlas con datos públicos le permite a la
> Gerencia Comercial y Técnica de Red Link proponerles a sus bancos dónde instalar las próximas terminales con un
> argumento medible, en vez de a ojo.

- **¿Qué resolvemos?** Dónde faltan terminales Link y dónde conviene sumarlas.
- **¿Para quién?** Gerencia Comercial y Técnica de Red Link (no especialistas en datos).
- **¿Cómo entrega valor?** Más vecinos con un Link a distancia caminable y más operaciones procesadas por la red, con una
  priorización que reemplaza la decisión a ojo.

### Los seis entregables (ninguno es opcional)

| # | Entregable | Qué mostramos | Archivo |
|---|---|---|---|
| 1 | Dominio y problema | Red Link, el problema de la cobertura, la hipótesis y el valor | `docs/informe.md` §1, diapositivas 1-4 |
| 2 | Arquitectura de la solución | Diagrama de la tubería: fuentes → descarga → variables por hexágono → modelos → app | `docs/informe.md` §3, `docs/figuras/arquitectura.svg`, diapositiva 6 |
| 3 | Análisis exploratorio | Calidad de datos (DAMA) y los cinco tipos de EDA; qué hay, qué falta, qué llamó la atención | `notebooks/01_calidad_y_eda.ipynb`, diapositivas 7-8 |
| 4 | Técnica de minería | **Machine learning:** regresión (7 modelos, de la línea de base a árboles, Random Forest, XGBoost y un ensamble) y clasificación (4 modelos) con validación cruzada espacial anidada, curva de sobreajuste, OOB, SHAP y Gini. **Prescriptivo:** K-Means de zonas y de estaciones, cobertura máxima exacta (MILP) y golosa. **Pronóstico** diario del subte (Random Forest y XGBoost sobre el calendario, validados hacia adelante). Por qué esos y no otros | `notebooks/02_machine_learning.ipynb`, `03_optimizacion.ipynb`, `04_demanda_por_hora.ipynb`, `05_pronostico_subte.ipynb`, diapositivas 9-21 |
| 5 | Conclusión | Mapas, tres zonas de ejemplo y curva de cobertura que responden la hipótesis | `notebooks/02_machine_learning.ipynb` §9, diapositivas 23-24 |
| 6 | Aplicación funcional | App con mapa 3D, predicción por zona (tiene / le corresponden / por qué), simulador de cajeros nuevos (exacto o goloso, caminando o en línea recta), demanda por hora, pronóstico del subte y carga del listado propio | `app/app.py`, diapositiva 22 |

### Condiciones para superar las expectativas mínimas

| Aspecto | Qué hicimos |
|---|---|
| **Datos** · dominio propio y múltiples orígenes | Dominio cercano (Red Link) y seis fuentes cruzadas: GCBA, INDEC, SBASE y OpenStreetMap; encontramos y aislamos un error de fechas en los molinetes de 2025 |
| **Técnicas** · combinar modelos y justificar qué aporta cada uno | Machine learning supervisado (árboles, Random Forest, XGBoost y un ensamble, en regresión y clasificación, con validación espacial anidada y explicación SHAP) + no supervisado (tipos de zona y de estación) + optimización exacta (dónde poner cajeros) + pronóstico en el tiempo, y una validación temporal con datos que el modelo no vio |
| **Visualización** · librerías modernas y navegación novedosa | Mapa 3D de hexágonos H3 con deck.gl, columnas 3D de pasajeros por hora, simulador interactivo y carga de datos propios |

### Material de soporte

| Capa | Herramienta |
|---|---|
| Presentación | Deck de diapositivas (exportable a PowerPoint y PDF) |
| Visualización | Python (matplotlib, pydeck/deck.gl) y Streamlit |
| Ejecución | JupyterLab o VS Code en local; Colab como respaldo |

### Criterios de evaluación

| Criterio | Cómo lo cuidamos |
|---|---|
| Creatividad, prolijidad y claridad | Una idea por diapositiva, gráficos con título que dice la conclusión, ortografía revisada |
| Coherencia | La conclusión responde exactamente la hipótesis (zonas con brecha + dónde sumar + cuánto mejora) |
| Valor para el negocio | Resultados en vecinos cubiertos, operaciones por mes y efectivo movido, no en métricas del modelo |
| Participación | Seis roles, cada uno presenta su parte y tiene preguntas probables preparadas |

**Condición para aprobar:** llevar todo funcionando sin depender de internet (sección 4).

## 2. Roles

| Rol | Responsable de | Presenta |
|---|---|---|
| **Negocio** (líder) | Problema, hipótesis, valor y conclusión | Apertura y cierre |
| **Ingeniería de datos** | Fuentes, descarga y arquitectura | Fuentes y diagrama |
| **Análisis exploratorio** | Calidad de datos y EDA | Hallazgos |
| **Machine learning** | Variables, comparación de modelos, validación espacial anidada, sobreajuste y OOB, SHAP, clasificación | Modelos, métricas y las tres zonas |
| **Segmentación y optimización** | K-Means, cobertura máxima (exacta y golosa), demanda por hora y pronóstico del subte, validación temporal | Tipos de zona, dónde sumar cajeros y cuánta gente va a pasar |
| **Producto** | App y visualizaciones | Demo en vivo |

- **Con 5 integrantes:** se unen *Machine learning* y *Segmentación y optimización*.
- **Con 4 integrantes:** además se unen *Ingeniería de datos* y *Análisis exploratorio*.

## 3. Guion de 15 minutos

La consigna pide **primero la conclusión, después cómo llegamos**. El corte es real: ensayar con reloj.

| Tiempo | Rol | Contenido | Diapositivas |
|---|---|---|---|
| 0:00 – 2:10 | Negocio | El problema, la hipótesis, **la respuesta** y los cuatro niveles de análisis | 1-4 |
| 2:10 – 4:20 | Ingeniería de datos · Análisis exploratorio | Seis fuentes y por qué cada una, la tubería, la calidad (el listado es de ≈2017) y dónde vive la gente vs. dónde están los cajeros | 5-8 |
| 4:20 – 9:00 | Machine learning | Cómo entrenamos y validamos, los 7 modelos y sus métricas, sobreajuste y OOB, qué aprendió (SHAP y Gini), dónde faltan terminales, **tres zonas, tres decisiones** y las 254 oportunidades del clasificador | 9-15 |
| 9:00 – 11:50 | Segmentación y optimización | Tipos de zona, por qué medir caminando, cobertura Link vs. Banelco y curva de cajeros nuevos, demanda por hora (15 segundos), **pronóstico del subte** y validación con OSM | 16-21 |
| 11:50 – 13:20 | Producto | Demo: predicción por zona (Belgrano y Villa Urquiza) → simulador (20 a 35 cajeros, caminando/línea recta) → "Usá tus datos" si sobra tiempo | 22 |
| 13:20 – 14:30 | Negocio | Valor en operaciones y efectivo, conclusión, próximos pasos y limitaciones | 23-24 |

El anexo (A1-A5: coeficientes, hiperparámetros, curvas ROC, árbol y explicación SHAP) queda para las preguntas.

**La fórmula de cada diapositiva técnica**, como pidió el profe: *dataset → modelo → métrica → hallazgo de negocio*. La
barra de abajo de cada una dice los datos, la técnica y la métrica; el título dice el hallazgo. Ejemplo para decir en voz
alta: "Con el censo, los usos del suelo y OpenStreetMap entrenamos un ensamble de regresión de Poisson y XGBoost, validado
en comunas que no vio (D² = 0,57, erra 4 terminales por zona). Con eso vimos que en Villa Urquiza, cerca de Juan Manuel de
Rosas, viven 17 mil personas pero hay poco comercio y no pasa el subte: el flujo de gente es bajo y le corresponde un solo
cajero Link."

Y para el pronóstico: "Con la serie diaria de los molinetes entrenamos un Random Forest sobre el calendario, validado en meses que no
vio (erra un 8 % por día, contra 22 % de repetir el año anterior). Con eso vimos que en enero entra un tercio
menos de gente que en septiembre: es el momento para el mantenimiento, y los días previos a los picos, para reforzar la carga
de efectivo."

## 4. Checklist del día de la entrega

La consigna es explícita: *si no abre, no se presentó.*

**La semana anterior**
- [ ] Ensayo completo cronometrado (dos veces), con cada rol presentando su parte.
- [ ] Revisión de ortografía de todas las diapositivas (una por persona, cruzada).
- [ ] App probada **en modo avión** en la notebook que se va a usar.
- [ ] Notebooks ejecutados de punta a punta, guardados con los resultados a la vista.
- [ ] Video de respaldo de la demo (2 minutos, grabado con la app real).

**El día**
- [ ] Notebook con el repo clonado y la app abierta al menos una vez antes (doble clic en `abrir_app.bat`: la primera vez instala todo).
- [ ] App abierta antes de empezar (`abrir_app.bat`, o `streamlit run app/app.py` con el entorno activado).
- [ ] Presentación descargada en PDF y en PowerPoint, en la notebook y en un pendrive.
- [ ] Cargador y adaptador HDMI.
- [ ] Notebook 02 (machine learning) abierto en la comparación de modelos, por si piden correr código en vivo
  (con `RECALCULAR = False` muestra los resultados en segundos).

## 5. Preguntas probables y respuestas

**¿Por qué no usaron las transacciones reales de Red Link?**
Son datos privados y no los usamos. El trabajo demuestra el método con datos públicos; la app permite cargar el listado
actual de cajeros, y el mismo modelo podría entrenarse con transacciones si la gerencia lo decide.

**El listado de cajeros es de 2017. ¿No invalida todo?**
No invalida el método, pero sí obliga a leer los resultados como una foto de esa oferta. Lo detectamos midiendo la
dimensión *oportunidad* de calidad de datos (bancos que ya no existen con ese nombre) y lo mitigamos de tres formas: lo
decimos explícitamente, validamos con los cajeros de OpenStreetMap 2026 y la app recalcula todo con el listado actual.

**¿No es circular predecir cajeros con cajeros?**
No usamos cajeros ni sucursales como variables: el modelo aprende cuántas terminales tiene *en promedio* una zona con
cierta demanda. La brecha es el desvío de cada zona respecto de ese promedio, calculado fuera de muestra. Su límite: si el
mercado entero desatiende un tipo de zona, la brecha no lo ve. Por eso la optimización no depende de la oferta actual:
cuenta personas sin un Link cerca.

**¿Por qué validación espacial y no un train/test al azar?**
Las zonas vecinas se parecen y comparten área de influencia. Al azar, el modelo "ve" la respuesta en los vecinos y el
error parece menor. Dejando afuera grupos de comunas medimos lo que importa: qué tan bien anda en una zona que no vio.
Con validación al azar habríamos elegido Random Forest (D² 0,81), que en comunas no vistas cae a 0,54.

**¿Por qué "anidada"?**
Porque los hiperparámetros también se eligen con datos. Si los elegimos mirando las mismas comunas con las que medimos,
el resultado queda optimista. Adentro de cada partición de entrenamiento corre otro GroupKFold (4 particiones) con
GridSearchCV; la partición que mide nunca participa de ninguna decisión.

**¿Qué es el D² y cómo se lee 0,57?**
Es el equivalente del R² para conteos: qué parte de la devianza de Poisson explica el modelo frente a predecir siempre el
promedio. 0,57 quiere decir que explica más de la mitad de la variación en terminales entre zonas que nunca vio. El ± 0,14
es cuánto varía entre grupos de comunas. El error medio es de 4,2 terminales por zona (área de 400 m).

**¿Por qué ganó un ensamble y no XGBoost solo?**
Por la regla fijada de antemano (mayor D² promedio en comunas no vistas): el ensamble da 0,57 contra 0,54 de XGBoost y es el
más estable. Combina lo que cada parte hace bien: la regresión de Poisson extrapola mejor a zonas extremas como el
microcentro y XGBoost capta relaciones no lineales. Promedia en escala logarítmica, así que se puede explicar exacto con SHAP.

**¿Qué es el sobreajuste y cómo lo ven?**
Un árbol más profundo siempre mejora en los datos de entrenamiento (D² 0,97 con 15 niveles) pero en comunas que no vio el
mejor está en 8 niveles y después empeora: memoriza ruido. Por eso Random Forest y XGBoost combinan muchos árboles, y
GridSearchCV le eligió a XGBoost árboles de 2 a 4 niveles.

**¿Qué es el OOB y por qué no lo usaron para elegir?**
Cada árbol del Random Forest se entrena con una muestra bootstrap y deja afuera un tercio de las filas: con ellas se mide un
error "gratis" (out-of-bag). Con datos espaciales es optimista, por la misma razón que la validación al azar: da R² 0,80
contra 0,29 en comunas no vistas. En clasificación la diferencia es chica (AUC 0,85 contra 0,84).

**¿Qué es SHAP?**
Una forma de repartir la predicción de cada zona entre las variables. Para el ensamble es exacta: XGBoost calcula sus
valores SHAP (TreeSHAP) y en la regresión de Poisson el aporte es coeficiente por valor. Se lee como un multiplicador: en la
zona de Belgrano cerca de Juramento, los comercios multiplican por 3,15 lo que le corresponde frente a una zona típica.

**¿Qué diferencia hay entre la importancia de Gini y SHAP?**
Gini (clase 8) mide cuánto ordena cada variable los cortes del Random Forest; tiende a favorecer variables con muchos valores
distintos. SHAP mide cuánto mueve cada variable la predicción. Coinciden en lo principal: el comercio primero, los vecinos
últimos.

**¿Cómo se lee el AUC del clasificador? ¿Y por qué ese umbral?**
AUC 0,84: si tomamos al azar una zona con cajero y una sin, el modelo le da más probabilidad a la que tiene en el 84 % de los
casos (0,5 sería azar). El umbral marca tantas zonas como las que hoy tienen cajero (la tasa base, 22 %), así precisión y
recall coinciden (0,57). Los falsos positivos son lo interesante: zonas sin cajero que se parecen a las que tienen.

**La regresión y el clasificador, ¿no dicen lo mismo?**
Se complementan. La regresión dice cuántas terminales le corresponden a un área y sirve para comparar barrios y redes. El
clasificador dice si a una zona puntual le corresponde al menos un cajero: sirve para encontrar huecos en zonas sin
ninguno, de ninguna red.

**¿Esto es machine learning o estadística?**
Las dos cosas, comparadas con la misma regla: la regresión de Poisson es un modelo estadístico clásico; el árbol, Random
Forest y XGBoost son machine learning (clases 6 y 8). La validación decide cuál generaliza mejor, no la preferencia.

**¿Por qué Poisson?**
El objetivo es un conteo (terminales), con muchos ceros y cola larga. La pérdida de Poisson modela eso directamente y
nunca predice valores negativos. La métrica acorde es la devianza de Poisson (D²).

**¿Por qué hexágonos y no barrios?**
Los barrios tienen tamaños muy distintos (Palermo vs. Boca) y eso distorsiona las comparaciones. Los hexágonos H3 tienen
todos el mismo tamaño y los mismos vecinos, y su escala (≈3 × 3 manzanas) es la de una caminata hasta un cajero.

**¿Cómo eligieron la cantidad de grupos del K-Means?**
Con el método del codo y la silueta promedio, entre 3 y 7 grupos, y verificando que cada grupo tenga una lectura de negocio.

**¿El algoritmo goloso da la mejor solución?**
No necesariamente: garantiza al menos el 63 % del óptimo porque la cobertura es submodular. Por eso también resolvimos el
problema de forma exacta como programa lineal entero (HiGHS, menos de un segundo): el goloso quedó a poco más del 1 % del
óptimo en el peor caso (50 cajeros). La app usa el exacto por defecto.

**El uso de efectivo está cayendo. ¿Tiene sentido sumar cajeros?**
Las extracciones en cajeros cayeron fuerte en el último año (BCRA, julio 2025). Justamente por eso conviene priorizar con
datos: las zonas que más dependen del efectivo (más NBI, menos acceso a internet) y las que hoy no tienen un Link cerca.

**¿Por qué la demanda por hora casi no cambió las recomendaciones?**
Porque la gran mayoría de las estaciones de subte ya tiene un Link a menos de 500 m caminando (80 de 90): la gente que
viaja ya pasa cerca de un cajero. Sumar a los pasajeros de la mañana no cambia ninguno de los 20 sitios. El faltante grande
está en los barrios, y que el resultado casi no cambie al sumar pasajeros es una prueba de robustez.

**¿Cómo pronosticaron cuánta gente va a pasar por el subte?**
Con la serie diaria de los molinetes (de enero de 2025 a agosto de 2026) entrenamos Random Forest
y XGBoost con 13 variables del calendario: día de la semana, feriados, puentes, vacaciones de invierno, época del año, entre
otras. Se conocen de antemano, así que se puede pronosticar cualquier fecha futura (incluso el feriado del 9 de noviembre de 2026
por la visita del Papa). Los comparamos contra una referencia ingenua: el mismo día de la semana del año anterior.

**¿Cómo validaron el pronóstico? ¿Qué es el MAPE?**
Hacia adelante, como se usaría en la realidad: entrenamos hasta diciembre de 2025 y medimos enero a abril de 2026; después,
hasta abril y medimos mayo a agosto. Nunca usamos fechas futuras para entrenar. El MAPE es el error porcentual medio por día:
Random Forest erra 8,0 %, XGBoost 8,2 % y la referencia 22,2 %. En el total de cada período de cuatro meses el error va de
−5,3 % a +4,4 %: por eso cada mes se lee como un rango (la banda del 80 %).

**¿Por qué no ARIMA o Prophet?**
Con menos de dos años de datos y un patrón dominado por el calendario, un modelo de árboles sobre variables del calendario capta
esos efectos de forma directa, incluidos los feriados que cambian de fecha cada año. ARIMA necesita una serie estacionaria y no
maneja bien los feriados. Con más años, valdría la pena sumarlos como candidatos y elegir con la misma validación.

**¿Qué encontraron en los datos del subte?**
Un error en el archivo de 2025: una parte de los registros del 1 al 12 de agosto tiene la fecha como mes/día y, leída como
día/mes, cae en el día 8 de otro mes. El 8 de cada mes tenía un tercio más de gente y del 1 al 12 de agosto faltaba lo mismo.
La prueba: marzo de 2025 no está en el archivo, pero el 8 de marzo tenía entradas, que sumadas a las del 3 de agosto dan un
domingo normal. Sacamos esos 22 días del entrenamiento; sin ese control, el modelo habría aprendido picos que no existen.

**¿El pronóstico sirve para decidir dónde poner cajeros?**
No es su objetivo: el dónde lo responde el modelo espacial. El pronóstico dice cuándo: cuánta gente va a pasar cada día y cada
mes, para planificar la carga de efectivo y el mantenimiento. Los molinetes cuentan entradas al subte, no extracciones: es un
indicador del flujo de gente, no de la demanda de efectivo.

**¿Por qué el intervalo sale de XGBoost si el modelo elegido es Random Forest?**
Porque XGBoost se puede entrenar para estimar un cuantil directamente (pérdida de cuantil): uno aprende el percentil 10 y
otro el 90. El Random Forest de scikit-learn no lo trae. La banda se ajusta para que siempre contenga el pronóstico.

**¿Por qué medir caminando y no en línea recta?**
Porque nadie camina en línea recta: hay manzanas, vías del tren, parques y autopistas. Bajamos la red peatonal de
OpenStreetMap con OSMnx (52.090 esquinas), proyectamos cada punto sobre la cuadra más cercana y calculamos el camino más
corto con Dijkstra. En la mediana, caminar es 1,34 veces la línea recta. Cambia la respuesta: con línea recta, el
68 % de los vecinos tenía un Link a 500 m; caminando, el 51 %. La línea recta sobreestimaba la cobertura. Probamos además
conectar cada punto con más de una cuadra (por ejemplo, en una esquina) y la cobertura cambia en poco más de un punto.

**¿Por qué 500 metros?**
Son unas 5 cuadras, 6 o 7 minutos a pie: una distancia que una persona está dispuesta a caminar hasta un cajero. Es un
supuesto, por eso el simulador permite elegir entre 300 y 800 m y ver cómo cambian las ubicaciones.

**¿Usaron inteligencia artificial?**
Sí, como apoyo para programar, revisar el código y redactar, y lo declaramos en el informe y el README. El planteo del
problema, la elección de fuentes y métodos, y la revisión de los resultados son del grupo. (Ajustar esta respuesta a lo que
efectivamente hizo cada uno.)

**¿Qué harían con más tiempo?**
Transacciones reales por terminal y por hora; restricciones de costo y de seguridad por sitio en la optimización; el
listado actual de cajeros; un modelo de competencia (cuántas operaciones le saca un Link nuevo a un Banelco cercano); y más
años de molinetes, clima y eventos para el pronóstico.
