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
| 1 | Dominio y problema | Red Link, el problema de la cobertura, la hipótesis y el valor | `docs/informe.md` §1, diapositivas 1-3 |
| 2 | Arquitectura de la solución | Diagrama de la tubería: fuentes → descarga → variables por hexágono → modelos → app | `docs/informe.md` §2, `docs/figuras/arquitectura.svg` |
| 3 | Análisis exploratorio | Calidad de datos (DAMA) y los cinco tipos de EDA; qué hay, qué falta, qué llamó la atención | `notebooks/01_calidad_y_eda.ipynb` |
| 4 | Técnica de minería | Regresión de conteos (6 modelos + ensamble, validación espacial), K-Means de zonas y de estaciones, cobertura máxima exacta (MILP) y golosa; por qué esos y no otros | `notebooks/02_modelos.ipynb`, `notebooks/03_demanda_por_hora.ipynb` |
| 5 | Conclusión | Mapas y curva de cobertura que responden la hipótesis | `notebooks/02_modelos.ipynb` §8, diapositivas finales |
| 6 | Aplicación funcional | App con mapa 3D, simulador de cajeros nuevos (exacto o goloso, caminando o en línea recta), demanda por hora y carga del listado propio | `app/app.py` |

### Condiciones para superar las expectativas mínimas

| Aspecto | Qué hicimos |
|---|---|
| **Datos** · dominio propio y múltiples orígenes | Dominio cercano (Red Link) y seis fuentes cruzadas: GCBA, INDEC, SBASE y OpenStreetMap |
| **Técnicas** · combinar modelos y justificar qué aporta cada uno | Supervisado (brecha) + no supervisado (tipos de zona y de estación) + optimización exacta (dónde poner cajeros), más un ensamble evaluado contra los modelos individuales y una validación temporal con datos que el modelo no vio |
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
| **Modelado supervisado** | Variables, comparación de modelos, validación, brecha | Técnica y por qué |
| **Segmentación y optimización** | K-Means, cobertura máxima (exacta y golosa), demanda por hora, validación temporal | Tipos de zona y dónde sumar cajeros |
| **Producto** | App y visualizaciones | Demo en vivo |

- **Con 5 integrantes:** se unen *Modelado supervisado* y *Segmentación y optimización*.
- **Con 4 integrantes:** además se unen *Ingeniería de datos* y *Análisis exploratorio*.

## 3. Guion de 15 minutos

La consigna pide **primero la conclusión, después cómo llegamos**. El corte es real: ensayar con reloj.

| Tiempo | Rol | Contenido | Diapositivas |
|---|---|---|---|
| 0:00 – 2:00 | Negocio | El problema, la hipótesis y **la respuesta**: dónde faltan, cuántos vecinos se suman con 20 cajeros | 1-3 |
| 2:00 – 3:30 | Ingeniería de datos | Seis fuentes, por qué cada una, y el diagrama de la tubería | 4-5 |
| 3:30 – 5:30 | Análisis exploratorio | Calidad (el listado de cajeros es de ≈2017), dónde vive la gente vs. dónde están los cajeros, la paradoja del centro | 6-7 |
| 5:30 – 8:00 | Modelado supervisado | Qué predecimos, seis modelos, por qué validación espacial, por qué ganó el elegido, qué aprendió, el mapa de brecha | 8-10 |
| 8:00 – 10:30 | Segmentación y optimización | Tipos de zona, cobertura caminando Link vs. Banelco, curva de cajeros nuevos, demanda por hora, validación con OSM | 11-14 |
| 10:30 – 13:00 | Producto | Demo: panorama → simulador (mover el slider, cambiar caminando/línea recta) → "Usá tus datos" | 15 |
| 13:00 – 15:00 | Negocio | Valor en operaciones y efectivo, conclusión, próximos pasos y limitaciones | 16-17 |

## 4. Checklist del día de la entrega

La consigna es explícita: *si no abre, no se presentó.*

**La semana anterior**
- [ ] Ensayo completo cronometrado (dos veces), con cada rol presentando su parte.
- [ ] Revisión de ortografía de todas las diapositivas (una por persona, cruzada).
- [ ] App probada **en modo avión** en la notebook que se va a usar.
- [ ] Notebooks ejecutados de punta a punta, guardados con los resultados a la vista.
- [ ] Video de respaldo de la demo (2 minutos, grabado con la app real).

**El día**
- [ ] Notebook con el repo clonado y el entorno instalado (`pip install -r requirements.txt`).
- [ ] App abierta antes de empezar: `streamlit run app/app.py`.
- [ ] Presentación descargada en PDF y en PowerPoint, en la notebook y en un pendrive.
- [ ] Cargador y adaptador HDMI.
- [ ] Notebook de modelos abierto en la sección de comparación, por si piden correr código en vivo.

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
error parece menor. Dejando afuera una comuna entera medimos lo que importa: qué tan bien anda en una zona que no vio.
La diferencia entre ambas validaciones está en la comparación de modelos.

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
problema de forma exacta como programa lineal entero (HiGHS, menos de un segundo): el goloso quedó a menos del 0,1 % del
óptimo midiendo caminando (y a menos del 1 % en línea recta). La app usa el exacto por defecto.

**El uso de efectivo está cayendo. ¿Tiene sentido sumar cajeros?**
Las extracciones en cajeros cayeron fuerte en el último año (BCRA, julio 2025). Justamente por eso conviene priorizar con
datos: las zonas que más dependen del efectivo (más NBI, menos acceso a internet) y las que hoy no tienen un Link cerca.

**¿Por qué la demanda por hora casi no cambió las recomendaciones?**
Porque la gran mayoría de las estaciones de subte ya tiene un Link a menos de 500 m caminando (80 de 90): la gente que
viaja ya pasa cerca de un cajero. Sumar a los pasajeros de la mañana deja 19 de los 20 sitios iguales. El faltante grande
está en los barrios, y que el resultado casi no cambie al sumar pasajeros es una prueba de robustez.

**¿Por qué medir caminando y no en línea recta?**
Porque nadie camina en línea recta: hay manzanas, vías del tren, parques y autopistas. Bajamos la red peatonal de
OpenStreetMap con OSMnx (52.090 esquinas), proyectamos cada punto sobre la cuadra más cercana y calculamos el camino más
corto con Dijkstra. En la mediana, caminar es 1,34 veces la línea recta. Cambia la respuesta: con línea recta, el
68 % de los vecinos tenía un Link a 500 m; caminando, el 51 %. La línea recta sobreestimaba la cobertura.

**¿Por qué 500 metros?**
Son unas 5 cuadras, 6 o 7 minutos a pie: una distancia que una persona está dispuesta a caminar hasta un cajero. Es un
supuesto, por eso el simulador permite elegir entre 300 y 800 m y ver cómo cambian las ubicaciones.

**¿Qué harían con más tiempo?**
Transacciones reales por terminal y por hora; restricciones de costo y de seguridad por sitio en la optimización; el
listado actual de cajeros; y un modelo de competencia (cuántas operaciones le saca un Link nuevo a un Banelco cercano).
