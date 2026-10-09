# Informe técnico · ¿Dónde faltan cajeros Link en CABA?

TPO de Ciencia de Datos · UADE · 2º cuatrimestre 2026 · Docente: Santiago Martín

Metodología: CRISP-DM. Código, datos y resultados: [github.com/ValentinoDiaz0509/CienciaDeDatos](https://github.com/ValentinoDiaz0509/CienciaDeDatos).
Este informe se genera a partir de los resultados del pipeline (`python docs/_construir/informe.py`); todos los números salen de `data/processed/`.

---

## Resumen ejecutivo

- Hoy el **68 %** de los vecinos de CABA tiene un cajero Link a menos de 500 m (en línea recta), contra el
  **79 %** de Banelco. Unas **976.441 personas** no tienen un Link a distancia caminable.
- Los cajeros **siguen al comercio, no a los vecinos**: en los corredores comerciales hay
  17,9 terminales cada 10.000 vecinos; en los barrios
  residenciales, 5,3.
- Hay **dos oportunidades**. **Competir** en los corredores comerciales del norte y el centro (Palermo, Balvanera, Belgrano),
  donde la demanda justifica más terminales Link de las que hay. **Cubrir** los barrios residenciales del oeste y el sur,
  donde no hay ningún Link cerca.
- Con **20 cajeros** en las ubicaciones óptimas, Link suma **257.165 vecinos** y pasa al
  **76,7 %**. La diferencia con Banelco baja de 11 a 3 puntos. Al promedio
  nacional de 3.242 extracciones por terminal por mes (BCRA, 2025), son unas **64.840 operaciones mensuales**.
- El modelo se entrega como una **app** que la gerencia puede usar con su listado actual de cajeros.

---

## 1. Comprensión del negocio

### 1.1 Dominio
En la Argentina operan dos grandes redes de cajeros automáticos: **Red Link** y **Banelco**. Los bancos compran e instalan
las terminales; la red las conecta y procesa las operaciones. Según el BCRA, en julio de 2025 se hicieron **57,2 millones de
extracciones** en **17.643 cajeros** del país (unas 3.242 por terminal por mes), con una caída interanual cercana al 40 %, en un
contexto de avance de los pagos digitales. Con menos uso del efectivo, **dónde** está cada terminal importa más: una terminal mal
ubicada es un costo fijo con poco uso.

### 1.2 Problema e hipótesis
> En CABA hay zonas con mucha demanda potencial y pocos cajeros Link; detectarlas con datos públicos le permite a la
> Gerencia Comercial y Técnica de Red Link proponerles a sus bancos dónde instalar las próximas terminales con un
> argumento medible, en vez de a ojo.

| Pregunta | Respuesta |
|---|---|
| ¿Qué resolvemos? | Dónde faltan terminales Link y dónde conviene sumarlas |
| ¿Para quién? | Gerencia Comercial y Técnica de Red Link (audiencia no especialista en datos) |
| ¿Cómo entrega valor? | Más vecinos con un Link cerca, más operaciones en la red y una priorización que reemplaza la decisión a ojo |

### 1.3 Supuestos y alcance
- Cada banco decide la instalación; Red Link puede proponer ubicaciones a los bancos de su red.
- Solo datos públicos: no se usan transacciones ni ningún dato interno de Red Link.
- Alcance: Ciudad Autónoma de Buenos Aires.
- Criterio de éxito: un modelo de demanda que generalice a zonas no vistas (validación espacial) y un conjunto de
  ubicaciones cuyo impacto se pueda medir en vecinos cubiertos.

## 2. Comprensión de los datos

| Fuente | Organismo | Período | Aporta |
|---|---|---|---|
| Cajeros automáticos (1.279 ubicaciones, 2.673 terminales) | GCBA · BA Data | ≈2017 | Oferta por red y banco |
| Censo 2022 por radio censal (3.820 radios) | INDEC | 2022 | Residentes, NBI, edad, educación, internet |
| Molinetes del subte (90 estaciones) | SBASE · BA Data | 2025 | Entradas por estación y hora |
| Relevamiento de usos del suelo (417.764 parcelas) | GCBA · BA Data | 2022-2024 | Comercio, oficinas, vivienda |
| OpenStreetMap (≈29 mil puntos y red peatonal) | Colaboradores de OSM | 2026 | Comercios, estaciones, calles |
| Barrios | GCBA · BA Data | vigente | Límites |

**Calidad de datos (DAMA).** El listado de cajeros es completo, válido y sin duplicados, pero falla en **oportunidad**: el 31 %
de las ubicaciones está a nombre de bancos que dejaron de operar con ese nombre entre 2017 y 2024 (Citibank, BBVA Banco
Francés, Santander Río, HSBC). Lo tratamos como una foto de ≈2017. El censo por radio cubre el 99,2 % de la población oficial.
Los molinetes cubren 335 de los 365 días de 2025. El CSV de usos del suelo no trae coordenadas, así que se tomaron del shapefile.
OpenStreetMap está incompleto para cajeros (206 contra 1.279).

**Hallazgos del EDA** (cinco tipos de análisis exploratorio, `notebooks/01_calidad_y_eda.ipynb`):
1. El 78 % de los hexágonos no tiene ninguna terminal y la distribución tiene cola larga, lo que justifica usar modelos de conteo.
2. Las terminales se asocian más con el comercio (ρ = 0,68) que con los residentes (ρ = 0,47).
3. El centro concentra el 40 % de las terminales con el 12 % de los vecinos.
4. Link tiene mayor participación en el sur, donde pesan los bancos públicos, y menor en el norte comercial.

## 3. Preparación de los datos

- **Unidad de análisis:** 2.793 hexágonos H3 de resolución 9 (≈0,07 km², unas 3 × 3 manzanas).
  Todos tienen el mismo tamaño y los mismos vecinos, lo que reduce el problema de la unidad de área modificable frente a
  unidades administrativas desiguales.
- **Residentes por hexágono:** interpolación areal desde los radios censales (se conservan los totales). Los porcentajes se
  calculan agregando numeradores y denominadores, nunca promediando porcentajes.
- **Área de influencia:** variables y objetivo se suman sobre el hexágono y sus 6 vecinos (≈400 m), la escala de una caminata.
- **Variables de demanda (14):** residentes, comercio (OSM y parcelas), oficinas, parcelas residenciales, pasajeros de
  subte, estaciones de tren, distancia al centro y al subte, % NBI, % sin internet, % de 65+, % universitarios, tasa de empleo.
  Se excluyen deliberadamente sucursales y cajeros: son oferta, no demanda.
- **Distancias:** Las distancias se miden en línea recta (la red peatonal de OpenStreetMap se incorpora con `notebooks/00b_red_peatonal.ipynb`).

## 4. Modelado

### 4.1 Regresión de conteos: ¿cuántas terminales tienen las zonas con esta demanda?
Objetivo: terminales (todas las redes) en el área de influencia. Seis modelos más un ensamble definido de antemano,
evaluados con **validación cruzada espacial** (GroupKFold por comuna, 5 particiones) y, a modo de comparación, aleatoria.

| Modelo | D² Poisson espacial | D² Poisson aleatoria | MAE espacial | R² espacial |
|---|---|---|---|---|
| Regresión de Poisson (GLM) | 0,623 | 0,746 | 4,47 | 0,378 |
| Ensamble (Poisson GLM + Boosting) | 0,615 | 0,815 | 4,18 | 0,363 |
| Random Forest | 0,564 | 0,820 | 4,34 | 0,305 |
| Gradient Boosting (Poisson) | 0,521 | 0,836 | 4,19 | 0,297 |
| Árbol de decisión (CART) | 0,495 | 0,680 | 4,81 | 0,277 |
| Línea de base (promedio) | -0,110 | -0,003 | 8,01 | -0,042 |
| Poisson solo con población | -0,138 | 0,097 | 7,11 | 0,012 |

**Elección: Regresión de Poisson (GLM).** Con validación aleatoria habría ganado Gradient Boosting (Poisson)
(D² = 0,84), pero en zonas no vistas cae a
0,52: los modelos de árboles memorizan la geografía (autocorrelación espacial).
El GLM generaliza mejor (D² = 0,62) y además se puede explicar: subir un desvío las parcelas comerciales multiplica
las terminales esperadas por 2,15; los comercios de OSM, por 2,05; el % de 65+, por
1,40; los pasajeros de subte, por 1,22; las parcelas residenciales, por 0,44;
y el % de hogares con NBI, por 0,87.

**Brecha** = terminales esperadas (predicción fuera de muestra) − terminales reales. **Brecha Link** = lo esperado × participación
de Link en la ciudad (34,5 %) − terminales Link reales. Barrios con más terminales Link faltantes:

| Barrio | Residentes | Terminales Link | Terminales Banelco | Participación Link | Link faltantes |
|---|---|---|---|---|---|
| Palermo | 242.296 | 47 | 138 | 25 % | 49 |
| Balvanera | 145.825 | 69 | 78 | 47 % | 41 |
| Belgrano | 145.112 | 30 | 93 | 24 % | 32 |
| Recoleta | 160.757 | 57 | 148 | 28 % | 22 |
| Retiro | 66.257 | 41 | 113 | 27 % | 13 |

### 4.2 K-Means: tipos de zona
k = 4 por silueta (0,36) y codo, sobre variables de demanda estandarizadas (conteos en escala log).

| Tipo de zona | Residentes | Pasajeros de subte/día | Terminales cada 10 mil | Link cada 10 mil | NBI (mediana) |
|---|---|---|---|---|---|
| Residencial denso | 1.825.673 | 0 | 5,3 | 1,5 | 2,6 % |
| Corredores comerciales con subte | 857.976 | 686.058 | 17,9 | 6,7 | 4,3 % |
| Mayor vulnerabilidad social | 399.094 | 4.884 | 3,5 | 1,8 | 10,6 % |

### 4.3 Optimización: dónde sumar cajeros
**Problema de cobertura máxima** (MCLP; Church y ReVelle, 1974): elegir N sitios (hexágonos con comercio) que maximicen los
vecinos que pasan a tener un Link a menos de 500 m. Se resuelve con el **algoritmo goloso** (garantía de 1 − 1/e del óptimo por
submodularidad; Nemhauser, Wolsey y Fisher, 1978) y de forma **exacta** como programa lineal entero con HiGHS.

| Cajeros | Goloso (vecinos) | Exacto (vecinos) | Diferencia | Sitios en común | Segundos (exacto) |
|---|---|---|---|---|---|
| 5 | 84.521 | 84.521 | 0,00 % | 5 | 0,28 |
| 10 | 146.755 | 146.755 | 0,00 % | 10 | 0,28 |
| 20 | 257.038 | 257.165 | 0,05 % | 17 | 0,28 |
| 30 | 355.443 | 357.314 | 0,52 % | 22 | 0,28 |
| 50 | 517.597 | 522.141 | 0,87 % | 35 | 0,28 |

El goloso queda a menos del 0,9 % del óptimo. Las 20 ubicaciones óptimas:

| # | Barrio | Comuna | Tipo de zona | Vecinos que suma |
|---|---|---|---|---|
| 1 | Barracas | 4 | Mayor vulnerabilidad social | 22.015 |
| 2 | Villa Lugano | 8 | Mayor vulnerabilidad social | 18.764 |
| 3 | Balvanera | 3 | Residencial denso | 15.071 |
| 4 | Villa Soldati | 8 | Mayor vulnerabilidad social | 14.725 |
| 5 | Belgrano | 13 | Residencial denso | 13.945 |
| 6 | Flores | 7 | Residencial denso | 13.676 |
| 7 | Flores | 7 | Residencial denso | 12.224 |
| 8 | Villa Pueyrredon | 12 | Residencial denso | 12.155 |
| 9 | Flores | 7 | Residencial denso | 11.853 |
| 10 | Villa Urquiza | 12 | Residencial denso | 11.848 |
| 11 | Flores | 7 | Mayor vulnerabilidad social | 11.830 |
| 12 | Villa Lugano | 8 | Residencial denso | 11.650 |
| 13 | Villa Urquiza | 12 | Residencial denso | 11.484 |
| 14 | Coghlan | 12 | Residencial denso | 11.406 |
| 15 | Villa Luro | 10 | Residencial denso | 11.222 |
| 16 | Floresta | 10 | Residencial denso | 10.757 |
| 17 | Almagro | 5 | Corredores comerciales con subte | 10.750 |
| 18 | Villa Del Parque | 11 | Residencial denso | 10.677 |
| 19 | Villa Santa Rita | 11 | Residencial denso | 10.565 |
| 20 | Boedo | 5 | Residencial denso | 10.549 |

### 4.4 Demanda por hora
Con los molinetes 2025 (690.863 entradas por día hábil), los picos son a las **8:00** y las
**17:00**. Un K-Means sobre la forma de la curva horaria de cada estación separa estaciones de **origen** (pico a la
mañana, barrios) y de **destino** (pico a la tarde, centro). 87 de 90
estaciones ya tienen un Link a menos de 500 m. Sumar a los pasajeros de la hora pico no cambia las ubicaciones óptimas: el
faltante está en los barrios, no en el transporte.

## 5. Evaluación

- **Generalización:** D² espacial = 0,62, MAE = 4,5 terminales por área de influencia.
- **Validación temporal:** de los 206 cajeros que OpenStreetMap tiene mapeados en 2026, 9 no
  estaban en el listado 2017. Cayeron, en promedio, en zonas con más brecha que el 66 % de la
  ciudad (p = 0,049, Monte Carlo). El AUC es 0,65, contra 0,51 de usar solo la población.
  Es una evidencia a favor, pero modesta, por la cantidad de casos.
- **Robustez:** las ubicaciones óptimas no cambian al sumar a los pasajeros de la hora pico. El goloso y el exacto coinciden casi
  por completo.

**Limitaciones.**
- La oferta es de ≈2017.
- No hay transacciones: la demanda se aproxima con residentes, comercio y pasajeros.
- La brecha compara contra el promedio del mercado, así que no detecta zonas que el mercado entero desatiende (la
  optimización por cobertura cubre ese caso).
- OpenStreetMap varía en completitud entre barrios.
- Los feriados en día de semana cuentan como hábiles.

## 6. Despliegue

App en Streamlit con mapas de deck.gl (`streamlit run app/app.py`). Funciona sin internet. Incluye:
- mapa 3D de la brecha;
- simulador de N cajeros (exacto o goloso, distancia caminando o en línea recta, con o sin pasajeros por franja horaria);
- demanda por hora;
- tipos de zona;
- carga del listado propio de cajeros, que recalcula cobertura y ubicaciones.

Una terminal nueva que alcance el promedio nacional suma unas 3.242 extracciones por mes, por unos 243 millones de pesos de efectivo (monto
promedio de julio 2025). El ingreso por operación es un dato interno: la app lo recibe como parámetro.

**Próximos pasos:**
- correr el modelo con el listado actual y las transacciones por terminal;
- sumar restricciones de costo y de seguridad por sitio a la optimización;
- validar la hipótesis horaria (carga de efectivo antes de los picos) con transacciones por hora.

## Referencias

- BCRA (2025). *Informe Mensual de Pagos Minoristas*, agosto de 2025 (datos de julio de 2025).
- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
- Church, R. y ReVelle, C. (1974). The maximal covering location problem. *Papers of the Regional Science Association*, 32, 101-118.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269-271.
- Fayyad, U., Piatetsky-Shapiro, G. y Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine*, 17(3).
- Friedman, J. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5).
- Huangfu, Q. y Hall, J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10, 119-142.
- INDEC (2024). *Censo Nacional de Población, Hogares y Viviendas 2022*, base por radio censal.
- McCullagh, P. y Nelder, J. (1989). *Generalized Linear Models* (2.ª ed.). Chapman & Hall.
- Nemhauser, G., Wolsey, L. y Fisher, M. (1978). An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14, 265-294.
- Openshaw, S. (1984). *The Modifiable Areal Unit Problem*. Geo Books.
- Roberts, D. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.
- Uber Technologies (2018). *H3: Uber's Hexagonal Hierarchical Spatial Index*.
- Wang, R. y Strong, D. (1996). Beyond accuracy: what data quality means to data consumers. *Journal of Management Information Systems*, 12(4).
- Fuentes de datos: GCBA (BA Data), INDEC, SBASE, © colaboradores de OpenStreetMap.
