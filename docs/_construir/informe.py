"""Genera docs/informe.md a partir de los resultados (python docs/_construir/informe.py)."""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.estilo import num_es as n  # noqa: E402

P = RAIZ / "data/processed"
R = json.loads((P / "resumen.json").read_text(encoding="utf-8"))
met = pd.read_csv(P / "metricas_modelos.csv")
esp = met[met["validación"].str.startswith("espacial")].set_index("modelo")
ale = met[met["validación"] == "aleatoria"].set_index("modelo")
coef = pd.read_csv(P / "coeficientes_glm.csv").set_index("variable")
perfil = pd.read_csv(P / "clusters_perfil.csv").set_index("tipo_zona")
ex = pd.read_csv(P / "recomendaciones_exactas_20.csv")
comp = pd.read_csv(P / "goloso_vs_exacto.csv")
res = pd.read_parquet(P / "resultados_hex.parquet")
cam = R["metodo_distancia"] == "caminando"
dist = "caminando por las calles" if cam else "en línea recta"

pb = (res.groupby("barrio").agg(r=("poblacion", "sum"), l=("term_link", "sum"), b=("term_banelco", "sum"),
                                f=("brecha_link", lambda s: s.clip(lower=0).sum() / 7)).sort_values("f", ascending=False))
pb["cuota"] = pb["l"] / (pb["l"] + pb["b"])
top = pb.head(5)

orden = esp.sort_values("D² Poisson", ascending=False).index
tabla_modelos = "\n".join(
    f"| {mo} | {n(esp.loc[mo, 'D² Poisson'], 3)} | {n(ale.loc[mo, 'D² Poisson'], 3)} | {n(esp.loc[mo, 'MAE'], 2)} | {n(esp.loc[mo, 'R²'], 3)} |"
    for mo in orden)
tabla_comp = "\n".join(
    f"| {int(r.cajeros)} | {n(r.goloso)} | {n(r.exacto)} | {n(r.diferencia_pct, 2)} % | {int(r.sitios_en_comun)} | {n(r.seg_exacto, 2)} |"
    for r in comp.itertuples())
zonas = [z for z in perfil.index if not z.startswith("Zonas sin")]
tabla_zonas = "\n".join(
    f"| {z} | {n(perfil.loc[z, 'residentes'])} | {n(perfil.loc[z, 'pax_subte'])} | {n(perfil.loc[z, 'terminales_cada_10mil'], 1)} | "
    f"{n(perfil.loc[z, 'link_cada_10mil'], 1)} | {n(100 * perfil.loc[z, 'pct_nbi_k1'], 1)} % |" for z in zonas)
tabla_top = "\n".join(f"| {b} | {n(r.r)} | {n(r.l)} | {n(r.b)} | {n(100 * r.cuota)} % | {n(r.f)} |" for b, r in top.iterrows())
tabla_ex = "\n".join(f"| {int(r.orden)} | {r.barrio} | {int(r.comuna)} | {r.tipo_zona} | {n(r.personas_nuevas)} |" for r in ex.itertuples())
ef = lambda v: n(coef.loc[v, "efecto"], 2)
dif_ini = 100 * (R["cobertura_banelco"] - R["cobertura_link"])
dif_fin = 100 * (R["cobertura_banelco"] - R["cobertura_link_con_20"])
red_txt = (f"Las distancias se miden **caminando** por la red peatonal de OpenStreetMap ({n(R['red_km'])} km de calles y sendas, "
           f"{n(R['red_nodos'])} esquinas), con el algoritmo de Dijkstra. En la mediana, caminar hasta el Link más cercano es "
           f"{n(R['factor_desvio_red'], 2)} veces la distancia en línea recta." if cam else
           "Las distancias se miden en línea recta (la red peatonal de OpenStreetMap se incorpora con `notebooks/00b_red_peatonal.ipynb`).")

texto = f"""# Informe técnico · ¿Dónde faltan cajeros Link en CABA?

TPO de Ciencia de Datos · UADE · 2º cuatrimestre 2026 · Docente: Santiago Martín

Metodología: CRISP-DM. Código, datos y resultados: [github.com/ValentinoDiaz0509/CienciaDeDatos](https://github.com/ValentinoDiaz0509/CienciaDeDatos).
Este informe se genera a partir de los resultados del pipeline (`python docs/_construir/informe.py`); todos los números salen de `data/processed/`.

---

## Resumen ejecutivo

- Hoy el **{n(100 * R['cobertura_link'])} %** de los vecinos de CABA tiene un cajero Link a menos de 500 m ({dist}), contra el
  **{n(100 * R['cobertura_banelco'])} %** de Banelco. Unas **{n(R['residentes_sin_link'])} personas** no tienen un Link a distancia caminable.
- Los cajeros **siguen al comercio, no a los vecinos**: en los corredores comerciales hay
  {n(perfil.loc['Corredores comerciales con subte', 'terminales_cada_10mil'], 1)} terminales cada 10.000 vecinos; en los barrios
  residenciales, {n(perfil.loc['Residencial denso', 'terminales_cada_10mil'], 1)}.
- Hay **dos oportunidades**. **Competir** en los corredores comerciales del norte y el centro ({", ".join(top.index[:3])}),
  donde la demanda justifica más terminales Link de las que hay. **Cubrir** los barrios residenciales del oeste y el sur,
  donde no hay ningún Link cerca.
- Con **20 cajeros** en las ubicaciones óptimas, Link suma **{n(R['nuevos_20_exacto'])} vecinos** y pasa al
  **{n(100 * R['cobertura_link_con_20'], 1)} %**. La diferencia con Banelco baja de {n(dif_ini)} a {n(dif_fin)} puntos. Al promedio
  nacional de {n(3242)} extracciones por terminal por mes (BCRA, 2025), son unas **{n(20 * 3242)} operaciones mensuales**.
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
| Cajeros automáticos ({n(R['ubicaciones'])} ubicaciones, {n(R['terminales'])} terminales) | GCBA · BA Data | ≈2017 | Oferta por red y banco |
| Censo 2022 por radio censal (3.820 radios) | INDEC | 2022 | Residentes, NBI, edad, educación, internet |
| Molinetes del subte ({R['estaciones']} estaciones) | SBASE · BA Data | 2025 | Entradas por estación y hora |
| Relevamiento de usos del suelo (417.764 parcelas) | GCBA · BA Data | 2022-2024 | Comercio, oficinas, vivienda |
| OpenStreetMap (≈29 mil puntos y red peatonal) | Colaboradores de OSM | 2026 | Comercios, estaciones, calles |
| Barrios | GCBA · BA Data | vigente | Límites |

**Calidad de datos (DAMA).** El listado de cajeros es completo, válido y sin duplicados, pero falla en **oportunidad**: el 31 %
de las ubicaciones está a nombre de bancos que dejaron de operar con ese nombre entre 2017 y 2024 (Citibank, BBVA Banco
Francés, Santander Río, HSBC). Lo tratamos como una foto de ≈2017. El censo por radio cubre el 99,2 % de la población oficial.
Los molinetes cubren 335 de los 365 días de 2025. El CSV de usos del suelo no trae coordenadas, así que se tomaron del shapefile.
OpenStreetMap está incompleto para cajeros ({R['cajeros_osm_2026']} contra {n(R['ubicaciones'])}).

**Hallazgos del EDA** (cinco tipos de análisis exploratorio, `notebooks/01_calidad_y_eda.ipynb`):
1. El 78 % de los hexágonos no tiene ninguna terminal y la distribución tiene cola larga, lo que justifica usar modelos de conteo.
2. Las terminales se asocian más con el comercio (ρ = 0,68) que con los residentes (ρ = 0,47).
3. El centro concentra el 40 % de las terminales con el 12 % de los vecinos.
4. Link tiene mayor participación en el sur, donde pesan los bancos públicos, y menor en el norte comercial.

## 3. Preparación de los datos

- **Unidad de análisis:** {n(R['hexagonos'])} hexágonos H3 de resolución 9 (≈{n(R['area_media_hex_km2'], 2)} km², unas 3 × 3 manzanas).
  Todos tienen el mismo tamaño y los mismos vecinos, lo que reduce el problema de la unidad de área modificable frente a
  unidades administrativas desiguales.
- **Residentes por hexágono:** interpolación areal desde los radios censales (se conservan los totales). Los porcentajes se
  calculan agregando numeradores y denominadores, nunca promediando porcentajes.
- **Área de influencia:** variables y objetivo se suman sobre el hexágono y sus 6 vecinos (≈400 m), la escala de una caminata.
- **Variables de demanda (14):** residentes, comercio (OSM y parcelas), oficinas, parcelas residenciales, pasajeros de
  subte, estaciones de tren, distancia al centro y al subte, % NBI, % sin internet, % de 65+, % universitarios, tasa de empleo.
  Se excluyen deliberadamente sucursales y cajeros: son oferta, no demanda.
- **Distancias:** {red_txt}

## 4. Modelado

### 4.1 Regresión de conteos: ¿cuántas terminales tienen las zonas con esta demanda?
Objetivo: terminales (todas las redes) en el área de influencia. Seis modelos más un ensamble definido de antemano,
evaluados con **validación cruzada espacial** (GroupKFold por comuna, 5 particiones) y, a modo de comparación, aleatoria.

| Modelo | D² Poisson espacial | D² Poisson aleatoria | MAE espacial | R² espacial |
|---|---|---|---|---|
{tabla_modelos}

**Elección: {R['modelo_elegido']}.** Con validación aleatoria habría ganado {R['modelo_mejor_aleatoria']}
(D² = {n(ale.loc[R['modelo_mejor_aleatoria'], 'D² Poisson'], 2)}), pero en zonas no vistas cae a
{n(esp.loc[R['modelo_mejor_aleatoria'], 'D² Poisson'], 2)}: los modelos de árboles memorizan la geografía (autocorrelación espacial).
El GLM generaliza mejor (D² = {n(R['d2_espacial'], 2)}) y además se puede explicar: subir un desvío las parcelas comerciales multiplica
las terminales esperadas por {ef('parcelas_comercio_k1')}; los comercios de OSM, por {ef('actividad_osm_k1')}; el % de 65+, por
{ef('pct_65_k1')}; los pasajeros de subte, por {ef('pax_subte_habil_k1')}; las parcelas residenciales, por {ef('parcelas_residencial_k1')};
y el % de hogares con NBI, por {ef('pct_nbi_k1')}.

**Brecha** = terminales esperadas (predicción fuera de muestra) − terminales reales. **Brecha Link** = lo esperado × participación
de Link en la ciudad ({n(100 * R['cuota_link'], 1)} %) − terminales Link reales. Barrios con más terminales Link faltantes:

| Barrio | Residentes | Terminales Link | Terminales Banelco | Participación Link | Link faltantes |
|---|---|---|---|---|---|
{tabla_top}

### 4.2 K-Means: tipos de zona
k = {R['k_clusters']} por silueta ({n(R['silueta'], 2)}) y codo, sobre variables de demanda estandarizadas (conteos en escala log).

| Tipo de zona | Residentes | Pasajeros de subte/día | Terminales cada 10 mil | Link cada 10 mil | NBI (mediana) |
|---|---|---|---|---|---|
{tabla_zonas}

### 4.3 Optimización: dónde sumar cajeros
**Problema de cobertura máxima** (MCLP; Church y ReVelle, 1974): elegir N sitios (hexágonos con comercio) que maximicen los
vecinos que pasan a tener un Link a menos de 500 m. Se resuelve con el **algoritmo goloso** (garantía de 1 − 1/e del óptimo por
submodularidad; Nemhauser, Wolsey y Fisher, 1978) y de forma **exacta** como programa lineal entero con HiGHS.

| Cajeros | Goloso (vecinos) | Exacto (vecinos) | Diferencia | Sitios en común | Segundos (exacto) |
|---|---|---|---|---|---|
{tabla_comp}

El goloso queda a menos del {n(R['goloso_max_diferencia_pct'], 1)} % del óptimo. Las 20 ubicaciones óptimas:

| # | Barrio | Comuna | Tipo de zona | Vecinos que suma |
|---|---|---|---|---|
{tabla_ex}

### 4.4 Demanda por hora
Con los molinetes 2025 ({n(R['pax_subte_habil'])} entradas por día hábil), los picos son a las **{R['hora_pico_manana']}:00** y las
**{R['hora_pico_tarde']}:00**. Un K-Means sobre la forma de la curva horaria de cada estación separa estaciones de **origen** (pico a la
mañana, barrios) y de **destino** (pico a la tarde, centro). {R['estaciones'] - R['estaciones_sin_link']} de {R['estaciones']}
estaciones ya tienen un Link a menos de 500 m. Sumar a los pasajeros de la hora pico no cambia las ubicaciones óptimas: el
faltante está en los barrios, no en el transporte.

## 5. Evaluación

- **Generalización:** D² espacial = {n(R['d2_espacial'], 2)}, MAE = {n(R['mae_espacial'], 1)} terminales por área de influencia.
- **Validación temporal:** de los {R['cajeros_osm_2026']} cajeros que OpenStreetMap tiene mapeados en 2026, {R['cajeros_nuevos_osm']} no
  estaban en el listado 2017. Cayeron, en promedio, en zonas con más brecha que el {n(100 * R['percentil_cajeros_nuevos'])} % de la
  ciudad (p = {n(R['p_valor_cajeros_nuevos'], 3)}, Monte Carlo). El AUC es {n(R['auc_validacion'], 2)}, contra 0,51 de usar solo la población.
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
"""
(RAIZ / "docs" / "informe.md").write_text(texto, encoding="utf-8")
print("ok docs/informe.md")
