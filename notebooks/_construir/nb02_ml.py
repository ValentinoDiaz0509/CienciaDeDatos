"""Genera notebooks/02_machine_learning.ipynb (correr: python notebooks/_construir/nb02_ml.py)."""
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SALIDA = Path(__file__).resolve().parents[1] / "02_machine_learning.ipynb"
TEXTOS = Path(__file__).resolve().parent / "textos_ml.py"
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
T = {"RAIZ": RAIZ}
if TEXTOS.exists():
    exec(TEXTOS.read_text(encoding="utf-8"), T)


def texto(clave, defecto=""):
    return T.get(clave, defecto)


cells = [
    md("""# 02 · Machine learning: ¿cuántos cajeros le corresponden a cada zona?

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ValentinoDiaz0509/CienciaDeDatos/blob/main/notebooks/02_machine_learning.ipynb)

**TPO Ciencia de Datos · ¿Dónde faltan cajeros Link en CABA?**

En la clase 2 vimos cuatro niveles de análisis. Este notebook cubre el **predictivo** y el **diagnóstico** que lo explica:

| Nivel | Pregunta | Dónde |
|---|---|---|
| Descriptivo | ¿Dónde están los cajeros y dónde está la gente? | notebook 01 |
| Diagnóstico | ¿Qué hace que una zona tenga más cajeros? | este notebook (§4 y §6) |
| **Predictivo** | ¿Cuántos cajeros le corresponden a cada zona? ¿Cuánta gente va a viajar? | este notebook y el 05 |
| Prescriptivo | ¿Dónde conviene poner los próximos cajeros? | notebook 03 |

Dos preguntas de negocio, dos modelos supervisados:

| Pregunta | Tipo | Objetivo (y) | Modelos comparados |
|---|---|---|---|
| ¿Cuántas terminales le corresponden a cada zona según su demanda? | Regresión | terminales en el área de influencia | regresión de Poisson, árbol de decisión (clase 6), Random Forest y XGBoost (clase 8), ensamble |
| ¿A esta zona le corresponde al menos un cajero? | Clasificación | la zona tiene o no un cajero | regresión logística, árbol de decisión, Random Forest, XGBoost |

Todo se evalúa con **validación cruzada espacial anidada**: el modelo siempre se mide en comunas que no vio."""),
    code("""import json, sys, time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

if "google.colab" in sys.modules:   # en Colab: clona el repositorio e instala lo que falta
    import subprocess
    RAIZ = Path("/content/CienciaDeDatos")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/ValentinoDiaz0509/CienciaDeDatos.git", str(RAIZ)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h3", "xgboost"], check=True)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import estilo as e, features as f, graficos_ml as gm, ml, modelos as m

e.mpl_estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
FIG = RAIZ / "docs/figuras"; FIG.mkdir(parents=True, exist_ok=True)
PROC = RAIZ / "data/processed"

# True: entrena y valida todos los modelos acá mismo (unos 6 minutos con 2 núcleos).
# False: muestra lo que ya calculó el pipeline (python -m src.pipeline), en segundos.
# Da lo mismo: todo usa semilla fija, así que los números coinciden.
RECALCULAR = True

tabla = pd.read_parquet(PROC / "hexagonos.parquet")      # variables por hexágono (src/features.py)
res = pd.read_parquet(PROC / "resultados_hex.parquet")   # + predicciones fuera de muestra del pipeline
barrios = f.cargar_barrios()
X, y, grupos, mascara = m.preparar(tabla)
print(f"{e.num_es(len(X))} hexágonos para modelar · {X.shape[1]} variables · objetivo: terminales en el área de influencia")"""),
    md("""## 1. Planteo

- **Unidad:** hexágono H3 de resolución 9 (unas 3 × 3 manzanas). Variables y objetivo se miden en su **área de influencia**:
  el hexágono y sus 6 vecinos, unos 400 m, la escala de una caminata.
- **Objetivo (regresión):** terminales de cajero de cualquier red en el área. Es un **conteo** con muchos ceros y cola larga,
  así que usamos pérdida de Poisson y medimos con la **devianza de Poisson (D²)**, el equivalente del R² para conteos.
  También reportamos el **error absoluto medio (MAE)**: cuántas terminales erra, en promedio, por zona.
- **Variables:** solo de **demanda**: residentes, perfil socioeconómico (censo), comercio y oficinas (usos del suelo y
  OpenStreetMap), pasajeros de subte (molinetes) y distancias. Dejamos afuera sucursales y cajeros: son oferta, y el
  modelo terminaría explicando cajeros con cajeros."""),
    code("""pd.DataFrame({"variable": X.columns, "descripción": [ml.NOMBRES.get(c, c) for c in X.columns],
              "mediana": X.median().values, "máximo": X.max().values})"""),
    md("""### Cómo separamos entrenamiento y prueba

Las zonas vecinas se parecen y comparten área de influencia. Si mezclamos al azar entrenamiento y prueba, el modelo "ve" la
respuesta en los vecinos y el error parece menor de lo que es. Por eso hacemos **validación cruzada espacial**: armamos
5 grupos de comunas (GroupKFold) y, por turno, entrenamos con 4 y medimos en el quinto.

Además es **anidada**: los hiperparámetros (profundidad de los árboles, cantidad de hojas, regularización) se eligen con
GridSearchCV **adentro** de las 4 partes de entrenamiento, también dejando comunas afuera. Así el grupo de prueba nunca
participa de ninguna decisión del modelo."""),
    code("""particion = pd.Series(0, index=X.index)
for k, (_, te) in enumerate(GroupKFold(n_splits=ml.PLIEGUES).split(X, y, grupos), start=1):
    particion.iloc[te] = k
g = gpd.GeoDataFrame({"particion": particion.values}, geometry=[f.poligono_hex(h) for h in tabla.loc[mascara, "h3"]], crs="EPSG:4326")
fig, ax = plt.subplots(figsize=(9, 8.5))
for k in range(1, ml.PLIEGUES + 1):
    comunas = sorted(grupos[particion == k].unique())
    g[g["particion"] == k].plot(ax=ax, color=e.CATEGORICO[k - 1], linewidth=0, label=f"Grupo {k}: comunas {', '.join(map(str, comunas))}")
barrios.boundary.plot(ax=ax, color=e.FONDO, linewidth=0.6)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=e.CATEGORICO[k - 1], label=f"Grupo {k}: comunas {', '.join(map(str, sorted(grupos[particion == k].unique())))}")
                   for k in range(1, ml.PLIEGUES + 1)], loc="lower left")
ax.set_axis_off(); ax.set_title("Validación cruzada espacial: 5 grupos de comunas")
fig.tight_layout(); fig.savefig(FIG / "ml_particion_espacial.png"); plt.show()"""),
    md("""## 2. Regresión: ¿cuántas terminales le corresponden a cada zona?

Comparamos, de menor a mayor complejidad:

| Modelo | Qué es | Hiperparámetros ajustados |
|---|---|---|
| Línea de base | predice siempre el promedio | — |
| Poisson solo con población | la regla simple "más vecinos, más cajeros" | — |
| Regresión de Poisson (GLM) | modelo lineal para conteos (escala logarítmica) | regularización α |
| Árbol de decisión (CART) | clase 6, con criterio de Poisson | profundidad, mínimo de zonas por hoja |
| Random Forest | clase 8: 300 árboles con bootstrap y variables al azar | mínimo por hoja, variables por corte |
| XGBoost | clase 8: boosting, árboles que corrigen el error de los anteriores | profundidad, peso mínimo por hoja, regularización λ |
| Ensamble (Poisson + XGBoost) | promedio en escala logarítmica del GLM y XGBoost | los de XGBoost |

**Regla de elección, fijada de antemano:** el mayor D² de Poisson promedio entre los 5 grupos de comunas."""),
    code("""t0 = time.time()
if RECALCULAR:
    metricas, oof, params, ajustes, _ = ml.comparar_regresion(X, y, grupos)
    print(f"Validación cruzada anidada terminada en {time.time() - t0:.0f} s")
else:
    metricas = pd.read_csv(PROC / "metricas_modelos.csv")
esp = metricas[metricas["validación"].str.startswith("espacial")].set_index("modelo")
ale = metricas[metricas["validación"] == "aleatoria"].set_index("modelo")
orden = esp.sort_values("D² Poisson (promedio por partición)", ascending=False).index
ganador = orden[0]
pd.DataFrame({
    "D² espacial (promedio)": esp.loc[orden, "D² Poisson (promedio por partición)"],
    "D² espacial (desvío)": esp.loc[orden, "D² Poisson (desvío entre particiones)"],
    "MAE espacial (terminales)": esp.loc[orden, "MAE"],
    "R² espacial": esp.loc[orden, "R²"],
    "D² aleatoria (promedio)": ale.loc[orden, "D² Poisson (promedio por partición)"],
})"""),
    code("""fig, ax = plt.subplots(figsize=(11, 5))
yy = np.arange(len(orden))
ax.barh(yy - 0.2, ale.loc[orden, "D² Poisson (promedio por partición)"], height=0.38, color=e.EJE,
        label="Validación aleatoria (optimista)")
ax.barh(yy + 0.2, esp.loc[orden, "D² Poisson (promedio por partición)"], height=0.38, color=e.LINK,
        xerr=esp.loc[orden, "D² Poisson (desvío entre particiones)"], error_kw={"ecolor": e.TINTA_SUAVE, "linewidth": 1, "capsize": 2},
        label="Validación espacial (comunas no vistas)")
for i, (v, sd) in enumerate(zip(esp.loc[orden, "D² Poisson (promedio por partición)"], esp.loc[orden, "D² Poisson (desvío entre particiones)"])):
    ax.text(max(v + sd, 0) + 0.02, i + 0.2, e.num_es(v, 2), va="center", fontsize=9, color=e.TINTA)
ax.set_yticks(yy, orden); ax.invert_yaxis()
ax.set_xlim(min(-0.3, esp["D² Poisson (promedio por partición)"].min() - 0.1), 1)
ax.axvline(0, color=e.TINTA_SUAVE, linewidth=0.8)
ax.set(title="¿Cuánto de la variación en terminales explica cada modelo? (D² de Poisson)", xlabel="D² de Poisson, promedio de las 5 particiones")
ax.legend(loc="lower right"); e.ejes_es(ax, x=1); fig.tight_layout(); fig.savefig(FIG / "ml_comparacion.png"); plt.show()
print("Modelo elegido:", ganador)"""),
    code("""hp = pd.read_csv(PROC / "hiperparametros.csv") if not RECALCULAR else pd.DataFrame(
    [{"modelo": k, "partición": i + 1, **p} for k, v in params.items() for i, p in enumerate(v)])
hp = hp.dropna(axis=1, how="all")
hp[hp["modelo"].isin(["Árbol de decisión (CART)", "Random Forest", "XGBoost", "Ensamble (Poisson + XGBoost)"])].set_index(["modelo", "partición"])"""),
    md(texto("comparacion", "")),
    md("""## 3. Sobreajuste (clase 6) y error OOB (clase 8)

**Sobreajuste.** Un árbol más profundo siempre se ajusta mejor a los datos con los que entrena, pero a partir de cierta
profundidad empieza a memorizar ruido y predice peor en zonas nuevas. Lo medimos entrenando árboles de 1 a 15 niveles.

**OOB (out-of-bag).** En un Random Forest cada árbol se entrena con una muestra bootstrap y deja afuera, en promedio, un
tercio de las filas: con ellas se calcula un error "gratis". Comparamos ese OOB con el error en comunas no vistas."""),
    code("""curva = ml.curva_profundidad(X, y, grupos) if RECALCULAR else pd.read_csv(PROC / "curva_profundidad.csv")
mejor = curva.loc[curva["validación espacial"].idxmax()]
fig, ax = plt.subplots(figsize=(10, 4.3))
ax.plot(curva["profundidad"], curva["entrenamiento"], color=e.TINTA_SUAVE, linewidth=2, marker="o", markersize=4, label="Entrenamiento")
ax.plot(curva["profundidad"], curva["validación espacial"], color=e.LINK, linewidth=2.4, marker="o", markersize=5, label="Comunas no vistas")
ax.axvspan(mejor["profundidad"] + 0.5, curva["profundidad"].max() + 0.5, color=e.GRILLA, alpha=0.6, zorder=0)
ax.annotate(f"mejor profundidad: {int(mejor['profundidad'])}", (mejor["profundidad"], mejor["validación espacial"]),
            xytext=(0, -26), textcoords="offset points", ha="center", fontsize=9, color=e.TINTA)
ax.text(curva["profundidad"].max(), curva["validación espacial"].min(), "sobreajuste: mejora en entrenamiento,\\nempeora en zonas nuevas",
        ha="right", va="bottom", fontsize=9, color=e.TINTA_2)
ax.set(title="Árbol de decisión: profundidad contra error", xlabel="profundidad del árbol", ylabel="D² de Poisson")
ax.legend(loc="upper left"); e.ejes_es(ax, y=1); fig.tight_layout(); fig.savefig(FIG / "ml_curva_arbol.png"); plt.show()"""),
    code("""if RECALCULAR:
    Xc, yc = ml.preparar_clasificacion(tabla, X, mascara)
    oob = ml.oob_vs_validacion(X, y, grupos, Xc, yc, ml.params_mas_frecuentes(params["Random Forest"]),
                               ml.params_mas_frecuentes(pd.read_csv(PROC / "hiperparametros_clasificacion.csv")
                                                        .query("modelo == 'Random Forest'")[["min_samples_leaf", "max_features"]]
                                                        .astype({"min_samples_leaf": int}).to_dict("records")))
else:
    oob = pd.read_csv(PROC / "oob_vs_validacion.csv")
oob"""),
    md(texto("sobreajuste", "")),
    md("""## 4. Qué aprendió el modelo (diagnóstico)

Tres miradas que se complementan:

- **Valores SHAP** del modelo elegido: para cada zona, cuánto empuja cada variable la predicción hacia arriba o hacia abajo
  respecto de una zona típica. El promedio de su valor absoluto mide la importancia de la variable. XGBoost los calcula de
  forma exacta (TreeSHAP); en la parte de Poisson es coeficiente × valor.
- **Importancia de Gini** (reducción de impureza) del Random Forest, la que vimos en la clase 8.
- **Coeficientes de la regresión de Poisson:** cuánto se multiplica lo esperado al subir un desvío cada variable."""),
    code("""imp_shap = pd.read_csv(PROC / "importancia_variables.csv").sort_values("importancia")
imp_gini = pd.read_csv(PROC / "importancia_gini.csv").sort_values("importancia")
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
ax[0].barh(imp_shap["nombre"], imp_shap["importancia"], color=e.LINK, height=0.65)
ax[0].set(title=f"SHAP · {ganador}", xlabel="aporte promedio (escala log; 0,1 ≈ ±10 %)")
ax[1].barh(imp_gini["nombre"], imp_gini["importancia"], color=e.TINTA_SUAVE, height=0.65)
ax[1].set(title="Gini · Random Forest", xlabel="reducción de impureza (proporción)")
for a in ax:
    a.grid(axis="y", visible=False); e.ejes_es(a, x=2)
fig.tight_layout(); fig.savefig(FIG / "ml_importancia.png"); plt.show()"""),
    code("""coef = pd.read_csv(PROC / "coeficientes_glm.csv")
fig, ax = plt.subplots(figsize=(10, 5))
colores = [e.DIVERGENTE[-2] if c > 0 else e.DIVERGENTE[1] for c in coef["coeficiente"]]
ax.barh(coef["nombre"], coef["efecto"] - 1, left=1, color=colores)
ax.axvline(1, color=e.TINTA_SUAVE, linewidth=1)
ax.set(title="Regresión de Poisson: cuánto se multiplican las terminales esperadas al subir 1 desvío cada variable",
       xlabel="efecto multiplicativo (1 = sin efecto)")
e.ejes_es(ax, x=1); fig.tight_layout(); fig.savefig(FIG / "modelo_coeficientes.png"); plt.show()
coef[["nombre", "coeficiente", "efecto"]].round(3)"""),
    md(texto("importancia", "")),
    md("""## 5. Cuántos cajeros le corresponden a cada barrio

Con el modelo elegido predecimos, para cada zona, cuántas terminales le corresponden según su demanda. Usamos las
predicciones **fuera de muestra**: cada zona la predice un modelo que no vio su comuna, así no se "explica" con sus propios
cajeros.

- **Brecha** = terminales que le corresponden − terminales que tiene.
- **Brecha Link** = lo que Link tendría con su participación en la ciudad − lo que tiene. Positivo = faltan terminales Link.

Para sumar por barrio, la predicción de cada área de influencia se reparte entre sus 7 hexágonos (si no, cada terminal se
contaría siete veces)."""),
    code("""g = gpd.GeoDataFrame(res, geometry=[f.poligono_hex(h) for h in res["h3"]], crs="EPSG:4326")
lim = np.nanquantile(np.abs(g["brecha_link"]), 0.95)
fig, ax = plt.subplots(figsize=(10, 9))
g.plot(column="brecha_link", cmap=e.cmap_divergente(), vmin=-lim, vmax=lim, ax=ax, linewidth=0, missing_kwds={"color": "#ffffff"},
       legend=True, legend_kwds={"label": "terminales Link: faltan (+) / sobran (−)", "shrink": 0.55})
barrios.boundary.plot(ax=ax, color=e.TINTA_SUAVE, linewidth=0.4)
ax.set_axis_off(); ax.set_title(f"Brecha Link por zona · {ganador}")
fig.tight_layout(); fig.savefig(FIG / "mapa_brecha_link.png"); plt.show()
print(f"Participación de Link en las terminales de la ciudad: {e.num_es(100 * res['term_link'].sum() / res['term_total'].sum(), 1)} %")"""),
    code("""bp = pd.read_csv(PROC / "barrios_prediccion.csv", index_col="barrio")
vista = bp[["residentes", "comercios", "pasajeros_subte", "tiene_total", "corresponden_total", "tiene_link",
            "corresponden_link", "diferencia_link"]].round(1)
vista.head(12)"""),
    code("""top = bp[bp["diferencia_link"] > 0].head(12)
fig, ax = plt.subplots(figsize=(10, 5.6))
gm.tiene_vs_corresponden(top, ax)
ax.set_title("Barrios donde a Link le faltan más terminales según su demanda")
fig.tight_layout(); fig.savefig(FIG / "ml_corresponden_barrios.png"); plt.show()
bp.tail(5)[["tiene_link", "corresponden_link", "diferencia_link"]].round(1)"""),
    md(texto("barrios", "")),
    md("""## 6. Por qué: la explicación de cada zona (SHAP)

El modelo elegido es aditivo en escala logarítmica, así que la predicción de cada zona se descompone **exactamente** en el
aporte de cada variable: se parte de una zona típica y cada variable la multiplica por un factor (×2 duplica, ×0,5 reduce a
la mitad). Elegimos tres zonas con reglas fijas:

- **la de mayor faltante:** donde a Link más terminales le corresponden y no tiene ninguna;
- **una que alcanza con un cajero:** una zona residencial sin ningún cajero a la que le corresponde uno de Link;
- **una con mucha gente y poco flujo:** la zona de más de 15 mil vecinos a la que menos cajeros le corresponden."""),
    code("""ap = pd.read_parquet(PROC / "aportes_hex.parquet").set_index("h3")
r = res[res["esperado_total_k1"].notna()].set_index("h3")
Xh = X.set_index(tabla.loc[mascara, "h3"])
est = pd.read_csv(PROC / "estaciones_subte.csv")

def cerca(h):
    d = np.hypot((est["lat"] - r.loc[h, "lat"]) * 111_000, (est["lon"] - r.loc[h, "lon"]) * 92_000)
    i = d.idxmin()
    return f"a {e.num_es(round(d[i], -2))} m de {est.loc[i, 'estacion']} (línea {est.loc[i, 'linea']})"

zona_alta = r[r["term_link_k1"] == 0]["esperado_link_k1"].idxmax()
uno = r[(r["term_total_k1"] == 0) & r["esperado_link_k1"].between(0.5, 1.5) & (r["tipo_zona"] == "Residencial denso")]
zona_uno = uno["poblacion_k1"].idxmax()
zona_baja = r[r["poblacion_k1"] >= 15_000]["esperado_total_k1"].idxmin()
zonas = {"alta": zona_alta, "uno": zona_uno, "baja": zona_baja}
for clave, h in zonas.items():
    z = r.loc[h]
    print(f"{clave}: {z['barrio']} ({cerca(h)}) · {e.num_es(z['poblacion_k1'])} vecinos, {e.num_es(z['actividad_osm_k1'])} comercios, "
          f"{e.num_es(z['pax_subte_habil_k1'])} pasajeros de subte · le corresponden {e.num_es(z['esperado_total_k1'], 1)} "
          f"terminales ({e.num_es(z['esperado_link_k1'], 1)} Link) · tiene {e.num_es(z['term_total_k1'])} ({e.num_es(z['term_link_k1'])} Link)")"""),
    code("""fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for a, clave in zip(ax, ["alta", "uno"]):
    h = zonas[clave]; z = r.loc[h]
    tabla_ap = ml.explicar_zona(ap, Xh, h, n=5)
    gm.explicacion(tabla_ap, a, f"{z['barrio']}: le corresponden {e.num_es(z['esperado_total_k1'], 1)} terminales "
                                f"({e.num_es(z['esperado_link_k1'], 1)} Link) · tiene {e.num_es(z['term_total_k1'])} ({e.num_es(z['term_link_k1'])} Link)")
    a.set_xlabel("cuánto multiplica cada variable lo esperado, frente a una zona típica (×1 = no cambia)")
fig.tight_layout(); fig.savefig(FIG / "ml_explicacion_zonas.png"); plt.show()
ml.explicar_zona(ap, Xh, zonas["baja"], n=5).round(3)"""),
    md(texto("zonas", "")),
    md("""## 7. Clasificación: ¿a esta zona le corresponde al menos un cajero?

Misma idea, como clasificación: la zona (el hexágono) tiene o no al menos un cajero de cualquier red. Al mercado lo
tomamos como referencia: si zonas con demanda parecida tienen cajero, esta debería tener uno. Agregamos variables del propio
hexágono (residentes, comercios, pasajeros) a las del área.

- **Métricas:** AUC (probabilidad de que una zona con cajero tenga más puntaje que una sin cajero; 0,5 = azar) y PR-AUC.
- **Umbral:** el modelo marca tantas zonas como las que hoy tienen cajero (umbral de la tasa base). Con ese umbral precisión
  y recall coinciden.
- **Lectura de negocio de la matriz de confusión:** los **falsos positivos** (el modelo dice "debería tener" y no tiene) son
  **oportunidades**."""),
    code("""Xc, yc = ml.preparar_clasificacion(tabla, X, mascara)
if RECALCULAR:
    met_c, oof_c, params_c, _ = ml.comparar_clasificacion(Xc, yc, grupos)
    clasif_oof = pd.DataFrame({"h3": tabla.loc[mascara, "h3"].to_numpy(), "real": yc.to_numpy(), **oof_c})
else:
    met_c = pd.read_csv(PROC / "metricas_clasificacion.csv")
    clasif_oof = pd.read_parquet(PROC / "clasificacion_oof.parquet")
esp_c = met_c[met_c["validación"].str.startswith("espacial")].set_index("modelo")
ale_c = met_c[met_c["validación"] == "aleatoria"].set_index("modelo")
orden_c = esp_c.sort_values("AUC (promedio por partición)", ascending=False).index
clasificador = orden_c[0]
pd.DataFrame({"AUC espacial (promedio)": esp_c.loc[orden_c, "AUC (promedio por partición)"],
              "AUC espacial": esp_c.loc[orden_c, "AUC"], "PR-AUC espacial": esp_c.loc[orden_c, "PR-AUC"],
              "Precisión = Recall": esp_c.loc[orden_c, "Precisión"], "AUC aleatoria": ale_c.loc[orden_c, "AUC"]})"""),
    code("""umbral = ml.umbral_tasa_base(clasif_oof["real"].to_numpy(), clasif_oof[clasificador].to_numpy())
cm = ml.matriz(clasif_oof["real"].to_numpy(), clasif_oof[clasificador].to_numpy(), umbral)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.1, 1]})
gm.roc(clasif_oof, list(orden_c), clasificador, ax[0])
ax[0].set_title("Curvas ROC en comunas no vistas")
gm.matriz(cm, ax[1])
ax[1].set_title(f"Matriz de confusión · {clasificador} (umbral {e.num_es(umbral, 2)})")
fig.tight_layout(); fig.savefig(FIG / "ml_clasificacion.png"); plt.show()
cm"""),
    code("""arbol = ml.arbol_explicativo(Xc, yc, profundidad=3)
fig, ax = plt.subplots(figsize=(16, 6.2))
gm.arbol(arbol, ml.NOMBRES, ax)
ax.set_title("Árbol de decisión de 3 niveles: las reglas que separan zonas con y sin cajero")
fig.tight_layout(); fig.savefig(FIG / "ml_arbol.png"); plt.show()"""),
    code("""rr = res.merge(clasif_oof[["h3", clasificador]].rename(columns={clasificador: "p"}), on="h3", how="left")
rr["clase"] = np.select([rr["term_total"] > 0, rr["p"] >= umbral], ["con cajero", "oportunidad"], "sin cajero, sin demanda suficiente")
gg = gpd.GeoDataFrame(rr, geometry=[f.poligono_hex(h) for h in rr["h3"]], crs="EPSG:4326")
fig, ax = plt.subplots(figsize=(10, 9))
gg[gg["clase"] == "sin cajero, sin demanda suficiente"].plot(ax=ax, color=e.GRILLA, linewidth=0)
gg[gg["clase"] == "con cajero"].plot(ax=ax, color=e.TINTA_SUAVE, linewidth=0)
gg[gg["clase"] == "oportunidad"].plot(ax=ax, color=e.SUGERIDO, linewidth=0)
barrios.boundary.plot(ax=ax, color=e.TINTA_2, linewidth=0.4)
ax.legend(handles=[Patch(color=e.SUGERIDO, label="Oportunidad: el modelo dice que debería tener cajero y no tiene"),
                   Patch(color=e.TINTA_SUAVE, label="Tiene al menos un cajero"),
                   Patch(color=e.GRILLA, label="Sin cajero y sin demanda suficiente")], loc="lower left")
ax.set_axis_off(); ax.set_title("Zonas sin cajero que, por su demanda, deberían tener uno")
fig.tight_layout(); fig.savefig(FIG / "ml_mapa_oportunidades.png"); plt.show()
op = rr[rr["clase"] == "oportunidad"]
print(f"{len(op)} zonas sin ningún cajero que deberían tener uno · {e.num_es(op['poblacion'].sum())} vecinos")
op.groupby("barrio").size().sort_values(ascending=False).head(10)"""),
    md(texto("clasificacion", "")),
    md("""## 8. Validación hacia el futuro

El listado de cajeros es de ≈2017. En OpenStreetMap hay cajeros mapeados hasta 2026: los que están a más de 150 m de
cualquier cajero del listado son, muy probablemente, **instalados después**. Si los modelos, entrenados con la foto de 2017,
predicen algo real, esos cajeros nuevos tendrían que haber aparecido donde marcaban brecha o probabilidad alta."""),
    code("""val = pd.read_csv(PROC / "validacion_temporal.csv")
R = json.loads((PROC / "resumen.json").read_text(encoding="utf-8"))
print(f"Cajeros nuevos en OSM: {R['cajeros_nuevos_osm']} · cayeron en el percentil {e.num_es(100 * R['percentil_cajeros_nuevos'])} "
      f"de la brecha 2017 (p = {e.num_es(R['p_valor_cajeros_nuevos'], 3)}, 20.000 sorteos al azar)")
val.round(3)"""),
    md(texto("validacion", "")),
    md("## 9. Conclusión\n\n" + texto("conclusion", "")),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, SALIDA)
print("ok", SALIDA)
