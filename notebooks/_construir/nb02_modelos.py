"""Genera notebooks/02_modelos.ipynb (correr: python notebooks/_construir/nb02_modelos.py)."""
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SALIDA = Path(__file__).resolve().parents[1] / "02_modelos.ipynb"
TEXTOS = Path(__file__).resolve().parent / "textos_nb02.py"
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
T = {"RAIZ": RAIZ}
if TEXTOS.exists():
    exec(TEXTOS.read_text(encoding="utf-8"), T)


def texto(clave, defecto=""):
    return T.get(clave, defecto)


cells = [
    md("""# 02 · Modelos: brecha, tipos de zona y dónde sumar cajeros

**TPO Ciencia de Datos · ¿Dónde faltan cajeros Link en CABA?**

Fases de CRISP-DM: **preparación**, **modelado** y **evaluación**. Usamos tres técnicas y cada una responde una pregunta
distinta del negocio:

| Técnica | Tipo | Pregunta |
|---|---|---|
| Regresión de conteos | supervisada | ¿Cuántas terminales tienen las zonas con esta demanda? → **brecha** |
| K-Means | no supervisada | ¿Qué tipos de zona hay y dónde se concentra la brecha? |
| Cobertura máxima (MCLP) | prescriptiva | Si se suman N cajeros, ¿dónde ubicarlos para llegar a más gente? |

Al final validamos contra datos que el modelo no vio: los cajeros que aparecieron en OpenStreetMap después de 2017."""),
    code("""import json, sys, time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))
from src import estilo as e, features as f, modelos as m, red as rp

e.mpl_estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
FIG = RAIZ / "docs/figuras"; FIG.mkdir(parents=True, exist_ok=True)

tabla = f.construir()  # variables por hexágono, a partir de data/base
cajeros = f.cargar_cajeros()
HAY_RED = rp.hay_red()
if HAY_RED:  # distancias caminando por la red peatonal de OpenStreetMap
    red = rp.RedPeatonal.desde_archivos()
    tabla = rp.agregar_distancias_red(tabla, cajeros, red)
SUF = "_red" if HAY_RED else ""
print("Distancias:", "caminando por las calles" if HAY_RED else "en línea recta")
barrios = f.cargar_barrios()
print(f"{e.num_es(len(tabla))} hexágonos · {e.num_es(tabla['poblacion'].sum())} residentes · {e.num_es(tabla['term_total'].sum())} terminales")"""),
    md("""## 1. Planteo del modelo

- **Unidad:** hexágono H3 (resolución 9). Variables y objetivo se miden en el **área de influencia** (hexágono + 6 vecinos, ≈400 m).
- **Objetivo:** terminales de cajero (de cualquier red) en el área de influencia. Es un **conteo**, con muchos ceros y cola larga,
  por eso usamos modelos con pérdida de Poisson y medimos con la **devianza de Poisson** (D²: proporción explicada, el R² de los conteos).
- **Variables:** solo de **demanda** (residentes, perfil socioeconómico, comercio, pasajeros, distancia al centro y al subte).
  Dejamos afuera las sucursales bancarias: son oferta, y el modelo terminaría explicando cajeros con cajeros.
- **Filas:** hexágonos con algo de demanda (se excluyen río, reservas y playones vacíos)."""),
    code("""X, y, grupos, mascara = m.preparar(tabla)
print(f"{e.num_es(len(X))} hexágonos para modelar · {X.shape[1]} variables")
pd.DataFrame({"variable": X.columns, "descripción": [m.NOMBRES.get(c, c) for c in X.columns],
              "mediana": X.median().values, "máximo": X.max().values})"""),
    md("""## 2. Comparación de modelos con validación cruzada espacial

Los hexágonos vecinos se parecen (autocorrelación espacial) y además comparten área de influencia. Si mezclamos al azar
entrenamiento y prueba, el modelo "ve" la respuesta en los vecinos y el error parece menor de lo que es. Por eso evaluamos
dejando afuera **una comuna entera por vez** (GroupKFold por comuna, 5 particiones) y lo comparamos con la validación aleatoria.

Modelos, de menor a mayor complejidad (los de clase: árbol de decisión, Random Forest y boosting, más dos referencias
estadísticas y un ensamble definido de antemano):"""),
    code("""t0 = time.time()
metricas, oof = m.comparar(X, y, grupos)
print(f"Validación terminada en {time.time() - t0:.0f} s")
tabla_metricas = metricas.pivot_table(index="modelo", columns="validación", values=["D² Poisson", "MAE", "R²"])
orden = metricas[metricas["validación"].str.startswith("espacial")].sort_values("D² Poisson", ascending=False)["modelo"]
tabla_metricas.loc[orden]"""),
    code("""esp = metricas[metricas["validación"].str.startswith("espacial")].set_index("modelo").loc[orden]
ale = metricas[metricas["validación"] == "aleatoria"].set_index("modelo").loc[orden]
fig, ax = plt.subplots(figsize=(11, 4.8))
yy = np.arange(len(orden))
ax.barh(yy - 0.2, ale["D² Poisson"], height=0.38, color=e.EJE, label="Validación aleatoria (optimista)")
ax.barh(yy + 0.2, esp["D² Poisson"], height=0.38, color=e.LINK, label="Validación espacial (por comuna)")
for i, v in enumerate(esp["D² Poisson"]):
    ax.text(max(v, 0) + 0.01, i + 0.2, e.num_es(v, 2), va="center", fontsize=9, color=e.TINTA)
ax.set_yticks(yy, orden); ax.invert_yaxis(); ax.set_xlim(left=min(0, esp["D² Poisson"].min() - 0.05))
ax.set(title="¿Cuánto de la variación en terminales explica cada modelo? (D² de Poisson)", xlabel="D² de Poisson")
ax.legend(loc="lower right"); e.ejes_es(ax, x=1); fig.tight_layout(); fig.savefig(FIG / "modelos_comparacion.png"); plt.show()"""),
    md(texto("comparacion", "")),
    md("## 3. Modelo elegido: qué aprendió"),
    code("""ganador = orden.iloc[0]
final = clone(m.candidatos()[ganador]).fit(X, y)
imp = permutation_importance(final, X, y, n_repeats=10, random_state=m.SEMILLA, scoring="neg_mean_absolute_error", n_jobs=-1)
importancia = (pd.DataFrame({"variable": X.columns, "importancia": imp.importances_mean, "desvío": imp.importances_std})
               .assign(nombre=lambda d: d["variable"].map(m.NOMBRES)).sort_values("importancia"))
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(importancia["nombre"], importancia["importancia"], xerr=importancia["desvío"], color=e.LINK,
        error_kw={"ecolor": e.TINTA_SUAVE, "linewidth": 1})
ax.set(title=f"Importancia por permutación · {ganador}", xlabel="aumento del error absoluto medio al desordenar la variable (terminales)"); e.ejes_es(ax, x=1)
fig.tight_layout(); fig.savefig(FIG / "modelo_importancia.png"); plt.show()
print("Modelo elegido:", ganador)"""),
    code("""top = importancia.sort_values("importancia", ascending=False)["variable"].head(3).tolist()
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
PartialDependenceDisplay.from_estimator(final, X, top, ax=ax, line_kw={"color": e.LINK, "linewidth": 2})
for a, v in zip(ax, top):
    a.set_xlabel(m.NOMBRES.get(v, v)); a.set_ylabel("terminales esperadas")
fig.suptitle("Dependencia parcial: cómo cambia lo esperado al mover una variable (las demás fijas)", x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "modelo_dependencia_parcial.png"); plt.show()"""),
    code("""glm = clone(m.candidatos()["Regresión de Poisson (GLM)"]).fit(X, y)
coef = m.coeficientes_glm(glm, list(X.columns))
fig, ax = plt.subplots(figsize=(10, 5))
colores = [e.DIVERGENTE[-2] if c > 0 else e.DIVERGENTE[1] for c in coef["coeficiente"]]
ax.barh(coef["nombre"], coef["efecto"] - 1, left=1, color=colores)
ax.axvline(1, color=e.TINTA_SUAVE, linewidth=1)
ax.set(title="Regresión de Poisson: cuánto se multiplican las terminales esperadas al subir 1 desvío cada variable",
       xlabel="efecto multiplicativo (1 = sin efecto)")
e.ejes_es(ax, x=1); fig.tight_layout(); fig.savefig(FIG / "modelo_coeficientes.png"); plt.show()
coef[["nombre", "coeficiente", "efecto"]].round(3)"""),
    md(texto("interpretacion", "")),
    md("""## 4. La brecha

**Brecha total** = terminales esperadas − terminales reales en el área de influencia. Usamos las predicciones **fuera de
muestra** de la validación espacial: cada zona se estima con un modelo que no vio su comuna, así no se "explica" con sus
propios cajeros.

**Brecha Link** = lo que Link tendría con su participación promedio en la ciudad − lo que tiene. Positivo = faltan terminales Link."""),
    code("""res = m.calcular_brechas(tabla, mascara, oof[ganador])
print(f"Participación de Link en las terminales de la ciudad: {e.num_es(100 * res.attrs['cuota_link'], 1)} %")
g = gpd.GeoDataFrame(res, geometry=[f.poligono_hex(h) for h in res["h3"]], crs="EPSG:4326")
lim = np.nanquantile(np.abs(g["brecha_link"]), 0.95)
fig, ax = plt.subplots(figsize=(10, 9))
g.plot(column="brecha_link", cmap=e.cmap_divergente(), vmin=-lim, vmax=lim, ax=ax, linewidth=0, missing_kwds={"color": "#ffffff"},
       legend=True, legend_kwds={"label": "terminales Link: faltan (+) / sobran (−)", "shrink": 0.55})
barrios.boundary.plot(ax=ax, color=e.TINTA_SUAVE, linewidth=0.4)
ax.set_axis_off(); ax.set_title("Brecha Link por zona")
fig.tight_layout(); fig.savefig(FIG / "mapa_brecha_link.png"); plt.show()"""),
    code("""por_barrio = (res.groupby("barrio").agg(residentes=("poblacion", "sum"), link=("term_link", "sum"), banelco=("term_banelco", "sum"),
                                        faltan_link=("brecha_link", lambda s: s.clip(lower=0).sum() / 7))
              .assign(link_10mil=lambda d: d["link"] / d["residentes"] * 1e4)
              .sort_values("faltan_link", ascending=False))
por_barrio.head(12).round(1)"""),
    md(texto("brecha", "")),
    md("""## 5. Tipos de zona (K-Means)

Agrupamos los hexágonos según su **perfil de demanda** (residentes, comercio y pasajeros en escala logarítmica, distancia al
centro, % NBI, % de 65+ y % universitarios), todo estandarizado. Elegimos k mirando la inercia (método del codo) y la
**silueta** (qué tan separado está cada grupo de los demás)."""),
    code("""Z = m.matriz_cluster(res, mascara)
ks = m.elegir_k(Z)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(ks["k"], ks["inercia"], marker="o", color=e.LINK, linewidth=2); ax[0].set(title="Método del codo", xlabel="k", ylabel="inercia")
ax[1].plot(ks["k"], ks["silueta"], marker="o", color=e.LINK, linewidth=2); ax[1].set(title="Silueta promedio", xlabel="k", ylabel="silueta")
fig.tight_layout(); fig.savefig(FIG / "clusters_k.png"); plt.show()
k = int(ks[ks["k"].between(3, 7)].sort_values("silueta", ascending=False).iloc[0]["k"])
print("k elegido:", k)
ks.round(3)"""),
    code("""res.loc[mascara, "cluster"] = m.agrupar(Z, k)
nombres = m.nombrar_clusters(res.loc[mascara])
res["tipo_zona"] = res["cluster"].map(nombres)
perfil = (res.loc[mascara].groupby("tipo_zona")
          .agg(hexágonos=("h3", "count"), residentes=("poblacion", "sum"), pasajeros_subte=("pax_subte_habil", "sum"),
               terminales=("term_total", "sum"), terminales_link=("term_link", "sum"),
               dist_centro_km=("dist_centro_km", "median"), pct_nbi=("pct_nbi_k1", "median"), pct_univ=("pct_univ_k1", "median"))
          .assign(terminales_cada_10mil=lambda d: d["terminales"] / d["residentes"] * 1e4,
                  link_cada_10mil=lambda d: d["terminales_link"] / d["residentes"] * 1e4,
                  faltan_link=res.loc[mascara].groupby("tipo_zona")["brecha_link"].apply(lambda s: s.clip(lower=0).sum() / 7))
          .sort_values("residentes", ascending=False))
perfil.round(2)"""),
    code("""zonas = list(perfil.index)
colores = {z: e.color_zona(z) for z in zonas}
g = gpd.GeoDataFrame(res[res["tipo_zona"].notna()], geometry=[f.poligono_hex(h) for h in res.loc[res["tipo_zona"].notna(), "h3"]], crs="EPSG:4326")
fig, ax = plt.subplots(figsize=(10, 9))
for z in zonas:
    g[g["tipo_zona"] == z].plot(ax=ax, color=colores[z], linewidth=0.3, edgecolor=e.FONDO, label=z)
barrios.boundary.plot(ax=ax, color=e.TINTA, linewidth=0.4)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=colores[z], label=z) for z in zonas], loc="lower left")
ax.set_axis_off(); ax.set_title("Tipos de zona")
fig.tight_layout(); fig.savefig(FIG / "mapa_tipos_zona.png"); plt.show()"""),
    md(texto("clusters", "")),
    md("""## 6. Dónde sumar cajeros: cobertura máxima

¿Qué porcentaje de los vecinos tiene un cajero de cada red a menos de 500 m (unas 5 cuadras)? Si está disponible la red
peatonal, la distancia se mide **caminando por las calles** (algoritmo de Dijkstra sobre OpenStreetMap); si no, en línea recta."""),
    code("""cob = m.resumen_cobertura(res, f.RADIO_COBERTURA_M, sufijo=SUF)
if HAY_RED:
    recta = m.resumen_cobertura(res, f.RADIO_COBERTURA_M).set_index("red")["pct_residentes"]
    cob["pct_en_linea_recta"] = cob["red"].map(recta)
    ok = np.isfinite(res["dist_link_m_red"]) & (res["dist_link_m"] > 50)
    print(f"Caminar es, en la mediana, {e.num_es(np.median(res.loc[ok, 'dist_link_m_red'] / res.loc[ok, 'dist_link_m']), 2)} veces la distancia en línea recta")
cob.style.format({"residentes_cubiertos": "{:,.0f}", "pct_residentes": "{:.1%}", "pct_en_linea_recta": "{:.1%}"})"""),
    md("""Planteamos el **problema de cobertura máxima** (MCLP, Church y ReVelle, 1974): elegir N ubicaciones que maximicen las
personas que pasan a tener un Link a menos de 500 m. Solo se consideran sitios con comercio o servicios, porque un cajero
necesita un local. Lo resolvemos de dos formas:

- **Algoritmo goloso:** agrega en cada paso el sitio con mayor ganancia marginal. Como la cobertura es submodular, garantiza
  al menos el 63 % (1 − 1/e) del óptimo (Nemhauser, Wolsey y Fisher, 1978). Sirve para la curva de "cuántos cajeros".
- **Óptimo exacto:** el mismo problema como **programa lineal entero**, resuelto con HiGHS (el solver que trae SciPy)."""),
    code("""conj = rp.conjuntos_red(res, red, f.RADIO_COBERTURA_M) if HAY_RED else m.conjuntos_recta(res, f.RADIO_COBERTURA_M)
w = m.demanda(res)
recs = m.resolver_goloso(res, 50, conj, w)
recs["tipo_zona"] = recs["h3"].map(res.set_index("h3")["tipo_zona"])
recs["brecha_link"] = recs["h3"].map(res.set_index("h3")["brecha_link"])
total = res["poblacion"].sum()
base = cob.set_index("red").loc["Link", "residentes_cubiertos"]
banelco = cob.set_index("red").loc["Banelco", "pct_residentes"] * 100
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(recs["orden"], (base + recs["personas_acumuladas"]) / total * 100, color=e.LINK, linewidth=2, marker="o", markersize=4)
ax.axhline(base / total * 100, color=e.EJE, linewidth=1)
ax.axhline(banelco, color=e.BANELCO, linewidth=1.2)
ax.text(50, base / total * 100, " Link hoy", va="bottom", ha="right", color=e.TINTA_2, fontsize=9)
ax.text(50, banelco, " Banelco hoy", va="bottom", ha="right", color=e.BANELCO, fontsize=9)
pct = (base + recs["personas_acumuladas"]) / total * 100
for n_ in (10, 20):
    v = pct.iloc[n_ - 1]
    ax.annotate(f"{n_} cajeros: {e.num_es(v, 1)} %", (n_, v), xytext=(8, -14), textcoords="offset points", fontsize=9, color=e.TINTA)
cruce = recs.loc[pct >= banelco, "orden"]
if len(cruce):
    n_c = int(cruce.iloc[0])
    ax.scatter([n_c], [pct.iloc[n_c - 1]], s=60, color=e.BANELCO, zorder=3)
    ax.annotate(f"con {n_c} cajeros iguala a Banelco", (n_c, pct.iloc[n_c - 1]), xytext=(-10, 12), textcoords="offset points",
                ha="right", fontsize=9, color=e.BANELCO)
from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.set(title="Vecinos con un Link a menos de 500 m " + ("caminando " if HAY_RED else "") + "según cuántos cajeros se suman",
       xlabel="cajeros nuevos", ylabel="% de vecinos")
e.ejes_es(ax, y=0)
fig.tight_layout(); fig.savefig(FIG / "cobertura_curva.png"); plt.show()"""),
    code("""comp = m.comparar_goloso_exacto(res, conj, w)
comp[["cajeros", "goloso", "exacto", "diferencia_pct", "sitios_en_comun", "seg_exacto"]].round(2)"""),
    code("""exacto = m.resolver_exacto(res, 20, conj, w)
exacto["tipo_zona"] = exacto["h3"].map(res.set_index("h3")["tipo_zona"])
print(f"Óptimo exacto con 20 cajeros: +{e.num_es(exacto.attrs['objetivo'])} vecinos · Link pasa de "
      f"{e.num_es(100 * base / total, 1)} % a {e.num_es(100 * (base + exacto.attrs['objetivo']) / total, 1)} %")
exacto[["orden", "barrio", "comuna", "tipo_zona", "personas_nuevas", "personas_acumuladas"]].round(0)"""),
    code("""col = "dist_link_m" + SUF
fig, ax = plt.subplots(figsize=(10, 9))
barrios.boundary.plot(ax=ax, color=e.TINTA_SUAVE, linewidth=0.4)
sin = res[res[col] > f.RADIO_COBERTURA_M]
gpd.GeoDataFrame(sin, geometry=[f.poligono_hex(h) for h in sin["h3"]], crs="EPSG:4326").plot(
    ax=ax, column="poblacion", cmap=e.cmap_secuencial(e.SECUENCIAL_NARANJA), linewidth=0, alpha=0.8, vmax=res["poblacion"].quantile(0.98))
lk = cajeros[cajeros["red"] == "LINK"]
ax.scatter(lk["lon"], lk["lat"], s=6, color=e.LINK, label="Cajero Link actual")
ax.scatter(exacto["lon"], exacto["lat"], s=140, facecolor="none", edgecolor=e.SUGERIDO, linewidth=2.2, label="Ubicación sugerida (óptimo con 20)")
for _, r in exacto.iterrows():
    ax.text(r["lon"], r["lat"], str(r["orden"]), fontsize=7, ha="center", va="center", color=e.TINTA, fontweight="bold")
ax.legend(loc="lower left"); ax.set_axis_off()
ax.set_title("Vecinos sin un Link cerca (naranja) y las 20 ubicaciones óptimas")
fig.tight_layout(); fig.savefig(FIG / "mapa_recomendaciones.png"); plt.show()"""),
    md(texto("optimizacion", "")),
    md("""## 7. Validación temporal con datos que el modelo no vio

El listado de cajeros es de ≈2017. OpenStreetMap tiene cajeros mapeados hasta 2026: los que están a más de 150 m de
cualquier cajero del listado son, muy probablemente, **cajeros instalados después**. Si la brecha que calculamos con datos
de 2017 tiene sentido, esos cajeros nuevos deberían aparecer más en zonas con brecha positiva.

Medimos el **AUC**: la probabilidad de que una zona con cajero nuevo tenga más brecha que una zona sin cajero nuevo
(0,5 = azar)."""),
    code("""osm = pd.read_csv(RAIZ / "data/base/osm-caba-puntos.csv")
nuevos = m.cajeros_nuevos_osm(osm, cajeros)
modelables = res.loc[mascara]
cerca_de_nuevo = modelables["h3"].isin({v for h in f.hex_de(nuevos["lat"], nuevos["lon"]) for v in f.h3.grid_disk(h, 1)}).astype(int)
validacion = pd.DataFrame([
    {"puntaje": "Brecha total del modelo (datos 2017)", "AUC": m.auc(modelables["brecha_total"].values, cerca_de_nuevo.values)},
    {"puntaje": "Demanda esperada del modelo", "AUC": m.auc(modelables["esperado_total_k1"].values, cerca_de_nuevo.values)},
    {"puntaje": "Solo cantidad de residentes", "AUC": m.auc(modelables["poblacion_k1"].values, cerca_de_nuevo.values)},
])
print(f"Cajeros en OSM 2026: {(osm['categoria'] == 'cajero').sum()} · nuevos (a > 150 m del listado 2017): {len(nuevos)} · "
      f"hexágonos en su área: {cerca_de_nuevo.sum()}")
display(validacion.round(3))
perc = m.percentil_cajeros_nuevos(res, mascara, nuevos)
print(f"Los {perc['n']} cajeros nuevos cayeron, en promedio, en el percentil {e.num_es(100 * perc['percentil_promedio'])} "
      f"de la brecha 2017 · p = {e.num_es(perc['p_valor'], 3)} (Monte Carlo, 20.000 sorteos de zonas al azar)")"""),
    md(texto("validacion", "")),
    md("## 8. Conclusión\n\n" + texto("conclusion", "")),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, SALIDA)
print("ok", SALIDA)
