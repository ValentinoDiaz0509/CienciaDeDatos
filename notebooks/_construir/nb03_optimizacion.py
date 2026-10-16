"""Genera notebooks/03_optimizacion.ipynb (correr: python notebooks/_construir/nb03_optimizacion.py)."""
import sys
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SALIDA = Path(__file__).resolve().parents[1] / "03_optimizacion.ipynb"
TEXTOS = Path(__file__).resolve().parent / "textos_opt.py"
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
T = {"RAIZ": RAIZ}
if TEXTOS.exists():
    exec(TEXTOS.read_text(encoding="utf-8"), T)


def texto(clave, defecto=""):
    return T.get(clave, defecto)


cells = [
    md("""# 03 · Segmentación y optimización: dónde conviene sumar cajeros

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ValentinoDiaz0509/CienciaDeDatos/blob/main/notebooks/03_optimizacion.ipynb)

**TPO Ciencia de Datos · ¿Dónde faltan cajeros Link en CABA?**

Es el nivel **prescriptivo** (clase 2): ¿qué conviene hacer? Usamos dos técnicas:

| Técnica | Tipo | Pregunta |
|---|---|---|
| K-Means | no supervisada | ¿Qué tipos de zona hay y dónde se concentra la brecha? |
| Cobertura máxima (MCLP) | optimización | Si se suman N cajeros, ¿dónde ubicarlos para llegar a más gente? |

Partimos de lo que calculó el notebook 02 (la brecha de cada zona) y del pipeline (`python -m src.pipeline`)."""),
    code("""import json, sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from src import estilo as e, features as f, modelos as m, red as rp

e.mpl_estilo()
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
FIG = RAIZ / "docs/figuras"; FIG.mkdir(parents=True, exist_ok=True)
PROC = RAIZ / "data/processed"

res = pd.read_parquet(PROC / "resultados_hex.parquet")   # variables, brecha y distancias por hexágono
cajeros = f.cargar_cajeros()
barrios = f.cargar_barrios()
mascara = res["esperado_total_k1"].notna()
HAY_RED = rp.hay_red()
red = rp.RedPeatonal.desde_archivos() if HAY_RED else None
SUF = "_red" if HAY_RED else ""
print("Distancias:", "caminando por las calles" if HAY_RED else "en línea recta")"""),
    md("""## 1. Tipos de zona (K-Means, no supervisado)

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
    md("""## 2. Cobertura: ¿qué parte de los vecinos tiene un cajero cerca?

¿Qué porcentaje de los vecinos tiene un cajero de cada red a menos de 500 m (unas 5 cuadras)? Si está disponible la red
peatonal, la distancia se mide **caminando por las calles** (algoritmo de Dijkstra sobre OpenStreetMap); si no, en línea recta."""),
    code("""cob = m.resumen_cobertura(res, f.RADIO_COBERTURA_M, sufijo=SUF)
if HAY_RED:
    recta = m.resumen_cobertura(res, f.RADIO_COBERTURA_M).set_index("red")["pct_residentes"]
    cob["pct_en_linea_recta"] = cob["red"].map(recta)
    ok = np.isfinite(res["dist_link_m_red"]) & (res["dist_link_m"] > 50)
    print(f"Caminar es, en la mediana, {e.num_es(np.median(res.loc[ok, 'dist_link_m_red'] / res.loc[ok, 'dist_link_m']), 2)} veces la distancia en línea recta")
cob.style.format({"residentes_cubiertos": "{:,.0f}", "pct_residentes": "{:.1%}", "pct_en_linea_recta": "{:.1%}"})"""),
    md("""**Un ejemplo de por qué la línea recta engaña.** En esta zona de San Cristóbal (elegida a mano porque la cuadrícula es
regular y el efecto se ve claro), el Link más cercano está a unos 400 m en línea recta, así que la línea recta la cuenta
como cubierta. Caminando por las calles son más de 600 m."""),
    code("""if HAY_RED:
    from src import figura_caminando as fc
    from IPython.display import Image
    print(fc.dibujar("89c2e311183ffff"))
    display(Image(str(FIG / "ejemplo_caminando.png"), width=560))"""),
    md("""## 3. Dónde sumar cajeros: cobertura máxima

Planteamos el **problema de cobertura máxima** (MCLP, Church y ReVelle, 1974): elegir N ubicaciones que maximicen las
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
exactos = pd.read_csv(RAIZ / "data/processed/goloso_vs_exacto.csv").set_index("cajeros")["exacto"]
for n_ in (10, 20):  # la curva es la del goloso; los valores anotados, los del óptimo exacto
    v = (base + exactos.loc[n_]) / total * 100
    ax.annotate(f"{n_} cajeros (óptimo): {e.num_es(v, 1)} %", (n_, pct.iloc[n_ - 1]), xytext=(8, -14), textcoords="offset points",
                fontsize=9, color=e.TINTA)
# cuántos cajeros hacen falta para igualar a Banelco, con el óptimo exacto (lo calcula el pipeline)
n_c = json.loads((RAIZ / "data/processed/resumen.json").read_text(encoding="utf-8")).get("cajeros_para_igualar_banelco")
if n_c:
    ax.scatter([n_c], [banelco], s=60, color=e.BANELCO, zorder=3)
    ax.annotate(f"óptimo exacto: con {n_c} cajeros iguala a Banelco", (n_c, banelco), xytext=(-10, 12),
                textcoords="offset points", ha="right", fontsize=9, color=e.BANELCO)
from matplotlib.ticker import MultipleLocator
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.set(title="Vecinos con un Link a menos de 500 m " + ("caminando " if HAY_RED else "") + "según cuántos cajeros se suman",
       xlabel="cajeros nuevos (curva: algoritmo goloso)", ylabel="% de vecinos")
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
    md("## 4. Conclusión\n\n" + texto("conclusion", "")),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, SALIDA)
print("ok", SALIDA)
