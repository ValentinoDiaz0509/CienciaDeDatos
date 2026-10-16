"""
App del TPO: ¿Dónde faltan cajeros Link en CABA?

Correr desde la raíz del repo:
    streamlit run app/app.py

Funciona sin internet: lee los resultados precalculados de data/processed/. El mapa de calles
de fondo es opcional (necesita conexión); sin él se dibujan los límites de los barrios.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src import estilo as e  # noqa: E402
from src import features as f  # noqa: E402
from src import graficos_ml as gm  # noqa: E402
from src import ml  # noqa: E402
from src import modelos as m  # noqa: E402
from src import red as rp  # noqa: E402

PROC = RAIZ / "data" / "processed"
BASE = RAIZ / "data" / "base"

# Referencia pública para traducir terminales en operaciones: BCRA, Informe Mensual de Pagos
# Minoristas de agosto 2025 (datos de julio 2025): 57,2 millones de extracciones en 17.643 ATM.
BCRA_EXTRACCIONES_POR_ATM_MES = 3242
BCRA_MONTO_PROMEDIO_EXTRACCION = 75_100
BCRA_FUENTE = "BCRA, Informe Mensual de Pagos Minoristas, agosto 2025 (datos de julio 2025)"

st.set_page_config(page_title="Cajeros Link · ¿Dónde faltan?", page_icon="🏧", layout="wide")


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
@st.cache_data
def cargar():
    res = pd.read_parquet(PROC / "resultados_hex.parquet")
    resumen = json.loads((PROC / "resumen.json").read_text(encoding="utf-8"))
    datos = {
        "res": res, "resumen": resumen, "cajeros": f.cargar_cajeros(BASE),
        "estaciones": pd.read_csv(PROC / "estaciones_subte.csv"),
        "horario": pd.read_csv(PROC / "subte_horario.csv"),
        "perfil": pd.read_csv(PROC / "clusters_perfil.csv"),
        "metricas": pd.read_csv(PROC / "metricas_modelos.csv"),
        "metricas_clasif": pd.read_csv(PROC / "metricas_clasificacion.csv"),
        "oob": pd.read_csv(PROC / "oob_vs_validacion.csv"),
        "importancia": pd.read_csv(PROC / "importancia_variables.csv"),
        "gini": pd.read_csv(PROC / "importancia_gini.csv"),
        "barrios_pred": pd.read_csv(PROC / "barrios_prediccion.csv", index_col="barrio"),
        "aportes": pd.read_parquet(PROC / "aportes_hex.parquet").set_index("h3"),
        "comparacion": pd.read_csv(PROC / "goloso_vs_exacto.csv"),
        "cobertura": pd.read_csv(PROC / "cobertura_redes.csv"),
        "barrios": json.loads((BASE / "barrios.geojson").read_text(encoding="utf-8")),
    }
    if (PROC / "pronostico_resumen.json").exists():   # pronóstico del subte (notebook 05), si se corrió
        datos["pronostico"] = {
            "resumen": json.loads((PROC / "pronostico_resumen.json").read_text(encoding="utf-8")),
            "diario": pd.read_csv(PROC / "pronostico_diario.csv", parse_dates=["fecha"]),
            "mensual": pd.read_csv(PROC / "pronostico_mensual.csv"),
            "validacion": pd.read_csv(PROC / "pronostico_validacion.csv"),
            "estaciones": pd.read_csv(PROC / "pronostico_estaciones.csv"),
        }
    return datos


@st.cache_data
def cargar_conjuntos():
    z = np.load(PROC / "conjuntos_cobertura.npz")
    salida = {}
    for clave in {k.rsplit("_", 1)[0] for k in z.files}:
        metodo, radio = clave.split("_")
        largos = z[f"{clave}_largos"]
        vecinos = np.split(z[f"{clave}_vecinos"], np.cumsum(largos)[:-1]) if len(largos) else []
        salida[(metodo, int(radio))] = {"candidatos": z[f"{clave}_candidatos"], "vecinos": vecinos}
    return salida


@st.cache_resource
def cargar_red():
    return rp.RedPeatonal.desde_archivos(BASE) if rp.hay_red(BASE) else None


D = cargar()
res, resumen, cajeros = D["res"], D["resumen"], D["cajeros"]
CONJUNTOS = cargar_conjuntos()
HAY_RED = "dist_link_m_red" in res.columns
# Valores de las variables del modelo en cada zona (para explicar sus predicciones) y escala común de la brecha
X_VALORES = res.set_index("h3")[[c for c in D["aportes"].columns if not c.startswith("_")]]
LIM_BRECHA = float(np.nanquantile(np.abs(res["brecha_link"]), 0.95))


def fmt(n: float, dec: int = 0) -> str:
    return f"{n:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, dec: int = 0) -> str:
    return fmt(100 * x, dec) + " %"


# --------------------------------------------------------------------------
# Mapas
# --------------------------------------------------------------------------
VISTA = pdk.ViewState(latitude=-34.615, longitude=-58.445, zoom=11.2, pitch=40, bearing=0)
TOOLTIP = "{tip}"


def capa_barrios():
    return pdk.Layer("GeoJsonLayer", D["barrios"], stroked=True, filled=False, get_line_color=[82, 81, 78, 160],
                     line_width_min_pixels=1, pickable=False)


def capa_hex(df, elevacion: str | None = None, escala: float = 25):
    return pdk.Layer("H3HexagonLayer", df, get_hexagon="h3", get_fill_color="color", pickable=True, stroked=False,
                     filled=True, extruded=elevacion is not None, get_elevation=elevacion or 0,
                     elevation_scale=escala, opacity=0.85)


def capa_cajeros(df, radio: int = 35):
    d = df.copy()
    d["color"] = d["red"].map({"LINK": e.hex_a_rgb(e.LINK, 230), "BANELCO": e.hex_a_rgb(e.BANELCO, 230)})
    d["color"] = d["color"].apply(lambda c: c if isinstance(c, list) else [137, 135, 129, 220])
    d["tip"] = ("<b>" + d.get("banco", pd.Series("", index=d.index)).fillna("").astype(str) + "</b><br/>Red "
                + d["red"].astype(str).str.title() + " · " + d["terminales"].astype(str) + " terminales")
    return pdk.Layer("ScatterplotLayer", d, get_position=["lon", "lat"], get_fill_color="color", get_radius=radio,
                     radius_min_pixels=2, stroked=True, get_line_color=[252, 252, 251], line_width_min_pixels=1,
                     pickable=True)


def capa_sugeridos(recs: pd.DataFrame, radio: int):
    r = recs.copy()
    r["color"] = [e.hex_a_rgb(e.SUGERIDO, 55)] * len(r)
    r["borde"] = [e.hex_a_rgb(e.SUGERIDO, 255)] * len(r)
    r["texto"] = r["orden"].astype(str)
    r["tip"] = ("<b>Sugerencia #" + r["texto"] + " · " + r["barrio"].astype(str) + "</b><br/>Suma "
                + r["personas_nuevas"].map(lambda v: fmt(v)) + " personas con un Link a distancia caminable")
    return [pdk.Layer("ScatterplotLayer", r, get_position=["lon", "lat"], get_radius=radio, get_fill_color="color",
                      get_line_color="borde", stroked=True, line_width_min_pixels=2, pickable=True),
            pdk.Layer("TextLayer", r, get_position=["lon", "lat"], get_text="texto", get_size=14,
                      get_color=[11, 11, 11], get_alignment_baseline="'center'")]


def mapa(capas: list, alto: int = 620, vista=VISTA):
    deck = pdk.Deck(layers=[capa_barrios(), *capas], initial_view_state=vista,
                    map_provider="carto" if con_base else None,
                    map_style=pdk.map_styles.CARTO_LIGHT if con_base else None,
                    tooltip={"html": TOOLTIP, "style": {"backgroundColor": "#0b0b0b", "color": "white", "fontSize": "12px"}})
    st.pydeck_chart(deck, height=alto)


def leyenda(items: list[tuple[str, str]]):
    html = " ".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;font-size:13px;color:#52514e">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};margin-right:6px;display:inline-block"></span>{t}</span>'
        for c, t in items)
    st.markdown(html, unsafe_allow_html=True)


def con_tips(df: pd.DataFrame, col_dist: str) -> pd.DataFrame:
    d = df.copy()
    v = lambda c, dec=0: d[c].fillna(0).map(lambda x: fmt(x, dec))
    brecha = d["brecha_link"].fillna(0).map(lambda x: ("+" if x > 0 else "") + fmt(x, 1))
    d["tip"] = ("<b>" + d["barrio"].fillna("").astype(str) + "</b> · " + d["tipo_zona"].fillna("—").astype(str)
                + "<br/>Residentes en el hexágono: " + v("poblacion")
                + "<br/>Terminales en el área (≈400 m): Link " + v("term_link_k1") + " · Banelco " + v("term_banelco_k1")
                + "<br/>Link esperado según la demanda: " + v("esperado_link_k1", 1)
                + "<br/><b>Brecha Link: " + brecha + "</b><br/>Cajero Link más cercano: "
                + d[col_dist].map(lambda x: fmt(x) + " m" if np.isfinite(x) else "—"))
    return d


def estacion_cercana(lat: float, lon: float) -> str:
    """'a 300 m de Juramento (D)' si hay una estación de subte a menos de 1 km; si no, vacío."""
    est = D["estaciones"]
    dist = np.hypot((est["lat"] - lat) * 111_000, (est["lon"] - lon) * 92_000)
    i = dist.idxmin()
    if dist[i] > 1_000:
        return ""
    return f"a {fmt(max(100, round(dist[i], -2)))} m de {est.loc[i, 'estacion']} ({est.loc[i, 'linea']})"


def vista_de(lat: pd.Series, lon: pd.Series, ancho_px: int = 480) -> pdk.ViewState:
    """Encuadre que muestra todos los puntos (por ejemplo, las zonas de un barrio). deck.gl usa teselas de 512 px."""
    extension = max(lon.max() - lon.min(), (lat.max() - lat.min()) * 1.21) + 0.006
    zoom = float(np.clip(np.log2(360 * ancho_px / (512 * 1.2 * extension)), 10.5, 15))
    return pdk.ViewState(latitude=float(lat.mean()), longitude=float(lon.mean()), zoom=zoom, pitch=0, bearing=0)


def grafico_explicacion(t: pd.DataFrame):
    """Aportes de cada variable a lo que le corresponde a una zona (valores SHAP, como factores multiplicativos)."""
    import altair as alt

    d = t.copy()
    d["etiqueta"] = [r["nombre"] if r["variable"] == "resto" else f"{r['nombre']}: {gm.valor_legible(r['variable'], r['valor'])}"
                     for _, r in d.iterrows()]
    d["efecto"] = np.select([d["variable"] == "resto", d["aporte_log"] >= 0], ["Resto", "Sube"], "Baja")
    d["factor_txt"] = "×" + d["factor"].map(lambda v: fmt(v, 2))
    lim = max(float(d["aporte_log"].abs().max()) * 1.4, float(np.log(1.6)))
    marcas = [float(np.log(v)) for v in ((0.25, 0.5, 1, 2, 4) if lim > np.log(3) else (0.5, 0.75, 1, 1.5, 2))
              if abs(np.log(v)) <= lim]
    x = alt.X("aporte_log:Q", scale=alt.Scale(domain=[-lim, lim]), title="multiplica lo que le corresponde (×1 = no cambia)",
              axis=alt.Axis(values=marcas, labelExpr="'×' + replace(format(exp(datum.value), '.2~f'), '.', ',')"))
    y = alt.Y("etiqueta:N", sort=list(d["etiqueta"]), title=None, axis=alt.Axis(labelLimit=340))
    barras = alt.Chart(d).mark_bar(cornerRadius=3, height=24).encode(
        x=x, y=y,
        color=alt.Color("efecto:N", legend=None,
                        scale=alt.Scale(domain=["Sube", "Baja", "Resto"], range=[e.DIVERGENTE[-2], e.DIVERGENTE[1], e.EJE])),
        tooltip=[alt.Tooltip("nombre:N", title="Variable"), alt.Tooltip("factor_txt:N", title="Multiplica por")])
    sube = alt.Chart(d).transform_filter("datum.aporte_log >= 0").mark_text(align="left", dx=5, fontSize=12, color=e.TINTA) \
        .encode(x=x, y=y, text="factor_txt:N")
    baja = alt.Chart(d).transform_filter("datum.aporte_log < 0").mark_text(align="right", dx=-5, fontSize=12, color=e.TINTA) \
        .encode(x=x, y=y, text="factor_txt:N")
    cero = alt.Chart(pd.DataFrame({"aporte_log": [0.0]})).mark_rule(color=e.TINTA_SUAVE).encode(x=x)
    return (barras + sube + baja + cero).properties(height=46 * len(d))


def grafico_pronostico(d: pd.DataFrame):
    """Pronóstico diario de entradas al subte con su intervalo del 80 %."""
    import altair as alt

    d = d.copy()
    d["dia"] = d["fecha"].dt.strftime("%d/%m/%Y") + " · " + d["tipo_dia"]
    for c in ("pronostico", "p10", "p90"):
        d[c + "_txt"] = d[c].map(lambda v: fmt(v))
    meses = "['ene','feb','mar','abr','may','jun','jul','ago','sep','oct','nov','dic']"
    x = alt.X("fecha:T", title=None,
              axis=alt.Axis(labelExpr=f"{meses}[month(datum.value)] + ' ' + timeFormat(datum.value, '%y')", tickCount="month"))
    banda = alt.Chart(d).mark_area(color=e.SECUENCIAL[0]).encode(
        x=x, y=alt.Y("p10:Q", title="entradas por día",
                     axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), regexp(',', 'g'), '.')")), y2="p90:Q")
    linea = alt.Chart(d).mark_line(color=e.LINK, strokeWidth=2).encode(
        x=x, y="pronostico:Q",
        tooltip=[alt.Tooltip("dia:N", title="Día"), alt.Tooltip("pronostico_txt:N", title="Pronóstico"),
                 alt.Tooltip("p10_txt:N", title="Mínimo probable"), alt.Tooltip("p90_txt:N", title="Máximo probable")])
    return (banda + linea).properties(height=300)


def grafico_tiene_corresponden(b: pd.DataFrame):
    """Por barrio: terminales Link que tiene contra las que le corresponden según su demanda."""
    import altair as alt

    d = b.reset_index()[["barrio", "tiene_link", "corresponden_link", "diferencia_link"]]
    d["texto"] = d["diferencia_link"].map(lambda v: ("faltan " if v >= 0 else "sobran ") + fmt(abs(v)))
    d["extremo"] = d[["tiene_link", "corresponden_link"]].max(axis=1)
    largo = d.melt(id_vars=["barrio"], value_vars=["tiene_link", "corresponden_link"], var_name="serie", value_name="terminales")
    nombres = {"tiene_link": "Terminales Link que tiene", "corresponden_link": "Le corresponden según su demanda"}
    largo["serie"] = largo["serie"].map(nombres)
    y = alt.Y("barrio:N", sort=list(d["barrio"]), title=None)
    regla = alt.Chart(d).mark_rule(color=e.EJE, strokeWidth=3).encode(
        y=y, x=alt.X("tiene_link:Q", title="terminales Link"), x2="corresponden_link:Q")
    puntos = alt.Chart(largo).mark_circle(size=120, opacity=1).encode(
        y=y, x="terminales:Q",
        color=alt.Color("serie:N", title=None, legend=alt.Legend(orient="top", labelLimit=400),
                        scale=alt.Scale(domain=list(nombres.values()), range=[e.TINTA_SUAVE, e.LINK])),
        tooltip=[alt.Tooltip("barrio:N", title="Barrio"), alt.Tooltip("serie:N", title=" "),
                 alt.Tooltip("terminales:Q", title="Terminales", format=".0f")])
    texto = alt.Chart(d).mark_text(align="left", dx=10, fontSize=12, color=e.TINTA).encode(y=y, x="extremo:Q", text="texto:N")
    return (regla + puntos + texto).properties(height=30 * len(d) + 40)


# --------------------------------------------------------------------------
# Navegación
# --------------------------------------------------------------------------
st.sidebar.markdown("### 🏧 Cajeros Link en CABA")
st.sidebar.caption("Dónde faltan y dónde conviene sumar terminales")
pagina = st.sidebar.radio("Sección", ["Panorama", "Predicción por zona", "Simulador de expansión", "Demanda por hora",
                                      "Tipos de zona", "Usá tus datos", "Cómo funciona"], label_visibility="collapsed")
st.sidebar.divider()
metodo = st.sidebar.radio("Distancia", ["Caminando por las calles", "En línea recta"] if HAY_RED else ["En línea recta"],
                          help="Caminando usa la red peatonal de OpenStreetMap (algoritmo de Dijkstra). "
                               "La línea recta ignora manzanas, vías y parques, así que sobreestima la cobertura.")
CAMINANDO = metodo.startswith("Caminando")
COL_DIST = "dist_link_m_red" if CAMINANDO else "dist_link_m"
COL_DIST_B = "dist_banelco_m_red" if CAMINANDO else "dist_banelco_m"
CLAVE_METODO = "red" if CAMINANDO else "recta"
con_base = st.sidebar.toggle("Mapa de calles de fondo", value=False, help="Necesita internet.")
st.sidebar.divider()
st.sidebar.caption("TPO Ciencia de Datos · UADE · 2C 2026\n\nDatos: BA Data (GCBA), INDEC Censo 2022, SBASE, "
                   "© colaboradores de OpenStreetMap, BCRA. Cajeros: relevamiento de ≈2017.")


def cobertura(radio: float, col: str) -> float:
    return res.loc[res[col] <= radio, "poblacion"].sum() / res["poblacion"].sum()


@st.cache_data
def optimo(clave_metodo: str, col_dist: str, radio: int, n: int) -> float:
    """Vecinos que suma el óptimo exacto con n cajeros nuevos."""
    conj = dict(CONJUNTOS[(clave_metodo, radio)])
    conj["cubierto"] = res[col_dist].to_numpy() <= radio
    return m.resolver_exacto(res, n, conj, m.demanda(res), tiempo_max=30).attrs["objetivo"]


# --------------------------------------------------------------------------
# 1. Panorama
# --------------------------------------------------------------------------
if pagina == "Panorama":
    st.title("¿Dónde faltan cajeros Link en CABA?")
    st.markdown(
        "Cruzamos dónde vive y circula la gente con dónde están los cajeros. Cada hexágono es una zona de unas "
        "3 × 3 manzanas. El color muestra la **brecha Link**: cuántas terminales Link le faltan (rojo) o le sobran "
        "(azul) a la zona, comparada con zonas de demanda parecida.")
    cl, cb = cobertura(500, COL_DIST), cobertura(500, COL_DIST_B)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vecinos con un Link a < 500 m", pct(cl), f"{fmt(100 * (cl - cb), 0)} puntos vs. Banelco")
    c2.metric("Vecinos con un Banelco a < 500 m", pct(cb))
    c3.metric("Vecinos sin un Link cerca", fmt(res["poblacion"].sum() * (1 - cl)))
    c4.metric("Link con 20 cajeros nuevos", pct(cl + optimo(CLAVE_METODO, COL_DIST, 500, 20) / res["poblacion"].sum(), 1),
              help="Óptimo exacto: las 20 ubicaciones que suman más vecinos (ver Simulador).")
    st.caption(f"Distancia {'caminando por las calles' if CAMINANDO else 'en línea recta'}.")

    d = con_tips(res[res["esperado_link_k1"].notna()], COL_DIST)
    d["color"] = e.color_divergente(d["brecha_link"])
    d["altura"] = d["brecha_link"].clip(lower=0)
    capas = [capa_hex(d, elevacion="altura", escala=60)]
    if st.checkbox("Mostrar cajeros (relevamiento ≈2017)", value=False):
        capas.append(capa_cajeros(cajeros))
    mapa(capas)
    leyenda([(e.DIVERGENTE[-1], "Faltan terminales Link (más alto = más brecha)"), (e.DIVERGENTE[3], "En equilibrio"),
             (e.DIVERGENTE[0], "Sobran respecto de la demanda"), (e.LINK, "Cajero Link"), (e.BANELCO, "Cajero Banelco")])

    st.subheader("Los 15 barrios donde a Link le faltan más terminales")
    bp = D["barrios_pred"].head(15)
    st.dataframe(pd.DataFrame({
        "Residentes": bp["residentes"], "Terminales Link que tiene": bp["tiene_link"],
        "Le corresponden según su demanda": bp["corresponden_link"], "Faltan": bp["diferencia_link"],
        "Terminales Banelco": bp["tiene_banelco"], "Link cada 10 mil vecinos": bp["tiene_link"] / bp["residentes"] * 1e4})
        .round({"Residentes": 0}).rename_axis("Barrio"),
        column_config={"Residentes": st.column_config.NumberColumn(format="%d"),
                       "Le corresponden según su demanda": st.column_config.NumberColumn(format="%.0f"),
                       "Faltan": st.column_config.NumberColumn(format="%.0f"),
                       "Link cada 10 mil vecinos": st.column_config.NumberColumn(format="%.1f")})
    st.caption(f"Le corresponden: predicción del modelo de machine learning ({resumen['modelo_elegido']}) con la "
               f"participación de Link en la ciudad ({pct(resumen['cuota_link'])}). El detalle, zona por zona, está en "
               "**Predicción por zona**.")


# --------------------------------------------------------------------------
# 2. Predicción por zona (machine learning)
# --------------------------------------------------------------------------
elif pagina == "Predicción por zona":
    st.title("¿Cuántos cajeros le corresponden a cada zona?")
    st.markdown(
        "Un modelo de **machine learning** aprendió, con toda la ciudad, cuántas terminales hay en las zonas según su "
        "demanda (residentes, comercios, oficinas, pasajeros de subte y perfil socioeconómico) y con eso estima cuántas le "
        "**corresponden** a cada zona. Cada zona la predice un modelo que **no vio su comuna** (validación cruzada "
        "espacial), así que no se explica con sus propios cajeros.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Variación que explica el modelo", pct(resumen["d2_espacial_promedio"]),
              help=f"D² de Poisson (el R² de los conteos) de {resumen['modelo_elegido']}, promedio de 5 grupos de comunas "
                   f"que el modelo no vio; varía ±{fmt(100 * resumen['d2_espacial_desvio'])} puntos entre grupos.")
    k2.metric("Error medio por zona", f"{fmt(resumen['mae_espacial'], 1)} terminales",
              help="Error absoluto medio (MAE) en comunas no vistas, sobre el área de 400 m de cada zona.")
    k3.metric("Acierto del clasificador (AUC)", fmt(resumen["auc_clasificador"], 2),
              help=f"{resumen['clasificador_elegido']}: probabilidad de que una zona con cajero reciba más puntaje que una "
                   "sin cajero (0,5 = azar; 1 = perfecto). Medido en comunas no vistas.")
    k4.metric("Oportunidades (zonas sin cajero)", fmt(resumen["zonas_sin_cajero_que_deberian"]),
              help=f"Zonas sin ningún cajero que, según el clasificador, deberían tener uno. Viven {fmt(resumen['residentes_zonas_sin_cajero_que_deberian'])} personas en ellas.")

    vista_ml = st.radio("Vista", ["Un barrio, zona por zona", "Ranking de barrios", "Oportunidades (clasificación)"],
                        horizontal=True, label_visibility="collapsed")
    bp = D["barrios_pred"]

    if vista_ml == "Un barrio, zona por zona":
        barrio = st.selectbox("Barrio", list(bp.index), help="Ordenados de mayor a menor faltante de terminales Link.")
        fb = bp.loc[barrio]
        dif = fb["diferencia_link"]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Terminales Link que le corresponden", fmt(fb["corresponden_link"]))
        b2.metric("Terminales Link que tiene", fmt(fb["tiene_link"]))
        b3.metric("Faltan" if dif >= 0 else "Sobran", fmt(abs(dif)))
        b4.metric("Todas las redes: le corresponden / tiene", f"{fmt(fb['corresponden_total'])} / {fmt(fb['tiene_total'])}")
        st.caption(f"{barrio}: {fb['resumen']}.")

        rb = (res[(res["barrio"] == barrio) & res["esperado_link_k1"].notna()]
              .sort_values("brecha_link", ascending=False).copy())
        rb["zona"] = np.arange(1, len(rb) + 1)
        rb["cerca"] = [estacion_cercana(a, o) for a, o in zip(rb["lat"], rb["lon"])]
        etiquetas = {f"Zona {z.zona} · {'faltan' if z.brecha_link >= 0 else 'sobran'} {fmt(abs(z.brecha_link), 1)} Link"
                     + (f" · {z.cerca}" if z.cerca else ""): z.h3 for z in rb.itertuples()}
        izq, der = st.columns([1, 1.15])
        with der:
            elegida = st.selectbox("Zona (de mayor a menor faltante)", list(etiquetas))
            h = etiquetas[elegida]
            z = rb.set_index("h3").loc[h]
            subte = (f"entran {fmt(z['pax_subte_habil_k1'])} pasajeros de subte por día hábil" if z["pax_subte_habil_k1"] > 0
                     else "no hay estaciones de subte")
            st.markdown(f"**En el área de 400 m alrededor de la zona** le corresponden **{fmt(z['esperado_total_k1'], 1)} "
                        f"terminales** ({fmt(z['esperado_link_k1'], 1)} de Link) y tiene **{fmt(z['term_total_k1'])}** "
                        f"({fmt(z['term_link_k1'])} de Link). Viven {fmt(z['poblacion_k1'])} personas, hay "
                        f"{fmt(z['actividad_osm_k1'])} comercios y {subte}.")
            if h in D["aportes"].index:
                st.altair_chart(grafico_explicacion(ml.explicar_zona(D["aportes"], X_VALORES, h, n=5)))
                st.caption("**Por qué:** cuánto multiplica cada variable lo que le corresponde a la zona, frente a una zona "
                           "típica de la ciudad (valores SHAP del modelo). ×2 duplica; ×0,5 reduce a la mitad.")
            if z["term_total"] == 0 and bool(z["deberia_tener_cajero"]):
                st.success(f"**Oportunidad.** La zona no tiene ningún cajero y el clasificador le da {pct(z['prob_cajero'])} "
                           "de probabilidad de tenerlo: las zonas con una demanda así suelen tener uno.")
        with izq:
            d = con_tips(rb, COL_DIST)
            d["tip"] = "<b>Zona " + d["zona"].astype(str) + "</b> · " + d["tip"]
            d["color"] = e.color_divergente(d["brecha_link"], limite=LIM_BRECHA)
            marco = pdk.Layer("H3HexagonLayer", d[d["h3"] == h][["h3"]], get_hexagon="h3", filled=False, stroked=True,
                              extruded=False, get_line_color=[11, 11, 11, 255], line_width_min_pixels=3, pickable=False)
            margen = 0.01
            caj = cajeros[cajeros["lat"].between(rb["lat"].min() - margen, rb["lat"].max() + margen)
                          & cajeros["lon"].between(rb["lon"].min() - margen, rb["lon"].max() + margen)]
            mapa([capa_hex(d), marco, capa_cajeros(caj, radio=18)], alto=560, vista=vista_de(rb["lat"], rb["lon"]))
            leyenda([(e.DIVERGENTE[-1], "Faltan terminales Link"), (e.DIVERGENTE[0], "Sobran"),
                     (e.LINK, "Cajero Link"), (e.BANELCO, "Cajero Banelco")])

    elif vista_ml == "Ranking de barrios":
        st.markdown("Para cada barrio, las terminales Link que **tiene** contra las que le **corresponden** según su demanda: "
                    "lo que el modelo predice para sus zonas, con la participación de Link en la ciudad "
                    f"({pct(resumen['cuota_link'])}).")
        a, b = st.columns([2, 1])
        cuantos = a.slider("Barrios a mostrar", 5, len(bp), 15)
        orden = b.radio("Orden", ["Donde más faltan", "Donde más sobran"], horizontal=True)
        sel = bp.head(cuantos) if orden == "Donde más faltan" else bp.tail(cuantos).iloc[::-1]
        st.altair_chart(grafico_tiene_corresponden(sel))
        if orden == "Donde más sobran":
            st.caption("En el microcentro el modelo marca terminales de más: a esas zonas llega gente de toda la ciudad y del "
                       "conurbano en colectivo y tren, una demanda que los datos públicos captan solo en parte.")
        tabla_b = pd.DataFrame({
            "Link que tiene": bp["tiene_link"], "Link que le corresponden": bp["corresponden_link"],
            "Faltan (+) / sobran (−) Link": bp["diferencia_link"], "Todas las redes: tiene": bp["tiene_total"],
            "Todas las redes: le corresponden": bp["corresponden_total"], "Residentes": bp["residentes"],
            "Comercios": bp["comercios"], "Pasajeros de subte por día": bp["pasajeros_subte"], "Comuna": bp["comuna"]}).round(0)
        tabla_b.index.name = "Barrio"
        st.dataframe(tabla_b, column_config={c: st.column_config.NumberColumn(format="%.0f") for c in tabla_b.columns})
        st.download_button("Descargar la tabla (CSV)", tabla_b.to_csv().encode("utf-8"), "barrios_prediccion.csv", "text/csv")

    else:
        st.markdown(f"Un segundo modelo, de **clasificación** ({resumen['clasificador_elegido']}), estima la probabilidad de "
                    "que una zona tenga al menos un cajero de cualquier red según su demanda. Si las zonas parecidas lo "
                    "tienen y esta no, es una **oportunidad**: hay "
                    f"**{fmt(resumen['zonas_sin_cajero_que_deberian'])} zonas** así, donde viven "
                    f"**{fmt(resumen['residentes_zonas_sin_cajero_que_deberian'])} personas**.")
        d = con_tips(res[res["prob_cajero"].notna()], COL_DIST)
        oportunidad = d["deberia_tener_cajero"].astype(bool) & (d["term_total"] == 0)
        d["tip"] = (d["tip"] + "<br/>Probabilidad de tener cajero (clasificador): " + d["prob_cajero"].map(lambda v: pct(v))
                    + np.where(oportunidad, "<br/><b>Oportunidad</b>", ""))
        d["color"] = [e.hex_a_rgb(e.SUGERIDO, 235) if o else e.hex_a_rgb(e.TINTA_SUAVE, 150) if t > 0 else e.hex_a_rgb(e.GRILLA, 140)
                      for o, t in zip(oportunidad, d["term_total"])]
        mapa([capa_hex(d)], vista=pdk.ViewState(latitude=-34.615, longitude=-58.445, zoom=11.3, pitch=0, bearing=0))
        leyenda([(e.SUGERIDO, "Oportunidad: debería tener cajero y no tiene"), (e.TINTA_SUAVE, "Tiene al menos un cajero"),
                 (e.GRILLA, "Sin cajero y sin demanda suficiente")])
        op = d[oportunidad]
        st.subheader("Barrios con más oportunidades")
        st.dataframe(op.groupby("barrio").agg(zonas=("h3", "count"), residentes=("poblacion", "sum")).round(0)
                     .sort_values(["zonas", "residentes"], ascending=False).head(15).rename_axis("Barrio")
                     .rename(columns={"zonas": "Zonas sin cajero que deberían tener", "residentes": "Residentes en esas zonas"}),
                     column_config={"Residentes en esas zonas": st.column_config.NumberColumn(format="%d")})
        st.caption(f"El clasificador marca tantas zonas como las que hoy tienen cajero (umbral de probabilidad "
                   f"{fmt(resumen['umbral_clasificador'], 2)}); con ese umbral acierta el {pct(resumen['precision_clasificador'])} "
                   "de las zonas que marca. Que una zona sea oportunidad no dice qué red la ocupa: la estrategia de cobertura "
                   "del Simulador la complementa.")


# --------------------------------------------------------------------------
# 3. Simulador
# --------------------------------------------------------------------------
elif pagina == "Simulador de expansión":
    st.title("Simulador: ¿dónde poner los próximos cajeros Link?")
    st.markdown("Elegí cuántos cajeros sumar. El modelo busca las ubicaciones que le acercan un Link a la mayor "
                "cantidad de personas que hoy no tienen ninguno a distancia caminable.")
    a, b, c, d_ = st.columns(4)
    n = a.slider("Cajeros nuevos", 1, 50, 20)
    radio = b.select_slider("Distancia máxima (m)", options=[300, 400, 500, 600, 700, 800], value=500)
    franja = c.selectbox("Pasajeros de subte a sumar", ["Ninguno (solo vecinos)", "Todo el día hábil",
                                                         "Hora pico mañana (6-10 h)", "Hora pico tarde (16-20 h)"])
    solucion = d_.selectbox("Método", ["Óptimo exacto", "Goloso", "K-Means"],
                            help="Exacto: programación lineal entera (HiGHS). Goloso: agrega de a un sitio. K-Means: agrupa "
                                 "a la gente sin Link y pone un cajero en el centro de cada grupo (clase 11), para comparar.")
    col_pax = {"Todo el día hábil": "pax_subte_habil", "Hora pico mañana (6-10 h)": "pax_subte_manana",
               "Hora pico tarde (16-20 h)": "pax_subte_tarde"}.get(franja, "pax_subte_habil")
    peso = 0.0 if franja.startswith("Ninguno") else 1.0

    conj = dict(CONJUNTOS[(CLAVE_METODO, radio)])
    conj["cubierto"] = res[COL_DIST].to_numpy() <= radio
    w = m.demanda(res, peso, col_pax)
    if solucion == "Óptimo exacto":
        recs = m.resolver_exacto(res, n, conj, w, tiempo_max=30)
    elif solucion == "Goloso":
        recs = m.resolver_goloso(res, n, conj, w)
    else:
        recs = m.resolver_kmeans(res, n, conj, w)
    pob = res["poblacion"].sum()
    antes = res.loc[conj["cubierto"], "poblacion"].sum()
    gan = recs["residentes_nuevos"].sum() if not recs.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Vecinos con Link cerca hoy", pct(antes / pob, 1))
    k2.metric(f"Con {n} cajeros nuevos", pct((antes + gan) / pob, 1), f"+{fmt(100 * gan / pob, 1)} puntos")
    k3.metric("Vecinos que ganan un Link cerca", fmt(gan))
    k4.metric("Extracciones por mes (estimadas)", fmt(len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES),
              help=f"{fmt(BCRA_EXTRACCIONES_POR_ATM_MES)} extracciones por terminal por mes, promedio nacional. {BCRA_FUENTE}.")
    if solucion == "K-Means":
        ex = m.resolver_exacto(res, n, conj, w, tiempo_max=30)
        gan_ex = ex["residentes_nuevos"].sum() if not ex.empty else 0
        st.info(f"K-Means, como se vio en la clase 11, agrupa a la gente que no tiene un Link cerca y pone un cajero en el "
                f"centro de cada grupo: minimiza la distancia a ese centro, no busca que la mayor cantidad de vecinos quede a "
                f"{radio} m, y no sabe dónde están los cajeros que ya existen. Con {n} cajeros suma {fmt(gan)} vecinos; el "
                f"óptimo exacto, {fmt(gan_ex)}" + (f" (K-Means llega al {fmt(100 * gan / gan_ex)} %)." if gan_ex else "."))

    d = res[res["poblacion"] > 0].copy()
    d["sin_link"] = np.where(d[COL_DIST] > radio, d["poblacion"], 0)
    d = con_tips(d, COL_DIST)
    d["color"] = e.color_secuencial(d["sin_link"], paradas=e.SECUENCIAL_NARANJA, alfa=170)
    capas = [capa_hex(d), capa_cajeros(cajeros[cajeros["red"] == "LINK"], radio=30)]
    if not recs.empty:
        capas += capa_sugeridos(recs, radio)
    mapa(capas, vista=pdk.ViewState(latitude=-34.615, longitude=-58.445, zoom=11.3, pitch=0, bearing=0))
    leyenda([(e.SECUENCIAL_NARANJA[4], "Más vecinos sin un Link cerca"), (e.LINK, "Cajero Link actual"),
             (e.SUGERIDO, "Ubicación sugerida (círculo = distancia máxima)")])

    if not recs.empty:
        st.subheader("Ubicaciones sugeridas, en orden de prioridad")
        t = recs[["orden", "barrio", "comuna", "personas_nuevas", "residentes_nuevos", "lat", "lon"]].copy()
        t.insert(3, "tipo de zona", recs["h3"].map(res.set_index("h3")["tipo_zona"]).values)
        st.dataframe(t.rename(columns={"orden": "#", "barrio": "Barrio", "comuna": "Comuna",
                                       "personas_nuevas": "Personas que suma", "residentes_nuevos": "Vecinos que suma"}),
                     column_config={"Personas que suma": st.column_config.NumberColumn(format="%d"),
                                    "Vecinos que suma": st.column_config.NumberColumn(format="%d"),
                                    "lat": st.column_config.NumberColumn(format="%.5f"),
                                    "lon": st.column_config.NumberColumn(format="%.5f")}, hide_index=True)
        st.download_button("Descargar ubicaciones (CSV)", t.to_csv(index=False).encode("utf-8"),
                           "ubicaciones_sugeridas.csv", "text/csv")

        st.subheader("¿Cuánto mueve?")
        ops = len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES
        st.markdown(
            f"Si cada terminal nueva alcanza el promedio nacional de **{fmt(BCRA_EXTRACCIONES_POR_ATM_MES)} extracciones "
            f"por mes**, los {len(recs)} cajeros sumarían unas **{fmt(ops)} operaciones mensuales** en la red Link, "
            f"por unos **${fmt(ops * BCRA_MONTO_PROMEDIO_EXTRACCION / 1e6)} millones** de efectivo por mes "
            f"(monto promedio de ${fmt(BCRA_MONTO_PROMEDIO_EXTRACCION)} por extracción).")
        ingreso = st.number_input("Ingreso de Red Link por operación (dato interno, en pesos)", min_value=0.0, value=0.0, step=10.0)
        if ingreso > 0:
            st.metric("Ingreso mensual estimado", f"${fmt(ops * ingreso)}")
        st.caption(f"Fuente: {BCRA_FUENTE}. Es un orden de magnitud: una terminal en una zona sin cajeros cerca puede "
                   "estar por encima o por debajo del promedio del país.")


# --------------------------------------------------------------------------
# 4. Demanda por hora
# --------------------------------------------------------------------------
elif pagina == "Demanda por hora":
    st.title("Demanda por hora del día")
    st.markdown("Entradas al subte por estación (SBASE, promedio 2025). Las columnas muestran cuánta gente entra "
                "a esa hora; el color dice si hay un **Link a distancia caminable** de la estación.")
    hor = D["horario"].merge(D["estaciones"][["linea", "estacion", "dist_link_m"] +
                                              (["dist_link_m_red"] if HAY_RED else [])], on=["linea", "estacion"], how="left")
    a, b = st.columns([3, 1])
    hora = a.slider("Hora", 5, 23, 8, format="%d h")
    tipo = b.radio("Día", ["Hábil", "Fin de semana"], horizontal=True)
    td = "habil" if tipo == "Hábil" else "fin_de_semana"
    h = hor[(hor["hora"] == hora) & (hor["tipo_dia"] == td)].copy()
    h["cerca"] = h[COL_DIST] <= 500
    h["color"] = h["cerca"].map({True: e.hex_a_rgb(e.LINK, 220), False: e.hex_a_rgb(e.DIVERGENTE[-2], 230)})
    h["tip"] = ("<b>" + h["estacion"] + " (línea " + h["linea"] + ")</b><br/>" + h["pasajeros"].map(lambda v: fmt(v))
                + f" entradas entre las {hora} y las {hora + 1} h<br/>" + h["perfil"].fillna("").astype(str)
                + "<br/>Link más cercano: " + h[COL_DIST].map(lambda v: fmt(v) + " m"))
    capa = pdk.Layer("ColumnLayer", h, get_position=["lon", "lat"], get_elevation="pasajeros", elevation_scale=0.6,
                     radius=90, get_fill_color="color", pickable=True, extruded=True)
    mapa([capa], vista=pdk.ViewState(latitude=-34.605, longitude=-58.43, zoom=11.6, pitch=45, bearing=0))
    leyenda([(e.LINK, "Estación con un Link a < 500 m"), (e.DIVERGENTE[-2], "Estación sin un Link cerca")])

    curva = hor.pivot_table(index="hora", columns="tipo_dia", values="pasajeros", aggfunc="sum").rename(
        columns={"habil": "Día hábil", "fin_de_semana": "Fin de semana"})
    st.subheader("Entradas por hora en toda la red")
    curva = curva[["Día hábil", "Fin de semana"]]
    st.line_chart(curva, color=[e.LINK, e.TINTA_SUAVE], x_label="hora", y_label="entradas por hora")
    est = D["estaciones"]
    c1, c2 = st.columns(2)
    for col, (perfil, grupo) in zip([c1, c2], est.groupby("perfil")):
        col.markdown(f"**{perfil}** — {len(grupo)} estaciones")
        col.caption(", ".join(grupo.sort_values("pax_habil", ascending=False)["estacion"].head(12)))
    st.caption("Los molinetes cuentan entradas: en las estaciones de origen la gente entra a la mañana (sale de su "
               "casa); en las de destino entra a la tarde (vuelve del trabajo). Por eso conviene mirar a qué hora "
               "pasa la gente por cada zona antes de elegir dónde y con qué horario instalar un cajero.")

    if "pronostico" in D:
        P = D["pronostico"]
        R = P["resumen"]
        st.subheader("Pronóstico: cuánta gente va a entrar al subte en los próximos meses")
        mape = R["mape_promedio"]
        st.markdown(
            f"Con la serie diaria de los molinetes ({pd.Timestamp(R['desde']):%m/%Y} a {pd.Timestamp(R['hasta']):%m/%Y}) "
            f"entrenamos **{R['modelo_elegido']}** con variables de calendario (día de la semana, mes, feriados y puentes). "
            f"En meses que el modelo no vio erra en promedio un **{fmt(mape[R['modelo_elegido']], 1)} %** por día, contra "
            f"{fmt(mape['Mismo día del año anterior'], 1)} % de la referencia ingenua (el mismo día del año anterior).")
        p1, p2, p3 = st.columns(3)
        p1.metric(f"Día hábil promedio en {R['primer_mes_nombre']}", fmt(R["pax_habil_primer_mes"]),
                  f"{'+' if R['variacion_primer_mes_pct'] >= 0 else ''}{fmt(R['variacion_primer_mes_pct'], 1)} % vs. un año antes",
                  delta_color="off")
        p2.metric("En estaciones sin Link cerca", fmt(R["pax_habil_estaciones_sin_link"]),
                  help=f"Entradas por día hábil en las {R['estaciones_sin_link']} estaciones sin un Link a distancia "
                       f"caminable, en {R['primer_mes_nombre']}.")
        p3.metric("Pronóstico hasta", f"{pd.Timestamp(R['pronostico_hasta']):%d/%m/%Y}")
        st.altair_chart(grafico_pronostico(P["diario"]))
        st.caption("La banda es un intervalo del 80 %: dos modelos XGBoost de cuantiles (10 y 90). Los días con datos "
                   "dudosos (paros, cortes y fechas mal cargadas en el archivo de 2025) no se usan para entrenar. Detalle "
                   "en el notebook 05.")
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
                 "noviembre", "diciembre"]
        m = P["mensual"]
        st.dataframe(pd.DataFrame({
            "Mes": m["mes"].map(lambda p_: f"{meses[int(p_[5:7]) - 1]} de {p_[:4]}"),
            "Día hábil promedio": m["pronostico"].map(fmt),
            "Rango probable (80 %)": m["p10"].map(fmt) + " a " + m["p90"].map(fmt),
            "Mismo mes, un año antes": m["anio_anterior"].map(lambda v: fmt(v) if pd.notna(v) else "sin dato"),
            "Variación": m["variacion_%"].map(lambda v: f"{'+' if v >= 0 else ''}{fmt(v, 1)} %" if pd.notna(v) else "sin dato"),
        }), hide_index=True)


# --------------------------------------------------------------------------
# 5. Tipos de zona
# --------------------------------------------------------------------------
elif pagina == "Tipos de zona":
    st.title("Tipos de zona")
    st.markdown(f"Agrupamos los hexágonos en **{resumen['k_clusters']} tipos de zona** con K-Means, según residentes, "
                "comercio, pasajeros de subte, distancia al centro y perfil socioeconómico. Sirve para entender "
                "**dónde** está la brecha: no es lo mismo faltar en el centro que en un barrio residencial.")
    perfil = D["perfil"].set_index("tipo_zona")
    zonas = list(perfil.index)
    colores = {z: e.color_zona(z) for z in zonas}  # la zona sin vecinos va en gris: es fondo
    elegidas = st.multiselect("Tipos de zona a mostrar", zonas, default=zonas)
    d = con_tips(res[res["tipo_zona"].isin(elegidas)], COL_DIST)
    d["color"] = d["tipo_zona"].map(lambda z: e.hex_a_rgb(colores[z], 190))
    mapa([capa_hex(d)])
    leyenda([(colores[z], z) for z in zonas])
    vista = pd.DataFrame({
        "Hexágonos": perfil["hexagonos"], "Residentes": perfil["residentes"], "Pasajeros de subte por día": perfil["pax_subte"],
        "Terminales cada 10 mil vecinos": perfil["terminales_cada_10mil"], "Link cada 10 mil vecinos": perfil["link_cada_10mil"],
        "Terminales Link faltantes": perfil["faltan_link"], "% hogares con NBI (mediana)": 100 * perfil["pct_nbi_k1"],
        "Distancia al centro, km (mediana)": perfil["dist_centro_km"]})
    st.dataframe(vista, column_config={c: st.column_config.NumberColumn(format="%.1f") for c in vista.columns}
                 | {c: st.column_config.NumberColumn(format="%d") for c in ("Hexágonos", "Residentes", "Pasajeros de subte por día")})


# --------------------------------------------------------------------------
# 6. Usá tus datos
# --------------------------------------------------------------------------
elif pagina == "Usá tus datos":
    st.title("Usá tu listado actual de cajeros")
    st.markdown("El análisis público usa el último listado abierto de cajeros (≈2017). Red Link tiene el listado al día: "
                "si lo cargás acá, se recalculan la cobertura y las ubicaciones sugeridas. **El archivo no sale de tu "
                "computadora**: la app corre en forma local.")
    st.markdown("El CSV necesita las columnas `lat`, `lon` (o `long`) y `red` (LINK / BANELCO). `terminales` es opcional.")
    archivo = st.file_uploader("Listado de cajeros (CSV)", type="csv")
    if archivo is None:
        with st.expander("¿Qué formato tiene que tener?"):
            st.dataframe(cajeros[["lat", "lon", "red", "terminales", "banco"]].head(5), hide_index=True)
            st.download_button("Descargar el listado público como plantilla",
                               cajeros[["lat", "lon", "red", "terminales", "banco"]].to_csv(index=False).encode("utf-8"),
                               "plantilla_cajeros.csv", "text/csv")
    else:
        nuevo = pd.read_csv(archivo)
        nuevo.columns = [c.lower().strip() for c in nuevo.columns]
        nuevo = nuevo.rename(columns={"long": "lon", "longitud": "lon", "latitud": "lat"})
        faltan = {"lat", "lon", "red"} - set(nuevo.columns)
        if faltan:
            st.error(f"Faltan columnas: {', '.join(sorted(faltan))}")
            st.stop()
        nuevo["red"] = nuevo["red"].astype(str).str.upper().str.strip()
        nuevo["terminales"] = nuevo.get("terminales", 1)
        radio = st.select_slider("Distancia máxima (m)", options=[300, 400, 500, 600, 700, 800], value=500)
        n = st.slider("Cajeros nuevos a ubicar", 1, 50, 20)
        red = cargar_red() if CAMINANDO else None

        def distancias(caj, red_):
            if red_ is not None:
                return red_.distancia_a_mas_cercano(res["lon"], res["lat"], caj["lon"], caj["lat"])
            return f.distancias_minimas(res, caj)

        filas = []
        dist_nuevo = None
        for nombre, caj in [("Listado público (≈2017)", cajeros), ("Tu listado", nuevo)]:
            dl = distancias(caj[caj["red"] == "LINK"], red)
            db = distancias(caj[caj["red"] == "BANELCO"], red)
            pobt = res["poblacion"].sum()
            filas.append({"Listado": nombre, "Terminales Link": caj.loc[caj["red"] == "LINK", "terminales"].sum(),
                          "Terminales Banelco": caj.loc[caj["red"] == "BANELCO", "terminales"].sum(),
                          "% vecinos con Link cerca": 100 * res.loc[dl <= radio, "poblacion"].sum() / pobt,
                          "% vecinos con Banelco cerca": 100 * res.loc[db <= radio, "poblacion"].sum() / pobt})
            dist_nuevo = dl
        st.dataframe(pd.DataFrame(filas), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                    ("% vecinos con Link cerca", "% vecinos con Banelco cerca")})
        conj = dict(CONJUNTOS[(CLAVE_METODO, radio)])
        conj["cubierto"] = dist_nuevo <= radio
        recs = m.resolver_exacto(res, n, conj, m.demanda(res), tiempo_max=30)
        st.subheader("Ubicaciones sugeridas con tu listado")
        capas = [capa_cajeros(nuevo, radio=30)] + (capa_sugeridos(recs, radio) if not recs.empty else [])
        mapa(capas)
        if not recs.empty:
            st.dataframe(recs[["orden", "barrio", "comuna", "residentes_nuevos", "lat", "lon"]], hide_index=True)


# --------------------------------------------------------------------------
# 7. Cómo funciona
# --------------------------------------------------------------------------
else:
    st.title("Cómo funciona")
    if HAY_RED and "red_km" in resumen:
        texto_distancias = (
            f"Caminando por la red peatonal de OpenStreetMap ({fmt(resumen['red_km'])} km, {fmt(resumen['red_nodos'])} esquinas): "
            "cada punto se proyecta sobre la cuadra más cercana y el camino más corto se calcula con Dijkstra. Caminar es, en la "
            f"mediana, {fmt(resumen['factor_desvio_red'], 2)} veces la distancia en línea recta; por eso la cobertura de Link "
            f"caminando ({pct(resumen['cobertura_link'])}) es menor que en línea recta ({pct(resumen['cobertura_link_recta'])})")
    else:
        texto_distancias = "En línea recta"
    met = D["metricas"]
    esp = met[met["validación"].str.startswith("espacial")].set_index("modelo")
    ale = met[met["validación"] == "aleatoria"].set_index("modelo")
    texto_tec = ""
    if (PROC / "tecnicas_resumen.json").exists():
        RT = json.loads((PROC / "tecnicas_resumen.json").read_text(encoding="utf-8"))
        texto_tec = (
            f"\n5. **Técnicas de la clase 11.** Con validación cruzada estratificada por grupos, el clasificador elegido no "
            f"cambia (AUC {fmt(RT.get('auc_estratificada_rf', 0), 2)}). PCA: cuatro componentes resumen el "
            f"{fmt(RT['pca_varianza_4'])} % de la información de las 14 variables y, con ellos, K-Means arma los mismos tipos de "
            f"zona (ARI {fmt(RT['ari_4_componentes'], 2)}). K-Means como ubicador de cajeros llega al "
            f"{fmt(RT['kmeans_pct_optimo_20'])} % de los vecinos que suma el óptimo exacto con 20 cajeros. DBSCAN encuentra "
            f"{RT['bolsones']} bolsones de vecinos sin un Link cerca, donde vive el {fmt(RT['pct_en_bolsones'])} % de esa gente "
            "(notebook 06).")
    texto_pron = ""
    if "pronostico" in D:
        RP = D["pronostico"]["resumen"]
        mp = RP["mape_promedio"]
        texto_pron = (f"\n   - *Pronóstico en el tiempo:* ¿cuánta gente va a entrar al subte cada día? **{RP['modelo_elegido']}** "
                      "con variables del calendario, validado hacia adelante (se entrena con el pasado y se mide en meses que no "
                      f"vio): erra un {fmt(mp[RP['modelo_elegido']], 1)} % por día, contra {fmt(mp['Mismo día del año anterior'], 1)} % "
                      "de repetir el mismo día del año anterior.")
    st.markdown(f"""
**Pregunta.** ¿En qué zonas de CABA hay más demanda que cajeros Link, y dónde conviene sumar terminales?

**Unidad de análisis.** {fmt(resumen['hexagonos'])} hexágonos H3 (resolución 9, ≈{fmt(resumen['area_media_hex_km2'], 2)} km²).
Un cajero sirve a quien está a pocas cuadras, así que las variables se miden sobre el hexágono y sus 6 vecinos (≈400 m).

**Datos, todos públicos.** Censo 2022 por radio censal (INDEC), cajeros automáticos (BA Data, relevamiento de ≈2017),
usos del suelo 2022-2024 (BA Data), molinetes del subte (SBASE: 2025 por hora y la serie diaria hasta agosto de 2026),
OpenStreetMap (comercios, estaciones de tren,
cajeros 2026 y red peatonal) y BCRA (extracciones por cajero).

**Cuatro niveles de análisis:**
1. **Descriptivo.** Dónde están los cajeros, la gente y el flujo del subte.
2. **Diagnóstico.** Qué hace que una zona tenga más cajeros: valores SHAP, importancia de Gini y coeficientes de la
   regresión de Poisson.
3. **Predictivo (machine learning).**
   - *Regresión:* ¿cuántas terminales le corresponden a cada zona? Comparamos {len(esp)} modelos (de una línea de base a
     árboles, Random Forest, XGBoost y un ensamble) con **validación cruzada espacial anidada**: se mide siempre en comunas
     que el modelo no vio, y los hiperparámetros se eligen con GridSearchCV adentro de cada partición. Ganó
     **{resumen['modelo_elegido']}**: D² de Poisson de {fmt(resumen['d2_espacial_promedio'], 2)} ± {fmt(resumen['d2_espacial_desvio'], 2)}
     y error medio de {fmt(resumen['mae_espacial'], 1)} terminales por zona. Con validación aleatoria habría ganado
     {resumen['modelo_mejor_aleatoria']} (D² {fmt(resumen['d2_aleatoria'], 2)}): memoriza la geografía. Su error OOB también es
     optimista (R² {fmt(resumen['oob_r2_rf'], 2)} contra {fmt(resumen['r2_espacial_rf'], 2)} en comunas no vistas).
   - *Clasificación:* ¿a la zona le corresponde al menos un cajero? Ganó **{resumen['clasificador_elegido']}**
     (AUC {fmt(resumen['auc_clasificador'], 2)}): {fmt(resumen['zonas_sin_cajero_que_deberian'])} zonas sin ningún cajero
     deberían tener uno.
   - *Validación hacia el futuro:* los {resumen['cajeros_nuevos_osm']} cajeros que aparecen en OpenStreetMap y no estaban en
     2017 cayeron en el percentil {fmt(100 * resumen['percentil_cajeros_nuevos'])} de la brecha calculada con datos de 2017
     (p = {fmt(resumen['p_valor_cajeros_nuevos'], 3)}).{texto_pron}
4. **Prescriptivo.** K-Means ({resumen['k_clusters']} tipos de zona, silueta = {fmt(resumen['silueta'], 2)}) y **cobertura
   máxima**: programa lineal entero resuelto con HiGHS en menos de un segundo; el algoritmo goloso queda a menos del
   {fmt(resumen['goloso_max_diferencia_pct'], 1)} % del óptimo.{texto_tec}

**Distancias.** {texto_distancias}.
""")
    st.subheader("Regresión: comparación de modelos")
    orden = esp.sort_values("D² Poisson (promedio por partición)", ascending=False).index
    st.dataframe(pd.DataFrame({
        "D² en comunas no vistas (promedio)": esp.loc[orden, "D² Poisson (promedio por partición)"],
        "± desvío entre comunas": esp.loc[orden, "D² Poisson (desvío entre particiones)"],
        "Error medio (terminales por zona)": esp.loc[orden, "MAE"],
        "D² con validación aleatoria": ale.loc[orden, "D² Poisson (promedio por partición)"]}).rename_axis("Modelo"),
        column_config={c: st.column_config.NumberColumn(format="%.2f") for c in
                       ("D² en comunas no vistas (promedio)", "± desvío entre comunas", "Error medio (terminales por zona)",
                        "D² con validación aleatoria")})
    st.subheader("Clasificación: comparación de modelos")
    mc = D["metricas_clasif"]
    esp_c = mc[mc["validación"].str.startswith("espacial")].set_index("modelo")
    ale_c = mc[mc["validación"] == "aleatoria"].set_index("modelo")
    orden_c = esp_c.sort_values("AUC (promedio por partición)", ascending=False).index
    st.dataframe(pd.DataFrame({
        "AUC en comunas no vistas (promedio)": esp_c.loc[orden_c, "AUC (promedio por partición)"],
        "PR-AUC": esp_c.loc[orden_c, "PR-AUC"], "Precisión = recall": esp_c.loc[orden_c, "Precisión"],
        "AUC con validación aleatoria": ale_c.loc[orden_c, "AUC"]}).rename_axis("Modelo"),
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                       ("AUC en comunas no vistas (promedio)", "PR-AUC", "Precisión = recall", "AUC con validación aleatoria")})
    st.subheader("Qué variables pesan más")
    imp = (D["importancia"].set_index("nombre")["importancia"].rename("SHAP").to_frame()
           .join(D["gini"].set_index("nombre")["importancia"].rename("Gini")).rename_axis("Variable"))
    st.dataframe(imp, column_config={
        "SHAP": st.column_config.ProgressColumn("SHAP (modelo elegido)", format="%.2f", min_value=0, max_value=float(imp["SHAP"].max())),
        "Gini": st.column_config.ProgressColumn("Gini (Random Forest)", format="%.2f", min_value=0, max_value=float(imp["Gini"].max()))})
    st.caption("SHAP: cuánto mueve, en promedio, cada variable la predicción de una zona (escala logarítmica; 0,1 ≈ ±10 %). "
               "Gini: cuánto ordena cada variable los cortes del Random Forest.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Error OOB contra comunas no vistas")
        st.dataframe(D["oob"].rename(columns={"modelo": "Modelo", "métrica": "Métrica", "OOB": "OOB",
                                              "comunas no vistas": "Comunas no vistas"}),
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ("OOB", "Comunas no vistas")},
                     hide_index=True)
        st.caption("El error OOB usa filas que cada árbol no vio, pero de los mismos barrios: con datos espaciales es optimista.")
    with c2:
        st.subheader("Goloso vs. óptimo exacto")
        st.dataframe(D["comparacion"][["cajeros", "goloso", "exacto", "diferencia_pct", "seg_exacto"]].rename(columns={
            "cajeros": "Cajeros", "goloso": "Goloso", "exacto": "Exacto", "diferencia_pct": "Dif. %",
            "seg_exacto": "Segundos"}),
            column_config={"Goloso": st.column_config.NumberColumn(format="%d"),
                           "Exacto": st.column_config.NumberColumn(format="%d"),
                           "Dif. %": st.column_config.NumberColumn(format="%.1f"),
                           "Segundos": st.column_config.NumberColumn(format="%.2f")}, hide_index=True)
        st.caption("Vecinos que suman los cajeros nuevos con cada método; segundos que tarda el óptimo exacto.")
    st.markdown("""
**Limitaciones**
- El listado abierto de cajeros es un relevamiento de alrededor de 2017 (aparecen bancos que ya no existen con ese nombre).
  Por eso la app permite cargar el listado actual.
- No hay datos de transacciones (son privados): la demanda se estima con residentes, comercio y pasajeros.
- OpenStreetMap no tiene todos los comercios ni todas las sendas; su cobertura puede variar entre barrios.
- Los feriados que caen en día de semana cuentan como días hábiles en el promedio del subte.
- El pronóstico del subte se entrena con menos de dos años de datos y no anticipa paros, lluvias ni eventos.
- Lo que le "corresponde" a una zona sale de cómo están repartidos hoy los cajeros entre zonas con demanda parecida: toma
  al mercado como referencia, no mide extracciones (son datos privados de las redes).
""")
