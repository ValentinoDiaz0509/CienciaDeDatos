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
        "importancia": pd.read_csv(PROC / "importancia_variables.csv"),
        "comparacion": pd.read_csv(PROC / "goloso_vs_exacto.csv"),
        "cobertura": pd.read_csv(PROC / "cobertura_redes.csv"),
        "barrios": json.loads((BASE / "barrios.geojson").read_text(encoding="utf-8")),
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


# --------------------------------------------------------------------------
# Navegación
# --------------------------------------------------------------------------
st.sidebar.markdown("### 🏧 Cajeros Link en CABA")
st.sidebar.caption("Dónde faltan y dónde conviene sumar terminales")
pagina = st.sidebar.radio("Sección", ["Panorama", "Simulador de expansión", "Demanda por hora", "Tipos de zona",
                                      "Usá tus datos", "Cómo funciona"], label_visibility="collapsed")
st.sidebar.divider()
metodo = st.sidebar.radio("Distancia", ["Caminando por las calles", "En línea recta"] if HAY_RED else ["En línea recta"],
                          help="Caminando usa la red peatonal de OpenStreetMap (algoritmo de Dijkstra).")
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
    c4.metric("Link con 20 cajeros nuevos", pct(cl + optimo(CLAVE_METODO, COL_DIST, 500, 20) / res["poblacion"].sum()),
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

    st.subheader("Los 15 barrios con más terminales Link faltantes")
    por_barrio = (res.groupby("barrio").agg(residentes=("poblacion", "sum"), link=("term_link", "sum"),
                                            banelco=("term_banelco", "sum"),
                                            faltan=("brecha_link", lambda s: s.clip(lower=0).sum() / 7))
                  .sort_values("faltan", ascending=False).head(15))
    por_barrio["Link cada 10 mil vecinos"] = por_barrio["link"] / por_barrio["residentes"] * 1e4
    st.dataframe(por_barrio.rename(columns={"residentes": "Residentes", "link": "Terminales Link",
                                            "banelco": "Terminales Banelco", "faltan": "Terminales Link faltantes"}),
                 column_config={"Residentes": st.column_config.NumberColumn(format="%d"),
                                "Terminales Link faltantes": st.column_config.NumberColumn(format="%.0f"),
                                "Link cada 10 mil vecinos": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Terminales faltantes: suma de las brechas positivas del barrio, corregida por la superposición de "
               "las áreas de influencia (cada terminal cuenta en 7 hexágonos).")


# --------------------------------------------------------------------------
# 2. Simulador
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
    solucion = d_.radio("Método", ["Óptimo exacto", "Goloso"], horizontal=True,
                        help="Exacto: programación lineal entera (HiGHS). Goloso: agrega de a un sitio.")
    col_pax = {"Todo el día hábil": "pax_subte_habil", "Hora pico mañana (6-10 h)": "pax_subte_manana",
               "Hora pico tarde (16-20 h)": "pax_subte_tarde"}.get(franja, "pax_subte_habil")
    peso = 0.0 if franja.startswith("Ninguno") else 1.0

    conj = dict(CONJUNTOS[(CLAVE_METODO, radio)])
    conj["cubierto"] = res[COL_DIST].to_numpy() <= radio
    w = m.demanda(res, peso, col_pax)
    recs = (m.resolver_exacto(res, n, conj, w, tiempo_max=30) if solucion == "Óptimo exacto"
            else m.resolver_goloso(res, n, conj, w))
    pob = res["poblacion"].sum()
    antes = res.loc[conj["cubierto"], "poblacion"].sum()
    gan = recs["residentes_nuevos"].sum() if not recs.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Vecinos con Link cerca hoy", pct(antes / pob, 1))
    k2.metric(f"Con {n} cajeros nuevos", pct((antes + gan) / pob, 1), f"+{fmt(100 * gan / pob, 1)} puntos")
    k3.metric("Vecinos que ganan un Link cerca", fmt(gan))
    k4.metric("Extracciones por mes (estimadas)", fmt(len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES),
              help=f"{fmt(BCRA_EXTRACCIONES_POR_ATM_MES)} extracciones por terminal por mes, promedio nacional. {BCRA_FUENTE}.")

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
# 3. Demanda por hora
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


# --------------------------------------------------------------------------
# 4. Tipos de zona
# --------------------------------------------------------------------------
elif pagina == "Tipos de zona":
    st.title("Tipos de zona")
    st.markdown(f"Agrupamos los hexágonos en **{resumen['k_clusters']} tipos de zona** con K-Means, según residentes, "
                "comercio, pasajeros de subte, distancia al centro y perfil socioeconómico. Sirve para entender "
                "**dónde** está la brecha: no es lo mismo faltar en el centro que en un barrio residencial.")
    perfil = D["perfil"].set_index("tipo_zona")
    zonas = list(perfil.index)
    colores = {}
    i = 0
    for z in zonas:  # la zona sin vecinos va en gris: es fondo, no un segmento de negocio
        if z.startswith("Zonas sin vecinos"):
            colores[z] = "#c3c2b7"
        else:
            colores[z] = e.CATEGORICO[i]; i += 1
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
# 5. Usá tus datos
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
# 6. Cómo funciona
# --------------------------------------------------------------------------
else:
    st.title("Cómo funciona")
    st.markdown(f"""
**Pregunta.** ¿En qué zonas de CABA hay más demanda que cajeros Link, y dónde conviene sumar terminales?

**Unidad de análisis.** {fmt(resumen['hexagonos'])} hexágonos H3 (resolución 9, ≈{fmt(resumen['area_media_hex_km2'], 2)} km²).
Un cajero sirve a quien está a pocas cuadras, así que las variables se miden sobre el hexágono y sus 6 vecinos (≈400 m).

**Tres técnicas, cada una con su pregunta:**
1. **Regresión de conteos (supervisado).** ¿Cuántas terminales tienen las zonas con este perfil de demanda? Comparamos seis
   modelos y un ensamble con validación cruzada **espacial** (se deja afuera una comuna entera por vez). Ganó
   **{resumen['modelo_elegido']}** (D² de Poisson = {fmt(resumen['d2_espacial'], 2)}). Con validación aleatoria habría ganado
   {resumen['modelo_mejor_aleatoria']}: los modelos complejos memorizan la geografía. La **brecha** es lo esperado menos lo que hay,
   con predicciones fuera de muestra.
2. **K-Means (no supervisado).** {resumen['k_clusters']} tipos de zona (silueta = {fmt(resumen['silueta'], 2)}); y las estaciones de
   subte en *origen* y *destino* según su curva horaria.
3. **Cobertura máxima (optimización).** Programa lineal entero resuelto con HiGHS en menos de un segundo; el algoritmo goloso
   queda a menos del {fmt(resumen['goloso_max_diferencia_pct'], 1)} % del óptimo.

**Distancias.** {"Caminando por la red peatonal de OpenStreetMap (Dijkstra)" if HAY_RED else "En línea recta"}.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Comparación de modelos (validación espacial)")
        esp = D["metricas"][D["metricas"]["validación"].str.startswith("espacial")].drop(columns="validación")
        st.dataframe(esp.set_index("modelo").sort_values("D² Poisson", ascending=False).round(3))
        st.subheader("Goloso vs. óptimo exacto")
        st.dataframe(D["comparacion"][["cajeros", "goloso", "exacto", "diferencia_pct", "seg_exacto"]].round(2), hide_index=True)
    with c2:
        st.subheader("Qué variables pesan más")
        st.bar_chart(D["importancia"].set_index("nombre")["importancia"].head(10), horizontal=True, color=e.LINK)
    st.markdown("""
**Limitaciones**
- El listado abierto de cajeros es un relevamiento de alrededor de 2017 (aparecen bancos que ya no existen con ese nombre).
  Por eso la app permite cargar el listado actual.
- No hay datos de transacciones (son privados): la demanda se estima con residentes, comercio y pasajeros.
- OpenStreetMap no tiene todos los comercios ni todas las sendas; su cobertura puede variar entre barrios.
- Los feriados que caen en día de semana cuentan como días hábiles en el promedio del subte.
""")
