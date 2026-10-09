"""
App del TPO: ¿Dónde faltan cajeros Link en CABA?

Correr desde la raíz del repo:
    streamlit run app/app.py

Funciona sin internet: lee los resultados precalculados de data/processed/. El mapa base
de calles es opcional (necesita conexión); sin él se dibujan los límites de los barrios.
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

PROC = RAIZ / "data" / "processed"
BASE = RAIZ / "data" / "base"

# Referencia pública para traducir terminales a operaciones (BCRA, Informe Mensual de
# Pagos Minoristas, agosto 2025, datos de julio 2025)
BCRA_EXTRACCIONES_POR_ATM_MES = 3242
BCRA_MONTO_PROMEDIO_EXTRACCION = 75_100  # pesos corrientes de julio 2025
BCRA_FUENTE = "BCRA, Informe Mensual de Pagos Minoristas (agosto 2025, datos de julio 2025)"

st.set_page_config(page_title="Cajeros Link · ¿Dónde faltan?", page_icon="🏧", layout="wide")


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------
@st.cache_data
def cargar():
    res = pd.read_parquet(PROC / "resultados_hex.parquet")
    resumen = json.loads((PROC / "resumen.json").read_text(encoding="utf-8"))
    cajeros = f.cargar_cajeros(BASE)
    estaciones = pd.read_csv(PROC / "estaciones_subte.csv")
    perfil = pd.read_csv(PROC / "clusters_perfil.csv")
    metricas = pd.read_csv(PROC / "metricas_modelos.csv")
    importancia = pd.read_csv(PROC / "importancia_variables.csv")
    barrios = json.loads((BASE / "barrios.geojson").read_text(encoding="utf-8"))
    return res, resumen, cajeros, estaciones, perfil, metricas, importancia, barrios


res, resumen, cajeros, estaciones, perfil, metricas, importancia, barrios = cargar()


def fmt(n: float, dec: int = 0) -> str:
    return f"{n:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, dec: int = 0) -> str:
    return fmt(100 * x, dec) + " %"


# --------------------------------------------------------------------------
# Capas de mapa
# --------------------------------------------------------------------------
VISTA = pdk.ViewState(latitude=-34.615, longitude=-58.445, zoom=11.2, pitch=40, bearing=0)


def capa_barrios():
    return pdk.Layer("GeoJsonLayer", barrios, stroked=True, filled=False, get_line_color=[82, 81, 78, 160],
                     line_width_min_pixels=1, pickable=False)


def capa_hex(df: pd.DataFrame, color_col: str = "color", elevacion: str | None = None, escala: float = 25):
    return pdk.Layer("H3HexagonLayer", df, get_hexagon="h3", get_fill_color=color_col, pickable=True,
                     stroked=False, filled=True, extruded=elevacion is not None,
                     get_elevation=elevacion or 0, elevation_scale=escala, opacity=0.85)


def capa_cajeros(df: pd.DataFrame, radio: int = 35):
    d = df.copy()
    d["color"] = d["red"].map({"LINK": e.hex_a_rgb(e.LINK, 230), "BANELCO": e.hex_a_rgb(e.BANELCO, 230)})
    d["tip"] = ("<b>" + d["banco"].fillna("").astype(str) + "</b><br/>Red " + d["red"].str.title() + " · "
                + d["terminales"].astype(str) + " terminales")
    return pdk.Layer("ScatterplotLayer", d, get_position=["lon", "lat"], get_fill_color="color", get_radius=radio,
                     radius_min_pixels=2, stroked=True, get_line_color=[252, 252, 251], line_width_min_pixels=1,
                     pickable=True)


def mapa(capas: list, tooltip: str, alto: int = 620, con_base: bool = False):
    deck = pdk.Deck(
        layers=[capa_barrios(), *capas], initial_view_state=VISTA,
        map_provider="carto" if con_base else None,
        map_style=pdk.map_styles.CARTO_LIGHT if con_base else None,
        tooltip={"html": tooltip, "style": {"backgroundColor": "#0b0b0b", "color": "white", "fontSize": "12px"}},
    )
    st.pydeck_chart(deck, height=alto)


def leyenda(items: list[tuple[str, str]]):
    html = " ".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;font-size:13px;color:#52514e">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};margin-right:6px;display:inline-block"></span>{t}</span>'
        for c, t in items)
    st.markdown(html, unsafe_allow_html=True)


def tabla_hex(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in ["poblacion", "term_link_k1", "term_banelco_k1", "esperado_link_k1", "brecha_link"]:
        d[c] = d[c].fillna(0)
    d["pob_txt"] = d["poblacion"].map(lambda v: fmt(v))
    d["link_txt"] = d["term_link_k1"].map(lambda v: fmt(v))
    d["banelco_txt"] = d["term_banelco_k1"].map(lambda v: fmt(v))
    d["esp_txt"] = d["esperado_link_k1"].map(lambda v: fmt(v, 1))
    d["brecha_txt"] = d["brecha_link"].map(lambda v: ("+" if v > 0 else "") + fmt(v, 1))
    d["dist_txt"] = d["dist_link_m"].map(lambda v: fmt(v) + " m")
    d["zona_txt"] = d["tipo_zona"].fillna("—")
    d["tip"] = ("<b>" + d["barrio"].fillna("") + "</b> · " + d["zona_txt"] + "<br/>Residentes en el hexágono: "
                + d["pob_txt"] + "<br/>Terminales en el área (≈400 m): Link " + d["link_txt"] + " · Banelco "
                + d["banelco_txt"] + "<br/>Link esperado según la demanda: " + d["esp_txt"]
                + "<br/><b>Brecha Link: " + d["brecha_txt"] + "</b><br/>Cajero Link más cercano: " + d["dist_txt"])
    return d


TOOLTIP = "{tip}"


# --------------------------------------------------------------------------
# Navegación
# --------------------------------------------------------------------------
st.sidebar.markdown("### 🏧 Cajeros Link en CABA")
st.sidebar.caption("Dónde faltan y dónde conviene sumar terminales")
pagina = st.sidebar.radio("Sección", ["Panorama", "Simulador de expansión", "Tipos de zona", "Usá tus datos",
                                      "Cómo funciona"], label_visibility="collapsed")
con_base = st.sidebar.toggle("Mapa de calles de fondo", value=False,
                             help="Necesita internet. Sin él se ven los límites de los barrios.")
st.sidebar.divider()
st.sidebar.caption("TPO Ciencia de Datos · UADE · 2C 2026\n\nDatos: BA Data (GCBA), INDEC Censo 2022, SBASE, "
                   "© colaboradores de OpenStreetMap. Cajeros: relevamiento de ~2017.")


# --------------------------------------------------------------------------
# 1. Panorama
# --------------------------------------------------------------------------
if pagina == "Panorama":
    st.title("¿Dónde faltan cajeros Link en CABA?")
    st.markdown(
        "Cruzamos dónde vive y circula la gente con dónde están los cajeros. Cada hexágono es una zona de unas "
        "3 × 3 manzanas; el color muestra la **brecha Link**: cuántas terminales Link le faltan (rojo) o le sobran "
        "(azul) a la zona comparada con zonas de demanda parecida.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vecinos con un Link a < 500 m", pct(resumen["cobertura_link"]),
              f"{pct(resumen['cobertura_link'] - resumen['cobertura_banelco'])} vs. Banelco", delta_color="normal")
    c2.metric("Vecinos con un Banelco a < 500 m", pct(resumen["cobertura_banelco"]))
    c3.metric("Vecinos sin un Link cerca", fmt(resumen["residentes_sin_link"]))
    c4.metric("Vecinos que suman 20 cajeros nuevos", f"+{fmt(resumen['nuevos_20'])}",
              help="Con las 20 ubicaciones que propone el simulador (radio de 500 m).")

    d = tabla_hex(res[res["esperado_link_k1"].notna()])
    d["color"] = e.color_divergente(d["brecha_link"])
    d["altura"] = d["brecha_link"].clip(lower=0)
    capas = [capa_hex(d, elevacion="altura", escala=60)]
    if st.checkbox("Mostrar cajeros (relevamiento ~2017)", value=False):
        capas.append(capa_cajeros(cajeros))
    mapa(capas, TOOLTIP, con_base=con_base)
    leyenda([(e.DIVERGENTE[-1], "Faltan terminales Link (más alto = más brecha)"), (e.DIVERGENTE[3], "En equilibrio"),
             (e.DIVERGENTE[0], "Sobran respecto de la demanda"), (e.LINK, "Cajero Link"), (e.BANELCO, "Cajero Banelco")])

    st.subheader("Los 15 barrios con más brecha Link")
    por_barrio = (res.groupby("barrio").agg(residentes=("poblacion", "sum"), link=("term_link", "sum"),
                                            banelco=("term_banelco", "sum"),
                                            brecha=("brecha_link", lambda s: s.clip(lower=0).sum() / 7))
                  .sort_values("brecha", ascending=False).head(15))
    por_barrio["Link cada 10 mil vecinos"] = por_barrio["link"] / por_barrio["residentes"] * 1e4
    st.dataframe(por_barrio.rename(columns={"residentes": "Residentes", "link": "Terminales Link",
                                            "banelco": "Terminales Banelco", "brecha": "Terminales Link faltantes"}),
                 column_config={"Residentes": st.column_config.NumberColumn(format="%d"),
                                "Terminales Link faltantes": st.column_config.NumberColumn(format="%.0f"),
                                "Link cada 10 mil vecinos": st.column_config.NumberColumn(format="%.1f")})
    st.caption("Terminales faltantes: suma de las brechas positivas de la zona, corregida por la superposición de "
               "las áreas de influencia (cada terminal cuenta en 7 hexágonos).")


# --------------------------------------------------------------------------
# 2. Simulador
# --------------------------------------------------------------------------
elif pagina == "Simulador de expansión":
    st.title("Simulador: ¿dónde poner los próximos cajeros Link?")
    st.markdown("Elegí cuántos cajeros sumar. El algoritmo busca, paso a paso, la ubicación que le acerca un Link "
                "a la mayor cantidad de personas que hoy no tienen ninguno a distancia caminable.")
    a, b, c = st.columns(3)
    n = a.slider("Cajeros nuevos", 1, 50, 20)
    radio = b.slider("Distancia caminable (m)", 300, 800, 500, step=50)
    peso = c.slider("Peso de los pasajeros de subte", 0.0, 1.0, 0.0, step=0.25,
                    help="0 = solo vecinos. 1 = cada persona que entra al subte en la zona un día hábil cuenta como un vecino.")

    base = res.copy()
    base["dist_link_m"] = f.distancias_minimas(base, cajeros[cajeros["red"] == "LINK"])
    recs = m.cobertura_golosa(base, n, radio_m=radio, peso_pasajeros=peso)
    pob = base["poblacion"].sum()
    antes = base.loc[base["dist_link_m"] <= radio, "poblacion"].sum()
    despues = antes + (recs["residentes_nuevos"].sum() if not recs.empty else 0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Vecinos con Link cerca hoy", pct(antes / pob, 1))
    k2.metric(f"Con {n} cajeros nuevos", pct(despues / pob, 1), f"+{pct((despues - antes) / pob, 1)}")
    k3.metric("Vecinos que ganan un Link cerca", fmt(despues - antes))
    k4.metric("Extracciones por mes (estimadas)", fmt(len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES),
              help=f"{fmt(BCRA_EXTRACCIONES_POR_ATM_MES)} extracciones por terminal por mes, promedio nacional. {BCRA_FUENTE}.")

    d = tabla_hex(base[base["poblacion"] > 0])
    d["sin_link"] = np.where(d["dist_link_m"] > radio, d["poblacion"], 0)
    d["color"] = e.color_secuencial(d["sin_link"], paradas=e.SECUENCIAL_NARANJA, alfa=170)
    capas = [capa_hex(d), capa_cajeros(cajeros[cajeros["red"] == "LINK"], radio=30)]
    if not recs.empty:
        r = recs.copy()
        r["color"] = [e.hex_a_rgb(e.SUGERIDO, 60)] * len(r)
        r["borde"] = [e.hex_a_rgb(e.SUGERIDO, 255)] * len(r)
        r["texto"] = r["orden"].astype(str)
        r["tip"] = ("<b>Sugerencia #" + r["texto"] + " · " + r["barrio"].astype(str) + "</b><br/>Suma "
                    + r["personas_nuevas"].map(lambda v: fmt(v)) + " personas con un Link a distancia caminable")
        capas.append(pdk.Layer("ScatterplotLayer", r, get_position=["lon", "lat"], get_radius=radio,
                               get_fill_color="color", get_line_color="borde", stroked=True,
                               line_width_min_pixels=2, pickable=True))
        capas.append(pdk.Layer("TextLayer", r, get_position=["lon", "lat"], get_text="texto", get_size=14,
                               get_color=[11, 11, 11], get_alignment_baseline="'center'"))
    mapa(capas, TOOLTIP, con_base=con_base)
    leyenda([(e.SECUENCIAL_NARANJA[4], "Más vecinos sin un Link cerca"), (e.LINK, "Cajero Link actual"),
             (e.SUGERIDO, "Ubicación sugerida (círculo = distancia caminable)")])

    if not recs.empty:
        st.subheader("Ubicaciones sugeridas, en orden de prioridad")
        t = recs[["orden", "barrio", "comuna", "personas_nuevas", "residentes_nuevos", "lat", "lon"]].copy()
        t["tipo de zona"] = t.index.map(lambda i: base.set_index("h3").loc[recs.loc[i, "h3"], "tipo_zona"])
        st.dataframe(t.rename(columns={"orden": "#", "barrio": "Barrio", "comuna": "Comuna",
                                       "personas_nuevas": "Personas que suma", "residentes_nuevos": "Vecinos que suma"}),
                     column_config={"Personas que suma": st.column_config.NumberColumn(format="%d"),
                                    "Vecinos que suma": st.column_config.NumberColumn(format="%d"),
                                    "lat": st.column_config.NumberColumn(format="%.5f"),
                                    "lon": st.column_config.NumberColumn(format="%.5f")},
                     hide_index=True)
        st.download_button("Descargar ubicaciones (CSV)", t.to_csv(index=False).encode("utf-8"),
                           "ubicaciones_sugeridas.csv", "text/csv")

        st.subheader("¿Cuánto mueve?")
        st.markdown(
            f"Si cada terminal nueva alcanza el promedio nacional de **{fmt(BCRA_EXTRACCIONES_POR_ATM_MES)} extracciones "
            f"por mes**, los {len(recs)} cajeros sumarían unas **{fmt(len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES)} "
            f"operaciones mensuales** en la red Link, por unos **${fmt(len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES * BCRA_MONTO_PROMEDIO_EXTRACCION / 1e6)} "
            f"millones** de efectivo por mes (monto promedio de ${fmt(BCRA_MONTO_PROMEDIO_EXTRACCION)} por extracción).")
        ingreso = st.number_input("Ingreso de Red Link por operación (dato interno, en pesos)", min_value=0.0,
                                  value=0.0, step=10.0)
        if ingreso > 0:
            st.metric("Ingreso mensual estimado", f"${fmt(len(recs) * BCRA_EXTRACCIONES_POR_ATM_MES * ingreso)}")
        st.caption(f"Fuente de los promedios: {BCRA_FUENTE}. Es un orden de magnitud: una terminal nueva en una zona "
                   "sin cajeros cerca puede estar por encima o por debajo del promedio del país.")


# --------------------------------------------------------------------------
# 3. Tipos de zona
# --------------------------------------------------------------------------
elif pagina == "Tipos de zona":
    st.title("Tipos de zona")
    st.markdown(f"Agrupamos los hexágonos en **{resumen['k_clusters']} tipos de zona** con K-Means, según residentes, "
                "comercio, pasajeros de subte, distancia al centro y perfil socioeconómico. Sirve para entender "
                "**dónde** está la brecha: no es lo mismo faltar en el centro que en un barrio residencial.")
    zonas = sorted(res["tipo_zona"].dropna().unique())
    colores = {z: e.CATEGORICO[i % len(e.CATEGORICO)] for i, z in enumerate(zonas)}
    elegidas = st.multiselect("Tipos de zona a mostrar", zonas, default=zonas)
    d = tabla_hex(res[res["tipo_zona"].isin(elegidas)])
    d["color"] = d["tipo_zona"].map(lambda z: e.hex_a_rgb(colores[z], 190))
    mapa([capa_hex(d)], TOOLTIP, con_base=con_base)
    leyenda([(colores[z], z) for z in zonas])

    p = perfil.set_index("tipo_zona")
    vista = pd.DataFrame({
        "Hexágonos": p["hexagonos"], "Residentes": p["residentes"], "Pasajeros de subte por día": p["pax_subte"],
        "Terminales cada 10 mil vecinos": p["terminales_cada_10mil"], "Link cada 10 mil vecinos": p["link_cada_10mil"],
        "% hogares con NBI (mediana)": 100 * p["pct_nbi_k1"], "Distancia al centro, km (mediana)": p["dist_centro_km"],
    })
    st.dataframe(vista,
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in vista.columns
                                if c not in ("Hexágonos", "Residentes", "Pasajeros de subte por día")} |
                               {c: st.column_config.NumberColumn(format="%d") for c in ("Hexágonos", "Residentes", "Pasajeros de subte por día")})


# --------------------------------------------------------------------------
# 4. Usá tus datos
# --------------------------------------------------------------------------
elif pagina == "Usá tus datos":
    st.title("Usá tu listado actual de cajeros")
    st.markdown(
        "El análisis público usa el último listado abierto de cajeros (≈2017). Red Link tiene el listado al día: "
        "si lo cargás acá, se recalculan la cobertura y las ubicaciones sugeridas. **El archivo no sale de tu "
        "computadora**: la app corre en forma local.")
    st.markdown("El CSV necesita las columnas `lat`, `lon` (o `long`) y `red` (LINK / BANELCO). "
                "`terminales` es opcional (si falta, cuenta 1 por fila).")
    archivo = st.file_uploader("Listado de cajeros (CSV)", type="csv")
    if archivo is not None:
        nuevo = pd.read_csv(archivo)
        nuevo.columns = [c.lower().strip() for c in nuevo.columns]
        nuevo = nuevo.rename(columns={"long": "lon", "longitud": "lon", "latitud": "lat"})
        faltan = {"lat", "lon", "red"} - set(nuevo.columns)
        if faltan:
            st.error(f"Faltan columnas: {', '.join(sorted(faltan))}")
        else:
            nuevo["red"] = nuevo["red"].astype(str).str.upper().str.strip()
            if "terminales" not in nuevo:
                nuevo["terminales"] = 1
            if "banco" not in nuevo:
                nuevo["banco"] = ""
            radio = st.slider("Distancia caminable (m)", 300, 800, 500, step=50)
            n = st.slider("Cajeros nuevos a ubicar", 1, 50, 20)
            base = res.copy()
            filas = []
            for nombre, caj in [("Listado público (~2017)", cajeros), ("Tu listado", nuevo)]:
                dl = f.distancias_minimas(base, caj[caj["red"] == "LINK"])
                db = f.distancias_minimas(base, caj[caj["red"] == "BANELCO"])
                pobt = base["poblacion"].sum()
                filas.append({"Listado": nombre, "Terminales Link": caj.loc[caj["red"] == "LINK", "terminales"].sum(),
                              "Terminales Banelco": caj.loc[caj["red"] == "BANELCO", "terminales"].sum(),
                              "Vecinos con Link cerca": base.loc[dl <= radio, "poblacion"].sum() / pobt,
                              "Vecinos con Banelco cerca": base.loc[db <= radio, "poblacion"].sum() / pobt})
            st.dataframe(pd.DataFrame(filas), hide_index=True,
                         column_config={"Vecinos con Link cerca": st.column_config.NumberColumn(format="%.3f"),
                                        "Vecinos con Banelco cerca": st.column_config.NumberColumn(format="%.3f")})
            base["dist_link_m"] = f.distancias_minimas(base, nuevo[nuevo["red"] == "LINK"])
            recs = m.cobertura_golosa(base, n, radio_m=radio)
            st.subheader("Ubicaciones sugeridas con tu listado")
            capas = [capa_cajeros(nuevo, radio=30)]
            if not recs.empty:
                r = recs.copy()
                r["color"] = [e.hex_a_rgb(e.SUGERIDO, 60)] * len(r)
                r["tip"] = "<b>Sugerencia #" + r["orden"].astype(str) + " · " + r["barrio"].astype(str) + "</b>"
                capas.append(pdk.Layer("ScatterplotLayer", r, get_position=["lon", "lat"], get_radius=radio,
                                       get_fill_color="color", get_line_color=e.hex_a_rgb(e.SUGERIDO), stroked=True,
                                       line_width_min_pixels=2, pickable=True))
                st.dataframe(recs[["orden", "barrio", "comuna", "residentes_nuevos", "lat", "lon"]], hide_index=True)
            mapa(capas, TOOLTIP, con_base=con_base)
    else:
        with st.expander("¿Qué formato tiene que tener?"):
            st.dataframe(cajeros[["lat", "lon", "red", "terminales", "banco"]].head(5), hide_index=True)
            st.download_button("Descargar el listado público como plantilla",
                               cajeros[["lat", "lon", "red", "terminales", "banco"]].to_csv(index=False).encode("utf-8"),
                               "plantilla_cajeros.csv", "text/csv")


# --------------------------------------------------------------------------
# 5. Cómo funciona
# --------------------------------------------------------------------------
else:
    st.title("Cómo funciona")
    st.markdown(f"""
**Pregunta.** ¿En qué zonas de CABA hay más demanda que cajeros Link, y dónde conviene sumar terminales?

**Unidad de análisis.** {fmt(resumen['hexagonos'])} hexágonos H3 (resolución 9, ≈{fmt(resumen['area_media_hex_km2'], 2)} km²).
Cada cajero sirve a quien está a pocas cuadras, así que las variables se miden sobre el hexágono y sus 6 vecinos (≈400 m).

**Tres técnicas, cada una con su pregunta:**
1. **Regresión** — ¿cuántas terminales tienen las zonas con este perfil de demanda? Comparamos seis modelos con
   validación cruzada **espacial** (se deja afuera una comuna entera por vez). Ganó **{resumen['modelo_elegido']}**
   (D² de Poisson = {fmt(resumen['d2_espacial'], 2)}). La **brecha** es lo esperado menos lo que hay, usando predicciones
   fuera de muestra para que una zona no se "explique" con sus propios cajeros.
2. **Clustering (K-Means)** — {resumen['k_clusters']} tipos de zona para entender dónde está la brecha.
3. **Optimización (cobertura máxima, algoritmo goloso)** — elige las ubicaciones que suman más personas con un Link a
   distancia caminable. Garantiza al menos el 63 % del óptimo teórico.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Comparación de modelos (validación espacial)")
        esp = metricas[metricas["validación"].str.startswith("espacial")].drop(columns="validación")
        st.dataframe(esp.set_index("modelo").round(3))
    with c2:
        st.subheader("Qué variables pesan más")
        st.bar_chart(importancia.set_index("nombre")["importancia"].head(10), horizontal=True, color=e.LINK)
    st.markdown("""
**Limitaciones que hay que tener presentes**
- El listado abierto de cajeros es un relevamiento de alrededor de 2017 (aparecen bancos que ya no existen con ese nombre).
  Por eso la app permite cargar el listado actual.
- No hay datos de transacciones (son privados): la demanda se estima con residentes, comercio y pasajeros.
- Las distancias son en línea recta; caminando son algo mayores.
- OpenStreetMap no tiene todos los comercios; su cobertura puede variar entre barrios.
""")
