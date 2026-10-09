"""
features.py
-----------
Construye la tabla analítica del TPO: una fila por hexágono H3 de CABA con
variables de demanda (población, perfil socioeconómico, comercio, flujo de
pasajeros) y de oferta (terminales de cada red de cajeros).

Unidad de análisis
    Hexágonos H3 de resolución 9 (≈0,105 km², ≈ 3 x 3 manzanas). Los hexágonos
    tienen todos el mismo tamaño y los mismos vecinos, lo que hace comparables
    las zonas; es la grilla estándar de la industria para análisis urbano.

Área de influencia
    Un cajero sirve a quien vive o pasa a pocas cuadras. Por eso, además del
    valor de cada hexágono, se calcula la suma sobre el hexágono y su primer
    anillo de vecinos (7 hexágonos, radio ≈ 350-450 m). Las variables con sufijo
    `_k1` están medidas sobre esa área de influencia.

Uso:
    python -m src.features      # lee data/base, escribe data/processed/hexagonos.parquet
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

RAIZ = Path(__file__).resolve().parent.parent
DIR_BASE = RAIZ / "data" / "base"
DIR_PROC = RAIZ / "data" / "processed"

RES_H3 = 9
CRS_METRICO = "EPSG:5347"  # POSGAR 2007 / Argentina 5 (metros); distorsión despreciable en CABA
PLAZA_DE_MAYO = (-34.6083, -58.3712)
RADIO_COBERTURA_M = 500


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c)).lower().strip()
    return re.sub(r"\s+", " ", texto)


def hex_de(lat, lon, res: int = RES_H3) -> list[str]:
    return [h3.latlng_to_cell(a, b, res) for a, b in zip(lat, lon)]


def poligono_hex(h: str) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)])


def suma_k1(serie: pd.Series) -> pd.Series:
    """Suma de cada hexágono más sus 6 vecinos (área de influencia)."""
    valores = serie.to_dict()
    return pd.Series(
        {h: sum(valores.get(v, 0.0) for v in h3.grid_disk(h, 1)) for h in serie.index},
        name=f"{serie.name}_k1",
    )


# --------------------------------------------------------------------------
# Grilla
# --------------------------------------------------------------------------
# El archivo de BA Data trae los nombres en mayúsculas y sin tildes; los corregimos para mostrarlos.
_NOMBRES_BARRIOS = {
    "Agronomia": "Agronomía", "Constitucion": "Constitución", "Nuñez": "Núñez", "San Cristobal": "San Cristóbal",
    "San Nicolas": "San Nicolás", "Velez Sarsfield": "Vélez Sarsfield", "Villa Del Parque": "Villa del Parque",
    "Villa Gral. Mitre": "Villa General Mitre", "Villa Ortuzar": "Villa Ortúzar", "Villa Pueyrredon": "Villa Pueyrredón",
}


def cargar_barrios(base: Path = DIR_BASE) -> gpd.GeoDataFrame:
    b = gpd.read_file(base / "barrios.geojson").to_crs("EPSG:4326")
    b["barrio"] = b["nombre"].str.title().replace(_NOMBRES_BARRIOS)
    b["comuna"] = pd.to_numeric(b["comuna"], errors="coerce").astype("Int64")
    return b[["barrio", "comuna", "geometry"]]


def grilla(barrios: gpd.GeoDataFrame, puntos_extra: list[tuple[float, float]] = ()) -> gpd.GeoDataFrame:
    """Hexágonos cuyo centro cae en CABA, más los que contienen algún punto de interés."""
    caba = barrios.union_all()
    celdas = set(h3.geo_to_cells(caba.__geo_interface__, RES_H3))
    celdas |= {h3.latlng_to_cell(lat, lon, RES_H3) for lat, lon in puntos_extra}
    g = gpd.GeoDataFrame({"h3": sorted(celdas)}, geometry=[poligono_hex(h) for h in sorted(celdas)], crs="EPSG:4326")

    centros = [h3.cell_to_latlng(h) for h in g["h3"]]
    g["lat"] = [c[0] for c in centros]
    g["lon"] = [c[1] for c in centros]
    pts = gpd.GeoDataFrame(g[["h3"]], geometry=gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326")
    asignado = gpd.sjoin_nearest(pts.to_crs(CRS_METRICO), barrios.to_crs(CRS_METRICO), how="left")
    asignado = asignado.drop_duplicates("h3").set_index("h3")
    g["barrio"] = g["h3"].map(asignado["barrio"])
    g["comuna"] = g["h3"].map(asignado["comuna"])
    g["area_km2"] = g.to_crs(CRS_METRICO).area / 1e6
    # superficie del hexágono que cae dentro de CABA (los bordes y la costa quedan parciales)
    g["frac_en_caba"] = (
        g.to_crs(CRS_METRICO).intersection(gpd.GeoSeries([caba], crs="EPSG:4326").to_crs(CRS_METRICO).iloc[0]).area
        / g.to_crs(CRS_METRICO).area
    ).clip(0, 1)
    plaza = np.array(gpd.GeoSeries(gpd.points_from_xy([PLAZA_DE_MAYO[1]], [PLAZA_DE_MAYO[0]]), crs="EPSG:4326")
                     .to_crs(CRS_METRICO).iloc[0].coords[0])
    xy = np.column_stack([pts.to_crs(CRS_METRICO).geometry.x, pts.to_crs(CRS_METRICO).geometry.y])
    g["dist_centro_km"] = np.linalg.norm(xy - plaza, axis=1) / 1000
    return g


# --------------------------------------------------------------------------
# Censo 2022
# --------------------------------------------------------------------------
# (variable, categorías) -> columna. Los porcentajes se calculan después de agregar
# numeradores y denominadores, nunca promediando porcentajes.
_CENSO = {
    "poblacion": ("PERSONA_P02", None),
    "edad_0_14": ("PERSONA_EDADGRU", {"1"}),
    "edad_65_mas": ("PERSONA_EDADGRU", {"3"}),
    "ocupados": ("PERSONA_CONDACT", {"1"}),
    "pob_actividad": ("PERSONA_CONDACT", None),
    "universitarios": ("PERSONA_MNI", {"9", "10", "11"}),
    "pob_instruccion": ("PERSONA_MNI", {str(i) for i in range(1, 12)}),
    "hogares": ("HOGAR_NBI_TOT", None),
    "hogares_nbi": ("HOGAR_NBI_TOT", {"1"}),
    "hogares_sin_internet": ("HOGAR_H24A", {"2"}),
    "hogares_sin_computadora": ("HOGAR_H24C", {"2"}),
    "hogares_con_privacion": ("HOGAR_IPMH", {"2", "3", "4"}),
    "hogares_ipmh": ("HOGAR_IPMH", None),
}


def censo_por_radio(base: Path = DIR_BASE) -> pd.DataFrame:
    largo = pd.read_parquet(base / "censo2022-caba-largo.parquet")
    largo["valor_categoria"] = largo["valor_categoria"].astype(str)
    columnas = {}
    for nombre, (variable, cats) in _CENSO.items():
        sub = largo[largo["codigo_variable"] == variable]
        if cats is not None:
            sub = sub[sub["valor_categoria"].isin(cats)]
        columnas[nombre] = sub.groupby("id_geo")["conteo"].sum()
    return pd.DataFrame(columnas).fillna(0).rename_axis("id_geo").reset_index()


def interpolar_areal(radios: gpd.GeoDataFrame, hexes: gpd.GeoDataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Reparte los conteos de cada radio censal entre los hexágonos según la superficie
    compartida (interpolación areal ponderada por área). Conserva el total.
    """
    r = radios.to_crs(CRS_METRICO)
    r["area_radio"] = r.area
    inter = gpd.overlay(r[["id_geo", "area_radio", *columnas, "geometry"]],
                        hexes[["h3", "geometry"]].to_crs(CRS_METRICO), how="intersection", keep_geom_type=True)
    peso = inter.area / inter["area_radio"]
    agregado = inter[columnas].multiply(peso, axis=0)
    agregado["h3"] = inter["h3"].values
    return agregado.groupby("h3")[columnas].sum()


# --------------------------------------------------------------------------
# Subte: pasajeros por estación
# --------------------------------------------------------------------------
_ALIAS_ESTACIONES = {
    "rosas": "juan manuel de rosas",
    "pza. de los virreyes": "plaza de los virreyes",
    "patricios": "parque patricios",
    "urquiza": "general urquiza",
    "general belgrano": "belgrano",
    "mariano moreno": "moreno",
    "flores": "san jose de flores",
    "retiro e": "retiro",
}


def _base_osm(nombre: str) -> tuple[str, str | None]:
    letra = re.search(r"\(([A-H])\)", nombre)
    base = re.sub(r"\([A-H]\)", "", nombre)
    base = re.split(r" - |-Mezquita", base)[0]
    return normalizar(base), (letra.group(1) if letra else None)


ANIO_MOLINETES = 2025  # año completo más reciente (2026 tiene solo el primer semestre)


def molinetes(base: Path = DIR_BASE, anio: int = ANIO_MOLINETES) -> pd.DataFrame:
    """Pasajeros promedio por día, por estación, hora y tipo de día, sin filas de prueba."""
    mol = pd.read_csv(base / f"molinetes-{anio}-estacion-hora.csv")
    mol = mol[~mol["linea"].isin(["PRUEBA"]) & ~mol["estacion"].isin(["#N/D", "Prueba", "CochePM"])].copy()
    mol["letra"] = mol["linea"].str.replace("LINEA", "", regex=False).str[0]
    # "Callao.B" -> "callao" (el sufijo es la línea), sin romper "Leandro N. Alem"
    mol["base"] = mol["estacion"].str.replace(r"\.[A-H]$", "", regex=True).map(normalizar).replace(_ALIAS_ESTACIONES)
    # registros espurios: una estación cargada en una línea que no le corresponde, con ~0 pasajeros
    total = mol.groupby(["letra", "base"])["pasajeros_promedio_dia"].transform("sum")
    return mol[total >= 50]


def estaciones_subte(base: Path = DIR_BASE, anio: int = ANIO_MOLINETES) -> pd.DataFrame:
    """Pasajeros por día (hábil y fin de semana) de cada estación, con coordenadas de OSM."""
    mol = molinetes(base, anio)
    pax = (mol.pivot_table(index=["letra", "base"], columns="tipo_dia", values="pasajeros_promedio_dia", aggfunc="sum")
              .fillna(0).reset_index())

    osm = pd.read_csv(base / "osm-caba-puntos.csv")
    osm = osm[(osm["categoria"] == "estacion_subte") & osm["red"].fillna("").str.contains("Subte")]
    osm[["base", "letra_osm"]] = pd.DataFrame([_base_osm(n) for n in osm["nombre"]], index=osm.index)

    filas = []
    for _, est in pax.iterrows():
        cands = osm[osm["base"] == est["base"]]
        if len(cands) > 1:
            misma = cands[cands["letra_osm"] == est["letra"]]
            cands = misma if len(misma) else cands[cands["letra_osm"].isna()] if cands["letra_osm"].isna().any() else cands
        if cands.empty:
            continue
        c = cands.iloc[0]
        filas.append({"linea": est["letra"], "estacion": c["nombre"], "base": est["base"], "lat": c["lat"], "lon": c["lon"],
                      "pax_habil": est.get("habil", 0.0), "pax_finde": est.get("fin_de_semana", 0.0)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Puntos de interés y cajeros
# --------------------------------------------------------------------------
def contar_por_hex(df: pd.DataFrame, columna: str | None = None, peso: str | None = None) -> pd.DataFrame:
    """Cuenta (o suma `peso`) por hexágono; si hay `columna`, una columna por categoría."""
    d = df.dropna(subset=["lat", "lon"]).copy()
    d["h3"] = hex_de(d["lat"], d["lon"])
    valor = d[peso] if peso else pd.Series(1, index=d.index)
    d["_v"] = valor
    if columna is None:
        return d.groupby("h3")["_v"].sum().to_frame()
    return d.pivot_table(index="h3", columns=columna, values="_v", aggfunc="sum", fill_value=0)


def cargar_cajeros(base: Path = DIR_BASE) -> pd.DataFrame:
    c = pd.read_csv(base / "cajeros-automaticos.csv").rename(columns={"long": "lon"})
    c["red"] = c["red"].str.upper().str.strip()
    return c


def distancias_minimas(hexes: pd.DataFrame, puntos: pd.DataFrame) -> np.ndarray:
    """Distancia en metros desde el centro de cada hexágono al punto más cercano."""
    if puntos.empty:
        return np.full(len(hexes), np.inf)
    a = gpd.GeoSeries(gpd.points_from_xy(hexes["lon"], hexes["lat"]), crs="EPSG:4326").to_crs(CRS_METRICO)
    b = gpd.GeoSeries(gpd.points_from_xy(puntos["lon"], puntos["lat"]), crs="EPSG:4326").to_crs(CRS_METRICO)
    arbol = cKDTree(np.column_stack([b.x, b.y]))
    d, _ = arbol.query(np.column_stack([a.x, a.y]), k=1)
    return d


# --------------------------------------------------------------------------
# Usos del suelo (opcional: solo si ya hay coordenadas)
# --------------------------------------------------------------------------
_USOS = {
    "UNICOMERCIAL": "parcelas_comercio",
    "MULTICOMERCIAL": "parcelas_comercio",
    "OFICINAS": "parcelas_oficinas",
    "RESIDENCIAL": "parcelas_residencial",
    "EQUIPAMIENTO": "parcelas_equipamiento",
    "INDUSTRIAL": "parcelas_industria",
    "BARRIO POPULAR": "parcelas_barrio_popular",
}


def usos_por_hex(base: Path = DIR_BASE) -> pd.DataFrame | None:
    ruta = base / "usos-suelo-2022-2024-puntos.parquet"
    if not ruta.exists():
        return None
    u = pd.read_parquet(ruta)
    u.columns = [c.upper() for c in u.columns]
    u = u.rename(columns={"LAT": "lat", "LON": "lon"})
    if "ESTADO" in u:
        u = u[(u["ESTADO"].fillna("ACTIVO").str.upper() == "ACTIVO") | (u["TIPO1"].str.upper() == "BARRIO POPULAR")]
    u["uso"] = u["TIPO1"].str.upper().map(_USOS)
    return contar_por_hex(u.dropna(subset=["uso"]), "uso")


# --------------------------------------------------------------------------
# Tabla final
# --------------------------------------------------------------------------
def construir(base: Path = DIR_BASE, radios_path: Path | None = None) -> gpd.GeoDataFrame:
    barrios = cargar_barrios(base)
    cajeros = cargar_cajeros(base)
    osm = pd.read_csv(base / "osm-caba-puntos.csv")
    estaciones = estaciones_subte(base)

    hexes = grilla(barrios, puntos_extra=list(zip(cajeros["lat"], cajeros["lon"])))
    hexes = hexes.set_index("h3", drop=False)

    # --- Demanda: residentes (Censo 2022)
    radios_path = radios_path or base / "censo2022-caba-radios.geojson"
    radios = gpd.read_file(radios_path)[["id_geo", "geometry"]]
    radios = radios.merge(censo_por_radio(base), on="id_geo", how="inner")
    censo_hex = interpolar_areal(radios, hexes, list(_CENSO))
    hexes = hexes.join(censo_hex)

    # --- Demanda: actividad económica (OSM 2026) y usos del suelo (2022-2024)
    act = contar_por_hex(osm[osm["categoria"].isin(["comercio", "gastronomia", "farmacia", "salud", "educacion",
                                                     "estacion_servicio"])], "categoria")
    hexes = hexes.join(act.add_prefix("osm_"))
    usos = usos_por_hex(base)
    if usos is not None:
        hexes = hexes.join(usos)

    # --- Demanda: flujo de pasajeros (subte 2025) y estaciones de tren
    hexes = hexes.join(contar_por_hex(estaciones, peso="pax_habil").rename(columns={"_v": "pax_subte_habil"}))
    hexes = hexes.join(contar_por_hex(estaciones, peso="pax_finde").rename(columns={"_v": "pax_subte_finde"}))
    from .horario import pasajeros_por_franja, perfiles_estaciones

    franjas = pasajeros_por_franja(perfiles_estaciones(base)).merge(estaciones[["linea", "estacion", "lat", "lon"]],
                                                                    on=["linea", "estacion"])
    hexes = hexes.join(contar_por_hex(franjas, peso="pax_manana").rename(columns={"_v": "pax_subte_manana"}))
    hexes = hexes.join(contar_por_hex(franjas, peso="pax_tarde").rename(columns={"_v": "pax_subte_tarde"}))
    tren = osm[osm["categoria"] == "estacion_tren"]
    hexes = hexes.join(contar_por_hex(tren).rename(columns={"_v": "estaciones_tren"}))

    # --- Contexto de oferta (no entra al modelo de demanda): sucursales bancarias OSM
    hexes = hexes.join(contar_por_hex(osm[osm["categoria"] == "banco"]).rename(columns={"_v": "sucursales_osm"}))

    # --- Oferta 2017: terminales por red
    term = contar_por_hex(cajeros, "red", peso="terminales").rename(columns=lambda r: f"term_{r.lower()}")
    ubic = contar_por_hex(cajeros, "red").rename(columns=lambda r: f"ubic_{r.lower()}")
    hexes = hexes.join(term).join(ubic)

    numericas = [c for c in hexes.columns if c not in ("h3", "geometry", "barrio", "comuna", "lat", "lon",
                                                       "area_km2", "frac_en_caba", "dist_centro_km")]
    hexes[numericas] = hexes[numericas].fillna(0.0)
    hexes["term_total"] = hexes.filter(like="term_").sum(axis=1)
    hexes["actividad_osm"] = hexes.filter(like="osm_").sum(axis=1)

    # --- Área de influencia (hexágono + primer anillo)
    for col in [*_CENSO, "actividad_osm", "pax_subte_habil", "estaciones_tren", "term_total", "term_link",
                "term_banelco", *(usos.columns if usos is not None else [])]:
        hexes[f"{col}_k1"] = suma_k1(hexes[col])

    # --- Indicadores derivados (sobre el área de influencia)
    k = hexes
    k["densidad_k1"] = k["poblacion_k1"] / (7 * k["area_km2"])
    k["pct_nbi_k1"] = k["hogares_nbi_k1"] / k["hogares_k1"].replace(0, np.nan)
    k["pct_sin_internet_k1"] = k["hogares_sin_internet_k1"] / k["hogares_k1"].replace(0, np.nan)
    k["pct_sin_computadora_k1"] = k["hogares_sin_computadora_k1"] / k["hogares_k1"].replace(0, np.nan)
    k["pct_privacion_k1"] = k["hogares_con_privacion_k1"] / k["hogares_ipmh_k1"].replace(0, np.nan)
    k["pct_65_k1"] = k["edad_65_mas_k1"] / k["poblacion_k1"].replace(0, np.nan)
    k["pct_univ_k1"] = k["universitarios_k1"] / k["pob_instruccion_k1"].replace(0, np.nan)
    k["tasa_empleo_k1"] = k["ocupados_k1"] / k["pob_actividad_k1"].replace(0, np.nan)

    # --- Distancias al cajero más cercano de cada red (cobertura a pie)
    k["dist_link_m"] = distancias_minimas(k, cajeros[cajeros["red"] == "LINK"])
    k["dist_banelco_m"] = distancias_minimas(k, cajeros[cajeros["red"] == "BANELCO"])
    k["dist_cajero_m"] = np.minimum(k["dist_link_m"], k["dist_banelco_m"])
    k["dist_subte_m"] = distancias_minimas(k, estaciones)
    return k.reset_index(drop=True)


def main() -> None:
    DIR_PROC.mkdir(parents=True, exist_ok=True)
    tabla = construir()
    tabla.to_parquet(DIR_PROC / "hexagonos.parquet", index=False)
    print(f"{len(tabla):,} hexágonos · {tabla['poblacion'].sum():,.0f} habitantes · "
          f"{tabla['term_total'].sum():,.0f} terminales -> {DIR_PROC / 'hexagonos.parquet'}")


if __name__ == "__main__":
    main()
