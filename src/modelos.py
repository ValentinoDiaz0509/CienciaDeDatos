"""
modelos.py
----------
Piezas de modelado que comparten el pipeline, los notebooks y la app:

1. Preparación y métricas de la regresión de conteos (los modelos de machine learning,
   su validación espacial anidada y su explicación están en ml.py). La diferencia entre
   lo que le corresponde a cada zona y lo que tiene es la BRECHA.
2. Clustering (no supervisado) -> ¿Qué tipos de zona hay? (segmentación para
   explicar dónde está la brecha).
3. Optimización (prescriptivo) -> Si Red Link promueve N cajeros nuevos, ¿dónde
   conviene ubicarlos para sumar la mayor cantidad de vecinos con un Link cerca?
4. Validación temporal con los cajeros que aparecieron en OpenStreetMap después de 2017.

Todas las funciones reciben la tabla de `features.construir()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

SEMILLA = 42
CRS_METRICO = "EPSG:5347"

# Variables de DEMANDA (área de influencia). No se usan sucursales bancarias ni
# cajeros: son oferta, y meterlas haría que el modelo "explique" cajeros con cajeros.
VARIABLES_BASE = [
    "poblacion_k1",
    "actividad_osm_k1",
    "pax_subte_habil_k1",
    "estaciones_tren_k1",
    "dist_centro_km",
    "dist_subte_m",
    "pct_nbi_k1",
    "pct_sin_internet_k1",
    "pct_65_k1",
    "pct_univ_k1",
    "tasa_empleo_k1",
]
VARIABLES_USOS = ["parcelas_comercio_k1", "parcelas_oficinas_k1", "parcelas_residencial_k1"]

NOMBRES = {
    "poblacion_k1": "Residentes en el área",
    "actividad_osm_k1": "Comercios y servicios (OSM)",
    "pax_subte_habil_k1": "Pasajeros de subte por día hábil",
    "estaciones_tren_k1": "Estaciones de tren",
    "dist_centro_km": "Distancia a Plaza de Mayo (km)",
    "dist_subte_m": "Distancia al subte (m)",
    "pct_nbi_k1": "% hogares con NBI",
    "pct_sin_internet_k1": "% hogares sin internet",
    "pct_65_k1": "% de 65 años o más",
    "pct_univ_k1": "% con nivel universitario",
    "tasa_empleo_k1": "Tasa de empleo",
    "parcelas_comercio_k1": "Parcelas comerciales",
    "parcelas_oficinas_k1": "Parcelas de oficinas",
    "parcelas_residencial_k1": "Parcelas residenciales",
}

OBJETIVO = "term_total_k1"


# --------------------------------------------------------------------------
# Datos para el modelo
# --------------------------------------------------------------------------
def variables_modelo(tabla: pd.DataFrame) -> list[str]:
    return VARIABLES_BASE + [v for v in VARIABLES_USOS if v in tabla.columns]


def filas_modelables(tabla: pd.DataFrame) -> pd.Series:
    """Hexágonos con algo de demanda (deja afuera río, reservas y playones vacíos)."""
    return (tabla["frac_en_caba"] >= 0.5) & ((tabla["poblacion_k1"] >= 50) | (tabla["actividad_osm_k1"] > 0)
                                            | (tabla["pax_subte_habil_k1"] > 0))


def preparar(tabla: pd.DataFrame):
    cols = variables_modelo(tabla)
    m = filas_modelables(tabla)
    X = tabla.loc[m, cols].copy()
    X = X.fillna(X.median())
    y = tabla.loc[m, OBJETIVO].astype(float)
    grupos = tabla.loc[m, "comuna"].astype(int)
    return X, y, grupos, m


# --------------------------------------------------------------------------
# 1. Regresión
# --------------------------------------------------------------------------
def metricas(y: np.ndarray, pred: np.ndarray) -> dict:
    pred = np.clip(pred, 1e-6, None)
    dev = mean_poisson_deviance(y, pred)
    dev0 = mean_poisson_deviance(y, np.full_like(y, y.mean()))
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R²": r2_score(y, pred),
        "D² Poisson": 1 - dev / dev0,  # % de la devianza explicada (el R² de los modelos de conteo)
    }


def calcular_brechas(tabla: pd.DataFrame, mascara: pd.Series, esperado: np.ndarray) -> pd.DataFrame:
    """
    Brecha total = terminales esperadas (fuera de muestra) - terminales reales en el área.
    Brecha Link  = lo que Link tendría con su participación promedio en la ciudad - lo que tiene.
    Positivo = faltan terminales.
    """
    t = tabla.copy()
    t["esperado_total_k1"] = np.nan
    t.loc[mascara, "esperado_total_k1"] = esperado
    cuota_link = t["term_link"].sum() / t["term_total"].sum()
    t["esperado_link_k1"] = t["esperado_total_k1"] * cuota_link
    t["brecha_total"] = t["esperado_total_k1"] - t["term_total_k1"]
    t["brecha_link"] = t["esperado_link_k1"] - t["term_link_k1"]
    t.attrs["cuota_link"] = cuota_link
    return t


# --------------------------------------------------------------------------
# 2. Clustering
# --------------------------------------------------------------------------
VARIABLES_CLUSTER = ["poblacion_k1", "actividad_osm_k1", "pax_subte_habil_k1", "dist_centro_km",
                     "pct_nbi_k1", "pct_65_k1", "pct_univ_k1"]


def matriz_cluster(tabla: pd.DataFrame, mascara: pd.Series) -> np.ndarray:
    Z = tabla.loc[mascara, VARIABLES_CLUSTER].copy()
    Z = Z.fillna(Z.median())
    for c in ["poblacion_k1", "actividad_osm_k1", "pax_subte_habil_k1"]:
        Z[c] = np.log1p(Z[c])  # conteos muy asimétricos
    return StandardScaler().fit_transform(Z)


def elegir_k(Z: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(Z)
        filas.append({"k": k, "inercia": km.inertia_, "silueta": silhouette_score(Z, km.labels_, random_state=SEMILLA)})
    return pd.DataFrame(filas)


def agrupar(Z: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit_predict(Z)


def nombrar_clusters(tabla: pd.DataFrame, columna: str = "cluster") -> dict:
    """Nombre de negocio para cada grupo según su perfil (comparado con la mediana de la ciudad)."""
    perfil = tabla.groupby(columna)[VARIABLES_CLUSTER].median()
    med = tabla[VARIABLES_CLUSTER].median()
    nombres = {}
    for c, p in perfil.iterrows():
        if p["poblacion_k1"] < 0.1 * med["poblacion_k1"]:
            nombres[c] = "Zonas sin vecinos (parques, puerto, playones)"
        elif p["dist_centro_km"] < 2.5 and p["actividad_osm_k1"] > 2 * med["actividad_osm_k1"]:
            nombres[c] = "Centro de negocios"
        elif p["pct_nbi_k1"] > 2 * med["pct_nbi_k1"]:
            nombres[c] = "Mayor vulnerabilidad social"
        elif p["pax_subte_habil_k1"] > 0 and p["actividad_osm_k1"] > med["actividad_osm_k1"]:
            nombres[c] = "Corredores comerciales con subte"
        elif p["poblacion_k1"] > med["poblacion_k1"]:
            nombres[c] = "Residencial denso"
        else:
            nombres[c] = "Residencial de baja densidad"
    # si dos grupos quedan con el mismo nombre, se diferencian por densidad (A = más poblado)
    repetidos = {n for n in nombres.values() if list(nombres.values()).count(n) > 1}
    for n in repetidos:
        grupo = [c for c in nombres if nombres[c] == n]
        for i, c in enumerate(sorted(grupo, key=lambda c: -perfil.loc[c, "poblacion_k1"])):
            nombres[c] = f"{n} {'ABCDEFG'[i]}"
    return nombres


# --------------------------------------------------------------------------
# 3. Optimización: problema de cobertura máxima (MCLP)
# --------------------------------------------------------------------------
def _xy(tabla: pd.DataFrame) -> np.ndarray:
    import geopandas as gpd

    p = gpd.GeoSeries(gpd.points_from_xy(tabla["lon"], tabla["lat"]), crs="EPSG:4326").to_crs(CRS_METRICO)
    return np.column_stack([p.x, p.y])


def demanda(tabla: pd.DataFrame, peso_pasajeros: float = 0.0, columna_pasajeros: str = "pax_subte_habil") -> np.ndarray:
    """Personas a cubrir por hexágono: residentes + (opcional) pasajeros de subte de un día hábil."""
    return tabla["poblacion"].to_numpy(float) + peso_pasajeros * tabla[columna_pasajeros].to_numpy(float)


def es_candidato(tabla: pd.DataFrame) -> np.ndarray:
    """Un cajero necesita un local: hexágonos con comercio o servicios (OSM o usos del suelo)."""
    comercio = tabla.get("parcelas_comercio", pd.Series(0, index=tabla.index)).fillna(0)
    return ((tabla["actividad_osm"] + comercio) > 0).to_numpy() & (tabla["frac_en_caba"] >= 0.5).to_numpy()


def conjuntos_recta(tabla: pd.DataFrame, radio_m: float = 500) -> dict:
    """
    Estructura de cobertura en línea recta: para cada sitio candidato, qué hexágonos quedan
    a menos de `radio_m`; y qué hexágonos ya tienen un Link a esa distancia.
    """
    xy = _xy(tabla)
    cand = np.flatnonzero(es_candidato(tabla))
    vecinos = [np.asarray(v, dtype=int) for v in cKDTree(xy).query_ball_point(xy[cand], r=radio_m)]
    return {"candidatos": cand, "vecinos": vecinos, "cubierto": tabla["dist_link_m"].to_numpy() <= radio_m,
            "metodo": "línea recta", "radio_m": radio_m}


def _fila_sitio(tabla: pd.DataFrame, h: int, orden: int, personas: float, nuevos: np.ndarray) -> dict:
    fila = tabla.iloc[h]
    return {"orden": orden, "h3": fila["h3"], "lat": fila["lat"], "lon": fila["lon"], "barrio": fila["barrio"],
            "comuna": fila["comuna"], "personas_nuevas": float(personas),
            "residentes_nuevos": float(tabla["poblacion"].to_numpy()[nuevos].sum())}


def resolver_goloso(tabla: pd.DataFrame, n_nuevos: int, conj: dict, w: np.ndarray) -> pd.DataFrame:
    """
    Algoritmo goloso: en cada paso agrega el sitio con mayor ganancia marginal. Como la cobertura
    es submodular, garantiza al menos el 63 % (1 - 1/e) del óptimo (Nemhauser, Wolsey y Fisher, 1978).
    """
    cubierto = conj["cubierto"].copy()
    elegidos = []
    for paso in range(n_nuevos):
        ganancias = np.array([w[v][~cubierto[v]].sum() for v in conj["vecinos"]])
        i = int(np.argmax(ganancias))
        if ganancias[i] <= 0:
            break
        nuevos = conj["vecinos"][i][~cubierto[conj["vecinos"][i]]]
        cubierto[nuevos] = True
        elegidos.append(_fila_sitio(tabla, conj["candidatos"][i], paso + 1, ganancias[i], nuevos))
    res = pd.DataFrame(elegidos)
    if not res.empty:
        res["personas_acumuladas"] = res["personas_nuevas"].cumsum()
    res.attrs.update(cubierto_final=cubierto, objetivo=float(res["personas_nuevas"].sum()) if not res.empty else 0.0)
    return res


def resolver_exacto(tabla: pd.DataFrame, n_nuevos: int, conj: dict, w: np.ndarray, tiempo_max: float = 120) -> pd.DataFrame:
    """
    MCLP como programa lineal entero (Church y ReVelle, 1974), resuelto con HiGHS:

        max  sum_i w_i y_i
        s.a. y_i <= sum_{j cubre i} x_j     para cada zona i sin Link cerca
             sum_j x_j <= N
             x_j en {0, 1},  0 <= y_i <= 1

    Los sitios elegidos se ordenan después por ganancia marginal para presentarlos.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    pendiente = (~conj["cubierto"]) & (w > 0)
    # zonas que algún candidato puede cubrir
    filas_i = {}
    pares = []
    for jj, v in enumerate(conj["vecinos"]):
        for i in v[pendiente[v]]:
            filas_i.setdefault(int(i), len(filas_i))
            pares.append((filas_i[int(i)], jj))
    if not pares:
        return pd.DataFrame()
    usados = sorted({jj for _, jj in pares})
    col_j = {jj: k for k, jj in enumerate(usados)}
    nJ, nI = len(usados), len(filas_i)
    zonas = np.array(sorted(filas_i, key=filas_i.get))
    c = np.concatenate([np.zeros(nJ), -w[zonas]])
    r = np.array([p[0] for p in pares] + list(range(nI)))
    q = np.array([col_j[p[1]] for p in pares] + [nJ + i for i in range(nI)])
    val = np.array([-1.0] * len(pares) + [1.0] * nI)
    A = coo_matrix((val, (r, q)), shape=(nI, nJ + nI)).tocsr()
    restricciones = [LinearConstraint(A, -np.inf, 0),
                     LinearConstraint(np.concatenate([np.ones(nJ), np.zeros(nI)])[None, :], 0, n_nuevos)]
    integralidad = np.concatenate([np.ones(nJ), np.zeros(nI)])
    sol = milp(c, constraints=restricciones, integrality=integralidad, bounds=Bounds(0, 1),
               options={"time_limit": tiempo_max, "mip_rel_gap": 1e-6, "disp": False})
    if sol.x is None:
        raise RuntimeError(f"HiGHS no encontró solución: {sol.message}")
    elegidos_j = [usados[k] for k in np.flatnonzero(sol.x[:nJ] > 0.5)]
    res = _ordenar_sitios(tabla, elegidos_j, conj, w)
    cota = -getattr(sol, "mip_dual_bound", np.nan) if getattr(sol, "mip_dual_bound", None) is not None else np.nan
    res.attrs.update(objetivo=float(-sol.fun), estado=sol.message,
                     gap=float(getattr(sol, "mip_gap", np.nan) or 0.0), cota=cota)
    return res


def _ordenar_sitios(tabla: pd.DataFrame, elegidos_j: list[int], conj: dict, w: np.ndarray) -> pd.DataFrame:
    """Ordena los sitios elegidos por ganancia marginal, como si se instalaran de a uno."""
    cubierto = conj["cubierto"].copy()
    restantes, filas = list(elegidos_j), []
    while restantes:
        gan = [w[conj["vecinos"][jj]][~cubierto[conj["vecinos"][jj]]].sum() for jj in restantes]
        k = int(np.argmax(gan))
        jj = restantes.pop(k)
        nuevos = conj["vecinos"][jj][~cubierto[conj["vecinos"][jj]]]
        cubierto[nuevos] = True
        filas.append(_fila_sitio(tabla, conj["candidatos"][jj], len(filas) + 1, gan[k], nuevos))
    res = pd.DataFrame(filas)
    if not res.empty:
        res["personas_acumuladas"] = res["personas_nuevas"].cumsum()
    res.attrs.update(cubierto_final=cubierto, objetivo=float(res["personas_nuevas"].sum()) if not res.empty else 0.0)
    return res


def resolver_kmeans(tabla: pd.DataFrame, n_nuevos: int, conj: dict, w: np.ndarray, semilla: int = SEMILLA) -> pd.DataFrame:
    """
    K-Means ponderado, como se propuso en la clase 11 para ubicar cajeros: agrupa a las personas que hoy no tienen un
    Link cerca en n grupos (cada hexágono pesa según su gente) y pone un cajero en el sitio candidato más cercano al
    centro de cada grupo. K-Means minimiza la distancia al cuadrado de esas personas al centro de su grupo: no busca que
    la mayor cantidad quede a la distancia máxima. Sirve para comparar con el óptimo.
    """
    xy = _xy(tabla)
    pendiente = (~conj["cubierto"]) & (w > 0)
    n = min(n_nuevos, int(pendiente.sum()))
    if n == 0:
        return pd.DataFrame()
    km = KMeans(n_clusters=n, n_init=10, random_state=semilla).fit(xy[pendiente], sample_weight=w[pendiente])
    vecinos = min(10, len(conj["candidatos"]))
    _, idx = cKDTree(xy[conj["candidatos"]]).query(km.cluster_centers_, k=vecinos)
    elegidos = []
    for fila in np.atleast_2d(idx):   # el sitio candidato más cercano a cada centro, sin repetir
        jj = next((int(j) for j in fila if int(j) not in elegidos), None)
        if jj is not None:
            elegidos.append(jj)
    res = _ordenar_sitios(tabla, elegidos, conj, w)
    res.attrs.update(centros=km.cluster_centers_)
    return res


def comparar_goloso_exacto(tabla: pd.DataFrame, conj: dict, w: np.ndarray, ns=(5, 10, 20, 30, 50)) -> pd.DataFrame:
    import time

    filas = []
    for n in ns:
        t0 = time.time(); g = resolver_goloso(tabla, n, conj, w); tg = time.time() - t0
        t0 = time.time(); x = resolver_exacto(tabla, n, conj, w); tx = time.time() - t0
        filas.append({"cajeros": n, "goloso": g.attrs["objetivo"], "exacto": x.attrs["objetivo"],
                      "diferencia_pct": 100 * (x.attrs["objetivo"] - g.attrs["objetivo"]) / x.attrs["objetivo"],
                      "sitios_en_comun": len(set(g["h3"]) & set(x["h3"])),
                      "seg_goloso": tg, "seg_exacto": tx, "estado": x.attrs["estado"]})
    return pd.DataFrame(filas)


def resumen_cobertura(tabla: pd.DataFrame, radio_m: float = 500, sufijo: str = "") -> pd.DataFrame:
    pob = tabla["poblacion"]
    total = pob.sum()
    filas = []
    for red, col in [("Link", "dist_link_m"), ("Banelco", "dist_banelco_m"), ("Cualquier red", "dist_cajero_m")]:
        cub = pob[tabla[col + sufijo] <= radio_m].sum()
        filas.append({"red": red, "residentes_cubiertos": cub, "pct_residentes": cub / total})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Validación temporal: ¿dónde aparecieron cajeros nuevos entre 2017 y 2026?
# --------------------------------------------------------------------------
def cajeros_nuevos_osm(osm: pd.DataFrame, cajeros_2017: pd.DataFrame, umbral_m: float = 150) -> pd.DataFrame:
    """Cajeros mapeados en OSM (2026) que no tienen ningún cajero del listado 2017 a menos de `umbral_m`."""
    import geopandas as gpd

    a = osm[osm["categoria"] == "cajero"].copy()
    pa = gpd.GeoSeries(gpd.points_from_xy(a["lon"], a["lat"]), crs="EPSG:4326").to_crs(CRS_METRICO)
    pb = gpd.GeoSeries(gpd.points_from_xy(cajeros_2017["lon"], cajeros_2017["lat"]), crs="EPSG:4326").to_crs(CRS_METRICO)
    d, _ = cKDTree(np.column_stack([pb.x, pb.y])).query(np.column_stack([pa.x, pa.y]), k=1)
    a["dist_cajero_2017_m"] = d
    return a[a["dist_cajero_2017_m"] > umbral_m]


def auc(puntaje: np.ndarray, etiqueta: np.ndarray) -> float:
    """Probabilidad de que un hexágono con cajero nuevo tenga más puntaje que uno sin (Mann-Whitney)."""
    from scipy.stats import rankdata

    pos, neg = etiqueta == 1, etiqueta == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = rankdata(puntaje)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def percentil_cajeros_nuevos(res: pd.DataFrame, mascara: pd.Series, nuevos: pd.DataFrame, columna: str = "brecha_total",
                             simulaciones: int = 20000) -> dict:
    """
    ¿En qué percentil de la brecha 2017 cayeron los cajeros que aparecieron después?
    Compara el promedio observado con el de la misma cantidad de zonas elegidas al azar (Monte Carlo).
    """
    from .features import hex_de

    val = res.loc[mascara, ["h3", columna]].dropna()
    pct = val[columna].rank(pct=True)
    pct.index = val["h3"].values
    hexes = [h for h in hex_de(nuevos["lat"], nuevos["lon"]) if h in pct.index]
    if not hexes:
        return {"n": 0}
    obs = float(pct.loc[hexes].mean())
    rng = np.random.default_rng(SEMILLA)
    azar = rng.choice(pct.to_numpy(), size=(simulaciones, len(hexes)), replace=True).mean(axis=1)
    return {"n": len(hexes), "percentil_promedio": obs, "p_valor": float((azar >= obs).mean()),
            "percentiles": pct.loc[hexes].round(3).tolist()}


def coeficientes_glm(modelo, columnas: list[str]) -> pd.DataFrame:
    """Efecto multiplicativo de subir 1 desvío estándar (en escala log) cada variable, con lo demás fijo."""
    glm = modelo[-1] if hasattr(modelo, "__getitem__") else modelo
    coef = glm.coef_
    return (pd.DataFrame({"variable": columnas, "coeficiente": coef, "efecto": np.exp(coef)})
            .assign(nombre=lambda d: d["variable"].map(NOMBRES))
            .sort_values("coeficiente"))
