"""
modelos.py
----------
Las tres técnicas del TPO, cada una responde una pregunta distinta:

1. Regresión (supervisado) -> ¿Cuántas terminales tienen las zonas con este perfil
   de demanda? La diferencia con lo que hay es la BRECHA (diagnóstico).
2. Clustering (no supervisado) -> ¿Qué tipos de zona hay? (segmentación para
   explicar dónde está la brecha).
3. Optimización (prescriptivo) -> Si Red Link promueve N cajeros nuevos, ¿dónde
   conviene ubicarlos para sumar la mayor cantidad de vecinos con un Link cerca?

Todas las funciones reciben la tabla de `features.construir()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor

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
_log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def candidatos() -> dict:
    """Modelos comparados, del más simple al más complejo."""
    return {
        "Línea de base (promedio)": DummyRegressor(strategy="mean"),
        "Poisson solo con población": make_pipeline(
            FunctionTransformer(lambda X: np.log1p(X[["poblacion_k1"]])), StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000)),
        "Regresión de Poisson (GLM)": _glm(),
        "Árbol de decisión (CART)": DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=SEMILLA),
        "Random Forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5,
                                               n_jobs=-1, random_state=SEMILLA),
        "Gradient Boosting (Poisson)": _boosting(),
        # Ensamble definido de antemano (no elegido mirando resultados): promedia un modelo
        # lineal interpretable y uno no lineal, los dos con pérdida de Poisson.
        "Ensamble (Poisson GLM + Boosting)": VotingRegressor([("glm", _glm()), ("gb", _boosting())]),
    }


def _glm():
    return make_pipeline(FunctionTransformer(lambda X: np.log1p(X.clip(lower=0))), StandardScaler(),
                         PoissonRegressor(alpha=1e-3, max_iter=2000))


def _boosting():
    return HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=SEMILLA)


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


def comparar(X, y, grupos, modelos: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Compara modelos con dos esquemas de validación cruzada:
    - espacial: se deja afuera una comuna entera por vez (GroupKFold por comuna), lo que
      simula predecir una zona que el modelo no vio;
    - aleatoria: KFold común, que filtra información entre vecinos y sobreestima el desempeño.
    Devuelve la tabla de métricas y las predicciones fuera de muestra (espaciales).
    """
    modelos = modelos or candidatos()
    esquemas = {"espacial (por comuna)": (GroupKFold(n_splits=5), grupos),
                "aleatoria": (KFold(n_splits=5, shuffle=True, random_state=SEMILLA), None)}
    filas, oof = [], {}
    for nombre, modelo in modelos.items():
        for esquema, (cv, g) in esquemas.items():
            pred = cross_val_predict(clone(modelo), X, y, groups=g, cv=cv)
            filas.append({"modelo": nombre, "validación": esquema, **metricas(y.values, pred)})
            if esquema.startswith("espacial"):
                oof[nombre] = pred
    return pd.DataFrame(filas), oof


def ensamble(oof: dict, nombres: list[str]) -> np.ndarray:
    return np.mean([oof[n] for n in nombres], axis=0)


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
        if p["dist_centro_km"] < 2.5 and p["actividad_osm_k1"] > 2 * med["actividad_osm_k1"]:
            nombres[c] = "Centro de negocios"
        elif p["pct_nbi_k1"] > 2 * med["pct_nbi_k1"]:
            nombres[c] = "Barrios vulnerables"
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
# 3. Optimización: problema de cobertura máxima (MCLP), algoritmo goloso
# --------------------------------------------------------------------------
def _xy(tabla: pd.DataFrame) -> np.ndarray:
    import geopandas as gpd

    p = gpd.GeoSeries(gpd.points_from_xy(tabla["lon"], tabla["lat"]), crs="EPSG:4326").to_crs(CRS_METRICO)
    return np.column_stack([p.x, p.y])


def demanda(tabla: pd.DataFrame, peso_pasajeros: float = 0.0) -> np.ndarray:
    """Personas a cubrir por hexágono: residentes + (opcional) pasajeros de subte de un día hábil."""
    return tabla["poblacion"].to_numpy(float) + peso_pasajeros * tabla["pax_subte_habil"].to_numpy(float)


def es_candidato(tabla: pd.DataFrame) -> np.ndarray:
    """Un cajero necesita un local: hexágonos con comercio o servicios (OSM o usos del suelo)."""
    comercio = tabla.get("parcelas_comercio", pd.Series(0, index=tabla.index)).fillna(0)
    return ((tabla["actividad_osm"] + comercio) > 0).to_numpy() & (tabla["frac_en_caba"] >= 0.5).to_numpy()


def cobertura_golosa(tabla: pd.DataFrame, n_nuevos: int, radio_m: float = 500, peso_pasajeros: float = 0.0,
                     cubiertos_iniciales: np.ndarray | None = None) -> pd.DataFrame:
    """
    Elige `n_nuevos` ubicaciones que maximizan las personas que pasan a tener un cajero Link
    a menos de `radio_m` metros (Maximal Covering Location Problem, Church & ReVelle 1974).
    El algoritmo goloso agrega en cada paso el sitio con mayor ganancia marginal; como la
    cobertura es submodular, garantiza al menos el 63 % (1 - 1/e) del óptimo.
    """
    xy = _xy(tabla)
    w = demanda(tabla, peso_pasajeros)
    cubierto = (tabla["dist_link_m"].to_numpy() <= radio_m) if cubiertos_iniciales is None else cubiertos_iniciales.copy()
    arbol = cKDTree(xy)
    cand = np.flatnonzero(es_candidato(tabla))
    vecinos = arbol.query_ball_point(xy[cand], r=radio_m)
    elegidos = []
    for paso in range(n_nuevos):
        ganancias = np.array([w[v][~cubierto[v]].sum() for v in vecinos])
        i = int(np.argmax(ganancias))
        if ganancias[i] <= 0:
            break
        nuevos = np.array(vecinos[i])[~cubierto[vecinos[i]]]
        cubierto[nuevos] = True
        h = cand[i]
        elegidos.append({
            "orden": paso + 1, "h3": tabla.iloc[h]["h3"], "lat": tabla.iloc[h]["lat"], "lon": tabla.iloc[h]["lon"],
            "barrio": tabla.iloc[h]["barrio"], "comuna": tabla.iloc[h]["comuna"],
            "personas_nuevas": float(ganancias[i]),
            "residentes_nuevos": float(tabla["poblacion"].to_numpy()[nuevos].sum()),
        })
    res = pd.DataFrame(elegidos)
    if not res.empty:
        res["personas_acumuladas"] = res["personas_nuevas"].cumsum()
    res.attrs["cubierto_final"] = cubierto
    return res


def resumen_cobertura(tabla: pd.DataFrame, radio_m: float = 500) -> pd.DataFrame:
    pob = tabla["poblacion"]
    total = pob.sum()
    filas = []
    for red, col in [("Link", "dist_link_m"), ("Banelco", "dist_banelco_m"), ("Cualquier red", "dist_cajero_m")]:
        cub = pob[tabla[col] <= radio_m].sum()
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
