"""
tecnicas.py
-----------
Técnicas de la clase 11 (técnicas avanzadas de minería de datos) aplicadas a la pregunta del TPO:

1. Validación cruzada ESTRATIFICADA por grupos (StratifiedGroupKFold). ¿Cambia el clasificador si cada partición
   mantiene la proporción de zonas con cajero, sin dejar de separar las comunas?
2. PCA (reducción de dimensionalidad). ¿Cuánta información comparten las 14 variables de demanda? ¿Los tipos de
   zona del K-Means dependen de cómo se miden las variables?
3. K-Means como ubicador de cajeros (el ejemplo de la clase) frente a la optimización exacta.
4. DBSCAN (clustering por densidad). ¿Dónde se juntan los vecinos que no tienen un Link a distancia caminable?

Las cuatro parten de lo que ya calculó el pipeline (data/processed). Uso: python -m src.tecnicas
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from . import ml
from . import modelos as m

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROC = RAIZ / "data" / "processed"
SEMILLA = m.SEMILLA
RADIO_M = 500
# conteos muy asimétricos: se pasan a logaritmo antes de escalar (como en la regresión de Poisson y el K-Means)
CONTEOS = ["poblacion_k1", "actividad_osm_k1", "pax_subte_habil_k1", "parcelas_comercio_k1", "parcelas_oficinas_k1",
           "parcelas_residencial_k1"]
# DBSCAN: dos hexágonos son vecinos si sus centros están a menos de 320 m (los centros de hexágonos contiguos están a
# unos 300 m); un bolsón necesita al menos 8.000 vecinos sin Link entre un hexágono y los de al lado. Como referencia,
# cada uno de los 20 cajeros óptimos suma en promedio unos 13 mil.
EPS_M = 320
MINIMO_BOLSON = 8_000


def cargar(proc: Path = DIR_PROC) -> tuple[pd.DataFrame, dict]:
    """Resultados por hexágono y la estructura de cobertura caminando a 500 m (o en línea recta si no hay red)."""
    res = pd.read_parquet(proc / "resultados_hex.parquet")
    z = np.load(proc / "conjuntos_cobertura.npz")
    metodo = "red" if f"red_{RADIO_M}_candidatos" in z.files else "recta"
    largos = z[f"{metodo}_{RADIO_M}_largos"]
    conj = {"candidatos": z[f"{metodo}_{RADIO_M}_candidatos"],
            "vecinos": np.split(z[f"{metodo}_{RADIO_M}_vecinos"], np.cumsum(largos)[:-1]), "metodo": metodo}
    col = "dist_link_m_red" if metodo == "red" else "dist_link_m"
    conj["cubierto"] = res[col].to_numpy() <= RADIO_M
    conj["columna_distancia"] = col
    return res, conj


# --------------------------------------------------------------------------
# 1. Validación cruzada estratificada por grupos
# --------------------------------------------------------------------------
def tasas_por_particion(yc: pd.Series, grupos: pd.Series, Xc: pd.DataFrame) -> pd.DataFrame:
    """Proporción de zonas con cajero en cada partición de prueba: por comuna sola y estratificada."""
    filas = []
    for nombre, cv in [("Por comuna (GroupKFold)", GroupKFold(n_splits=ml.PLIEGUES)),
                       ("Por comuna y estratificada (StratifiedGroupKFold)",
                        StratifiedGroupKFold(n_splits=ml.PLIEGUES, shuffle=True, random_state=SEMILLA))]:
        for k, (_, te) in enumerate(cv.split(Xc, yc, grupos), start=1):
            filas.append({"validación": nombre, "partición": k, "zonas": len(te), "con cajero (%)": 100 * yc.iloc[te].mean()})
    return pd.DataFrame(filas)


def validacion_estratificada(res: pd.DataFrame) -> pd.DataFrame:
    """
    Repite la comparación de clasificadores con StratifiedGroupKFold afuera y adentro (para los hiperparámetros):
    cada partición deja comunas enteras afuera y mantiene la proporción de zonas con cajero (22 %).
    """
    X, _, grupos, mascara = m.preparar(res)
    Xc, yc = ml.preparar_clasificacion(res, X, mascara)
    externos = StratifiedGroupKFold(n_splits=ml.PLIEGUES, shuffle=True, random_state=SEMILLA)
    filas = []
    for nombre, (est, grilla) in ml.especificaciones_clasificacion().items():
        aucs = []
        for tr, te in externos.split(Xc, yc, grupos):
            internos = StratifiedGroupKFold(n_splits=ml.PLIEGUES_INTERNOS, shuffle=True, random_state=SEMILLA)
            gs = GridSearchCV(clone(est), grilla, cv=internos, scoring="roc_auc", n_jobs=-1)
            gs.fit(Xc.iloc[tr], yc.iloc[tr], groups=grupos.iloc[tr])
            aucs.append(roc_auc_score(yc.iloc[te], gs.predict_proba(Xc.iloc[te])[:, 1]))
        filas.append({"modelo": nombre, "AUC estratificada": float(np.mean(aucs)),
                      "desvío estratificada": float(np.std(aucs, ddof=1))})
    est = pd.DataFrame(filas)
    previo = pd.read_csv(DIR_PROC / "metricas_clasificacion.csv")
    previo = previo[previo["validación"].str.startswith("espacial")].set_index("modelo")
    est["AUC por comuna"] = est["modelo"].map(previo["AUC (promedio por partición)"])
    est["desvío por comuna"] = est["modelo"].map(previo["AUC (desvío entre particiones)"])
    return est[["modelo", "AUC por comuna", "desvío por comuna", "AUC estratificada", "desvío estratificada"]]


# --------------------------------------------------------------------------
# 2. PCA de las variables de demanda
# --------------------------------------------------------------------------
def matriz_pca(res: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.Series]:
    """Las 14 variables del modelo, con los conteos en logaritmo y todo estandarizado (media 0, desvío 1)."""
    X, _, _, mascara = m.preparar(res)
    Xl = X.copy()
    for c in CONTEOS:
        if c in Xl:
            Xl[c] = np.log1p(Xl[c].clip(lower=0))
    return StandardScaler().fit_transform(Xl), list(X.columns), mascara


def pca(res: pd.DataFrame, n_cargas: int = 4) -> dict:
    Z, columnas, mascara = matriz_pca(res)
    p = PCA(random_state=SEMILLA).fit(Z)
    var = pd.DataFrame({"componente": [f"CP{i + 1}" for i in range(len(p.explained_variance_ratio_))],
                        "varianza (%)": 100 * p.explained_variance_ratio_,
                        "acumulada (%)": 100 * np.cumsum(p.explained_variance_ratio_),
                        "autovalor": p.explained_variance_})
    cargas = pd.DataFrame(p.components_[:n_cargas].T, index=columnas,
                          columns=[f"CP{i + 1}" for i in range(n_cargas)]).rename_axis("variable")
    cargas.insert(0, "nombre", [m.NOMBRES.get(c, c) for c in columnas])
    puntajes = pd.DataFrame(p.transform(Z)[:, :n_cargas], columns=[f"CP{i + 1}" for i in range(n_cargas)])
    puntajes.insert(0, "h3", res.loc[mascara, "h3"].to_numpy())
    puntajes["tipo_zona"] = res.loc[mascara, "tipo_zona"].to_numpy()
    acum = var["acumulada (%)"].to_numpy()
    return {"varianza": var, "cargas": cargas.reset_index(), "puntajes": puntajes,
            "kaiser": int((p.explained_variance_ > 1).sum()), "n_80": int(np.argmax(acum >= 80) + 1),
            "n_95": int(np.argmax(acum >= 95) + 1)}


def zonas_con_pca(res: pd.DataFrame, ns=(2, 3, 4)) -> pd.DataFrame:
    """
    El K-Means de tipos de zona repetido sobre los primeros componentes principales de sus 7 variables. El índice de
    Rand ajustado (ARI) compara las dos agrupaciones: 1 es la misma, 0 es lo que daría el azar.
    """
    mascara = res["esperado_total_k1"].notna()
    Z = m.matriz_cluster(res, mascara)
    original = res.loc[mascara, "cluster"].to_numpy()
    k = int(pd.Series(original).nunique())
    p = PCA(random_state=SEMILLA).fit(Z)
    filas = []
    for n in ns:
        S = p.transform(Z)[:, :n]
        lab = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit_predict(S)
        filas.append({"componentes": n, "varianza retenida (%)": 100 * p.explained_variance_ratio_[:n].sum(),
                      "ARI contra los tipos de zona": adjusted_rand_score(original, lab),
                      "silueta": silhouette_score(S, lab, random_state=SEMILLA)})
    filas.append({"componentes": Z.shape[1], "varianza retenida (%)": 100.0, "ARI contra los tipos de zona": 1.0,
                  "silueta": silhouette_score(Z, original, random_state=SEMILLA)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# 3. K-Means como ubicador de cajeros frente a la optimización exacta
# --------------------------------------------------------------------------
def _distancia_media(res: pd.DataFrame, conj: dict, w: np.ndarray, sitios: pd.DataFrame, cajeros: pd.DataFrame,
                     red=None) -> float:
    """
    Distancia media (ponderada por personas) de la gente que hoy no tiene un Link cerca al Link más cercano, contando
    los cajeros nuevos. Caminando si hay red; si no, en línea recta.
    """
    pend = (~conj["cubierto"]) & (w > 0)
    link = cajeros[cajeros["red"] == "LINK"][["lon", "lat"]]
    fuentes = pd.concat([link, sitios[["lon", "lat"]]], ignore_index=True) if len(sitios) else link
    if red is not None:
        d = red.distancia_a_mas_cercano(res["lon"].to_numpy()[pend], res["lat"].to_numpy()[pend],
                                        fuentes["lon"].to_numpy(), fuentes["lat"].to_numpy())
    else:
        xy = m._xy(res)
        f_xy = m._xy(fuentes)
        d = cKDTree(f_xy).query(xy[pend])[0]
    return float(np.average(d, weights=w[pend]))


def kmeans_vs_optimo(res: pd.DataFrame, conj: dict, cajeros: pd.DataFrame, red=None,
                     ns=(5, 10, 15, 20, 25, 30, 35, 40, 50)) -> tuple[pd.DataFrame, dict]:
    """Vecinos que suma y distancia media que deja cada método, para distintas cantidades de cajeros nuevos."""
    w = m.demanda(res)
    filas, sitios = [], {}
    sin_cambios = _distancia_media(res, conj, w, pd.DataFrame(columns=["lon", "lat"]), cajeros, red)
    for n in ns:
        ex = m.resolver_exacto(res, n, conj, w)
        km = m.resolver_kmeans(res, n, conj, w)
        go = m.resolver_goloso(res, n, conj, w)
        filas.append({"cajeros": n, "óptimo exacto": ex.attrs["objetivo"], "goloso": go.attrs["objetivo"],
                      "K-Means": km.attrs["objetivo"], "K-Means / óptimo (%)": 100 * km.attrs["objetivo"] / ex.attrs["objetivo"],
                      "distancia media hoy (m)": sin_cambios,
                      "distancia media con el óptimo (m)": _distancia_media(res, conj, w, ex, cajeros, red),
                      "distancia media con K-Means (m)": _distancia_media(res, conj, w, km, cajeros, red),
                      "sitios en común": len(set(ex["h3"]) & set(km["h3"]))})
        sitios[n] = {"exacto": ex, "kmeans": km}
    return pd.DataFrame(filas), sitios


# --------------------------------------------------------------------------
# 4. DBSCAN: bolsones de vecinos sin un Link cerca
# --------------------------------------------------------------------------
def _lista(nombres: list[str]) -> str:
    """'Belgrano, Núñez y Palermo'."""
    return nombres[0] if len(nombres) == 1 else ", ".join(nombres[:-1]) + " y " + nombres[-1]


def _dbscan(xy: np.ndarray, peso: np.ndarray, eps_m: float, minimo: float) -> np.ndarray:
    return DBSCAN(eps=eps_m, min_samples=minimo).fit(xy, sample_weight=peso).labels_


def bolsones(res: pd.DataFrame, conj: dict, sitios_optimos: pd.DataFrame | None = None, eps_m: float = EPS_M,
             minimo: float = MINIMO_BOLSON) -> tuple[pd.Series, pd.DataFrame]:
    """
    DBSCAN sobre los hexágonos con vecinos sin un Link a distancia caminable, pesados por su gente. Devuelve la etiqueta
    de cada hexágono (-1: disperso; vacío: con Link cerca o sin vecinos) y un resumen por bolsón, del más grande al más
    chico, con los barrios donde está y si le toca alguno de los cajeros óptimos.
    """
    w = m.demanda(res)
    pend = (~conj["cubierto"]) & (w > 0)
    xy = m._xy(res)[pend]
    lab = _dbscan(xy, w[pend], eps_m, minimo)
    # numerar los bolsones del más poblado al menos poblado
    tam = pd.Series(w[pend]).groupby(lab).sum().drop(-1, errors="ignore").sort_values(ascending=False)
    nuevo = {viejo: i + 1 for i, viejo in enumerate(tam.index)}
    lab = np.array([nuevo.get(v, -1) for v in lab])
    etiqueta = pd.Series(np.nan, index=res.index)
    etiqueta[pend] = lab
    filas = []
    sub = res[pend].assign(bolson=lab, gente=w[pend])
    con_sitio = set()
    if sitios_optimos is not None and len(sitios_optimos):
        arbol = cKDTree(xy)
        for _, s in sitios_optimos.iterrows():
            sx = m._xy(pd.DataFrame({"lon": [s["lon"]], "lat": [s["lat"]]}))[0]
            cerca = [lab[i] for i in arbol.query_ball_point(sx, r=eps_m) if lab[i] > 0]
            if cerca:
                con_sitio.add(pd.Series(cerca).mode().iloc[0])
    for b, g in sub[sub["bolson"] > 0].groupby("bolson"):
        barrios = g.groupby("barrio")["gente"].sum().sort_values(ascending=False)
        filas.append({"bolsón": int(b), "vecinos sin Link": float(g["gente"].sum()), "hexágonos": len(g),
                      "barrios": _lista(list(barrios.index[:3])),
                      "lat": float(np.average(g["lat"], weights=g["gente"])),
                      "lon": float(np.average(g["lon"], weights=g["gente"])),
                      "le toca un cajero óptimo": int(b) in con_sitio})
    return etiqueta.rename("bolson"), pd.DataFrame(filas)


def sensibilidad_bolsones(res: pd.DataFrame, conj: dict, minimos=(6_000, 8_000, 10_000), eps_m: float = EPS_M) -> pd.DataFrame:
    w = m.demanda(res)
    pend = (~conj["cubierto"]) & (w > 0)
    xy = m._xy(res)[pend]
    filas = []
    for minimo in minimos:
        lab = _dbscan(xy, w[pend], eps_m, minimo)
        tam = pd.Series(w[pend]).groupby(lab).sum().drop(-1, errors="ignore")
        filas.append({"mínimo de vecinos": minimo, "bolsones": int(len(tam)),
                      "vecinos sin Link en bolsones (%)": 100 * tam.sum() / w[pend].sum(),
                      "el más grande": float(tam.max()) if len(tam) else 0.0})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------
# Todo junto: python -m src.tecnicas
# --------------------------------------------------------------------------
def resultados(salida: Path = DIR_PROC, estratificada: bool = True) -> dict:
    """Calcula las cuatro técnicas y guarda las tablas que usan el notebook 06, la app y la presentación."""
    from . import features as f
    from . import red as rp

    res, conj = cargar(salida)
    cajeros = f.cargar_cajeros()
    red = rp.RedPeatonal.desde_archivos() if conj["metodo"] == "red" and rp.hay_red() else None
    X, _, grupos, mascara = m.preparar(res)
    Xc, yc = ml.preparar_clasificacion(res, X, mascara)

    out = {"particiones": tasas_por_particion(yc, grupos, Xc)}
    if estratificada:
        out["estratificada"] = validacion_estratificada(res)
    elif (salida / "tecnicas_estratificada.csv").exists():
        out["estratificada"] = pd.read_csv(salida / "tecnicas_estratificada.csv")
    out["pca"] = pca(res)
    out["zonas_pca"] = zonas_con_pca(res)
    out["kmeans"], sitios = kmeans_vs_optimo(res, conj, cajeros, red)
    out["sitios_20"] = sitios[20]
    out["etiquetas"], out["bolsones"] = bolsones(res, conj, sitios[20]["exacto"])
    out["sensibilidad"] = sensibilidad_bolsones(res, conj)

    out["particiones"].to_csv(salida / "tecnicas_particiones.csv", index=False)
    if "estratificada" in out:
        out["estratificada"].to_csv(salida / "tecnicas_estratificada.csv", index=False)
    out["pca"]["varianza"].to_csv(salida / "tecnicas_pca_varianza.csv", index=False)
    out["pca"]["cargas"].to_csv(salida / "tecnicas_pca_cargas.csv", index=False)
    out["pca"]["puntajes"].to_parquet(salida / "tecnicas_pca_puntajes.parquet", index=False)
    out["zonas_pca"].to_csv(salida / "tecnicas_zonas_pca.csv", index=False)
    out["kmeans"].to_csv(salida / "tecnicas_kmeans_vs_optimo.csv", index=False)
    pd.concat([sitios[20]["exacto"].assign(metodo="óptimo exacto"), sitios[20]["kmeans"].assign(metodo="K-Means")]) \
        .to_csv(salida / "tecnicas_sitios_20.csv", index=False)
    pd.DataFrame({"h3": res["h3"], "bolson": out["etiquetas"]}).dropna().to_parquet(salida / "tecnicas_bolsones_hex.parquet",
                                                                                  index=False)
    out["bolsones"].to_csv(salida / "tecnicas_bolsones.csv", index=False)
    out["sensibilidad"].to_csv(salida / "tecnicas_bolsones_sensibilidad.csv", index=False)

    k20 = out["kmeans"].set_index("cajeros").loc[20]
    w = m.demanda(res)
    pend = (~conj["cubierto"]) & (w > 0)
    b = out["bolsones"]
    zp = out["zonas_pca"].set_index("componentes")
    v = out["pca"]["varianza"]
    resumen = {
        "distancia": "caminando" if red is not None else "en línea recta",
        "kmeans_20": float(k20["K-Means"]), "exacto_20": float(k20["óptimo exacto"]),
        "kmeans_pct_optimo_20": float(k20["K-Means / óptimo (%)"]),
        "kmeans_pct_optimo_min": float(out["kmeans"]["K-Means / óptimo (%)"].min()),
        "kmeans_pct_optimo_max": float(out["kmeans"]["K-Means / óptimo (%)"].max()),
        "dist_media_hoy": float(k20["distancia media hoy (m)"]),
        "dist_media_exacto_20": float(k20["distancia media con el óptimo (m)"]),
        "dist_media_kmeans_20": float(k20["distancia media con K-Means (m)"]),
        "sitios_en_comun_20": int(k20["sitios en común"]),
        "bolsones": int(len(b)), "vecinos_sin_link": float(w[pend].sum()),
        "vecinos_en_bolsones": float(b["vecinos sin Link"].sum()),
        "pct_en_bolsones": float(100 * b["vecinos sin Link"].sum() / w[pend].sum()),
        "bolsones_con_cajero_optimo": int(b["le toca un cajero óptimo"].sum()),
        "bolson_mayor": b.iloc[0][["vecinos sin Link", "barrios"]].to_dict() if len(b) else {},
        "eps_m": EPS_M, "minimo_bolson": MINIMO_BOLSON,
        "pca_varianza_2": float(v["acumulada (%)"].iloc[1]), "pca_varianza_4": float(v["acumulada (%)"].iloc[3]),
        "pca_kaiser": out["pca"]["kaiser"], "pca_n_80": out["pca"]["n_80"], "pca_n_95": out["pca"]["n_95"],
        "ari_4_componentes": float(zp.loc[4, "ARI contra los tipos de zona"]),
        "varianza_4_componentes_zonas": float(zp.loc[4, "varianza retenida (%)"]),
    }
    if "estratificada" in out:
        e = out["estratificada"].set_index("modelo")
        mejor = e["AUC estratificada"].idxmax()
        resumen.update(mejor_estratificada=mejor, auc_estratificada_mejor=float(e.loc[mejor, "AUC estratificada"]),
                       auc_por_comuna_rf=float(e.loc["Random Forest", "AUC por comuna"]),
                       auc_estratificada_rf=float(e.loc["Random Forest", "AUC estratificada"]))
    p = out["particiones"]
    for nombre, clave in [("Por comuna (GroupKFold)", "tasa_por_comuna"),
                          ("Por comuna y estratificada (StratifiedGroupKFold)", "tasa_estratificada")]:
        t = p[p["validación"] == nombre]["con cajero (%)"]
        resumen[clave + "_min"], resumen[clave + "_max"] = float(t.min()), float(t.max())
    (salida / "tecnicas_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    out["resumen"] = resumen
    return out


if __name__ == "__main__":
    print(json.dumps(resultados()["resumen"], ensure_ascii=False, indent=2))
