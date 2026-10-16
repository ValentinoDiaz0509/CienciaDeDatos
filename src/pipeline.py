"""
pipeline.py
-----------
Corre todo el análisis de punta a punta y deja en data/processed/ lo que usan los
notebooks, la app y la presentación.

    hexagonos.parquet / resultados_hex.parquet   variables, brecha, tipo de zona y distancias
    metricas_modelos.csv                          regresión: 6 modelos + ensamble, validación espacial anidada y aleatoria
    hiperparametros.csv                           hiperparámetros elegidos en cada partición
    aportes_hex.parquet                           aporte de cada variable a la predicción de cada zona (SHAP)
    importancia_variables.csv, importancia_gini.csv, coeficientes_glm.csv
    curva_profundidad.csv, oob_vs_validacion.csv  sobreajuste del árbol y OOB vs. validación espacial
    metricas_clasificacion.csv, clasificacion_oof.parquet, matriz_confusion.csv   ¿le corresponde un cajero?
    barrios_prediccion.csv                        terminales que tiene y que le corresponden a cada barrio
    clusters_k.csv, clusters_perfil.csv           K-Means de zonas
    cobertura_redes.csv                           % de vecinos con un cajero de cada red cerca
    recomendaciones.csv                           secuencia golosa de 50 sitios (curva de cobertura)
    recomendaciones_exactas_20.csv                óptimo exacto para 20 cajeros (MILP)
    goloso_vs_exacto.csv                          brecha de optimalidad del goloso
    conjuntos_cobertura.npz                       quién cubre a quién (para el simulador de la app)
    subte_horario.csv, estaciones_perfil.csv      demanda por hora y tipo de estación
    estaciones_subte.csv, cajeros_nuevos_osm.csv, validacion_temporal.csv
    resumen.json                                  números clave

Uso:
    python -m src.pipeline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from . import features as f
from . import horario as ho
from . import ml
from . import modelos as m
from . import red as rp

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROC = RAIZ / "data" / "processed"
N_RECOMENDACIONES = 50
N_PRINCIPAL = 20
RADIOS_SIMULADOR = (300, 400, 500, 600, 700, 800)


def _guardar_conjuntos(ruta: Path, conjuntos: dict) -> None:
    """Guarda, para cada método y radio, los candidatos y a quién cubre cada uno (listas aplanadas)."""
    datos = {}
    for (metodo, radio), c in conjuntos.items():
        clave = f"{metodo}_{radio}"
        datos[f"{clave}_candidatos"] = c["candidatos"]
        datos[f"{clave}_largos"] = np.array([len(v) for v in c["vecinos"]])
        datos[f"{clave}_vecinos"] = np.concatenate(c["vecinos"]) if c["vecinos"] else np.array([], dtype=int)
    np.savez_compressed(ruta, **datos)


def _cajeros_para_igualar(res, conj, w, recs, cob, total) -> int | None:
    """Menor cantidad de cajeros nuevos con la que Link alcanza la cobertura actual de Banelco (óptimo exacto)."""
    falta = cob.loc["Banelco", "residentes_cubiertos"] - cob.loc["Link", "residentes_cubiertos"]
    alcanza = recs.loc[recs["personas_acumuladas"] >= falta, "orden"]
    if alcanza.empty:
        return None
    n = int(alcanza.iloc[0])  # el goloso da una cota superior; el exacto puede necesitar menos
    while n > 1 and m.resolver_exacto(res, n - 1, conj, w).attrs["objetivo"] >= falta:
        n -= 1
    return n


def correr(salida: Path = DIR_PROC) -> dict:
    t0 = time.time()
    salida.mkdir(parents=True, exist_ok=True)
    cajeros = f.cargar_cajeros()

    # 1. Variables por hexágono
    tabla = f.construir()
    con_red = rp.hay_red()
    if con_red:
        red = rp.RedPeatonal.desde_archivos()
        tabla = rp.agregar_distancias_red(tabla, cajeros, red)
    tabla.drop(columns="geometry").to_parquet(salida / "hexagonos.parquet", index=False)

    # 2. Machine learning (predictivo). Regresión: ¿cuántas terminales le corresponden a cada zona?
    X, y, grupos, mascara = m.preparar(tabla)
    metricas, oof, params, ajustes, _ = ml.comparar_regresion(X, y, grupos)
    metricas.to_csv(salida / "metricas_modelos.csv", index=False)
    pd.DataFrame([{"modelo": k, "partición": i + 1, **p} for k, v in params.items() for i, p in enumerate(v)]) \
        .to_csv(salida / "hiperparametros.csv", index=False)
    ganador = ml.elegir_regresion(metricas)
    res = m.calcular_brechas(tabla, mascara, oof[ganador])
    cuota_link = res.attrs["cuota_link"]

    aportes = ml.aportes_fuera_de_muestra(ajustes[ganador], X)
    aportes.assign(h3=res.loc[mascara, "h3"].to_numpy()).to_parquet(salida / "aportes_hex.parquet")
    (aportes.drop(columns="_base").abs().mean().rename("importancia").rename_axis("variable").reset_index()
       .assign(nombre=lambda d: d["variable"].map(ml.NOMBRES)).sort_values("importancia", ascending=False)
       .to_csv(salida / "importancia_variables.csv", index=False))
    params_rf = ml.params_mas_frecuentes(params["Random Forest"])
    (ml.importancia_gini(X, y, params_rf).rename("importancia").rename_axis("variable").reset_index()
       .assign(nombre=lambda d: d["variable"].map(ml.NOMBRES)).to_csv(salida / "importancia_gini.csv", index=False))
    glm = clone(m.candidatos()["Regresión de Poisson (GLM)"]).fit(X, y)
    m.coeficientes_glm(glm, list(X.columns)).to_csv(salida / "coeficientes_glm.csv", index=False)
    ml.curva_profundidad(X, y, grupos).to_csv(salida / "curva_profundidad.csv", index=False)

    # Clasificación: ¿a esta zona le corresponde al menos un cajero?
    Xc, yc = ml.preparar_clasificacion(tabla, X, mascara)
    met_c, oof_c, params_c, _ = ml.comparar_clasificacion(Xc, yc, grupos)
    met_c.to_csv(salida / "metricas_clasificacion.csv", index=False)
    pd.DataFrame([{"modelo": k, "partición": i + 1, **p} for k, v in params_c.items() for i, p in enumerate(v)]) \
        .to_csv(salida / "hiperparametros_clasificacion.csv", index=False)
    clasificador = ml.elegir_clasificador(met_c)
    umbral = ml.umbral_tasa_base(yc.to_numpy(), oof_c[clasificador])
    pd.DataFrame({"h3": res.loc[mascara, "h3"].to_numpy(), "real": yc.to_numpy(), **oof_c}) \
        .to_parquet(salida / "clasificacion_oof.parquet", index=False)
    ml.matriz(yc.to_numpy(), oof_c[clasificador], umbral).to_csv(salida / "matriz_confusion.csv")
    res.loc[mascara, "prob_cajero"] = oof_c[clasificador]
    res["deberia_tener_cajero"] = res["prob_cajero"] >= umbral
    oob = ml.oob_vs_validacion(X, y, grupos, Xc, yc, params_rf, ml.params_mas_frecuentes(params_c["Random Forest"]))
    oob.to_csv(salida / "oob_vs_validacion.csv", index=False)

    # 3. Clustering de zonas
    Z = m.matriz_cluster(res, mascara)
    ks = m.elegir_k(Z)
    ks.to_csv(salida / "clusters_k.csv", index=False)
    k = int(ks[ks["k"].between(3, 7)].sort_values("silueta", ascending=False).iloc[0]["k"])
    res.loc[mascara, "cluster"] = m.agrupar(Z, k)
    nombres = m.nombrar_clusters(res.loc[mascara])
    res["tipo_zona"] = res["cluster"].map(nombres)
    perfil = (res.loc[mascara].groupby("tipo_zona")
              .agg(hexagonos=("h3", "count"), residentes=("poblacion", "sum"),
                   pax_subte=("pax_subte_habil", "sum"), terminales=("term_total", "sum"),
                   terminales_link=("term_link", "sum"),
                   faltan_link=("brecha_link", lambda s: s.clip(lower=0).sum() / 7),
                   **{v: (v, "median") for v in m.VARIABLES_CLUSTER})
              .sort_values("residentes", ascending=False))
    perfil["terminales_cada_10mil"] = perfil["terminales"] / perfil["residentes"] * 1e4
    perfil["link_cada_10mil"] = perfil["terminales_link"] / perfil["residentes"] * 1e4
    perfil.to_csv(salida / "clusters_perfil.csv")

    barrios_pred = ml.por_barrio(res)
    barrios_pred["resumen"] = [ml.frase_barrio(r) for _, r in barrios_pred.iterrows()]
    barrios_pred.to_csv(salida / "barrios_prediccion.csv")

    # 4. Cobertura y optimización (caminando si hay red; si no, en línea recta)
    radio = f.RADIO_COBERTURA_M
    cob = m.resumen_cobertura(res, radio).assign(metodo="línea recta")
    if con_red:
        cob = pd.concat([cob, m.resumen_cobertura(res, radio, sufijo="_red").assign(metodo="caminando")])
    cob.to_csv(salida / "cobertura_redes.csv", index=False)
    metodo = "caminando" if con_red else "línea recta"
    if con_red:
        res["dist_link_m_usada"] = res["dist_link_m_red"]
    else:
        res["dist_link_m_usada"] = res["dist_link_m"]

    conjuntos = {}
    for r_ in RADIOS_SIMULADOR:
        conjuntos[("recta", r_)] = m.conjuntos_recta(res, r_)
        if con_red:
            conjuntos[("red", r_)] = rp.conjuntos_red(res, red, r_)
    _guardar_conjuntos(salida / "conjuntos_cobertura.npz", conjuntos)
    conj = conjuntos[("red" if con_red else "recta", radio)]
    w = m.demanda(res)

    recs = m.resolver_goloso(res, N_RECOMENDACIONES, conj, w)
    exacto = m.resolver_exacto(res, N_PRINCIPAL, conj, w)
    por_h3 = res.set_index("h3")
    for d in (recs, exacto):
        d["tipo_zona"] = d["h3"].map(por_h3["tipo_zona"])
        d["brecha_link"] = d["h3"].map(por_h3["brecha_link"])
    recs.to_csv(salida / "recomendaciones.csv", index=False)
    exacto.to_csv(salida / "recomendaciones_exactas_20.csv", index=False)
    comp = m.comparar_goloso_exacto(res, conj, w)
    comp.to_csv(salida / "goloso_vs_exacto.csv", index=False)
    # ¿Cambian los 20 sitios si los pasajeros de la mañana (6-10 h) cuentan igual que un vecino?
    con_manana = m.resolver_exacto(res, N_PRINCIPAL, conj, m.demanda(res, 1.0, "pax_subte_manana"))
    sitios_comunes_manana = len(set(exacto["h3"]) & set(con_manana["h3"]))

    # 5. Demanda por hora
    largo = ho.perfiles_estaciones()
    perfiles = ho.matriz_perfiles(largo)
    tipo_est, ks_est = ho.agrupar_estaciones(perfiles)
    est = f.estaciones_subte().merge(tipo_est.reset_index(), on=["linea", "estacion"], how="left")
    est["h3"] = f.hex_de(est["lat"], est["lon"])
    est = est.merge(res[["h3", "dist_link_m", "brecha_link", "tipo_zona"]
                        + (["dist_link_m_red"] if con_red else [])], on="h3", how="left")
    est["dist_link_usada_m"] = est["dist_link_m_red"] if con_red else est["dist_link_m"]
    est["link_cerca"] = est["dist_link_usada_m"] <= radio
    est.to_csv(salida / "estaciones_subte.csv", index=False)
    largo.merge(est[["linea", "estacion", "perfil", "link_cerca"]], on=["linea", "estacion"], how="left") \
         .to_csv(salida / "subte_horario.csv", index=False)
    ks_est.to_csv(salida / "estaciones_k.csv", index=False)

    # 6. Validación temporal con OpenStreetMap 2026
    osm = pd.read_csv(f.DIR_BASE / "osm-caba-puntos.csv")
    nuevos = m.cajeros_nuevos_osm(osm, cajeros)
    nuevos.to_csv(salida / "cajeros_nuevos_osm.csv", index=False)
    val = res.loc[mascara]
    etiqueta = val["h3"].isin({v for h in f.hex_de(nuevos["lat"], nuevos["lon"]) for v in f.h3.grid_disk(h, 1)}).astype(int)
    perc = m.percentil_cajeros_nuevos(res, mascara, nuevos)
    validacion = pd.DataFrame([
        {"puntaje": "Brecha total del modelo (2017)", "AUC": m.auc(val["brecha_total"].to_numpy(), etiqueta.to_numpy())},
        {"puntaje": "Demanda esperada del modelo", "AUC": m.auc(val["esperado_total_k1"].to_numpy(), etiqueta.to_numpy())},
        {"puntaje": "Solo residentes", "AUC": m.auc(val["poblacion_k1"].to_numpy(), etiqueta.to_numpy())},
        {"puntaje": "Probabilidad de cajero (clasificador)", "AUC": m.auc(val["prob_cajero"].to_numpy(), etiqueta.to_numpy())},
    ])
    validacion["hexagonos_cerca_de_cajero_nuevo"] = int(etiqueta.sum())
    validacion.to_csv(salida / "validacion_temporal.csv", index=False)

    res.drop(columns="geometry").to_parquet(salida / "resultados_hex.parquet", index=False)

    # 7. Resumen
    esp = metricas[metricas["validación"].str.startswith("espacial")].set_index("modelo")
    ale = metricas[metricas["validación"] == "aleatoria"].set_index("modelo")
    esp_c = met_c[met_c["validación"].str.startswith("espacial")].set_index("modelo")
    c_usada = cob[cob["metodo"] == metodo].set_index("red")
    c_recta = cob[cob["metodo"] == "línea recta"].set_index("red")
    total = res["poblacion"].sum()
    curva = ho.curva_red(largo)
    sin_link = est[~est["link_cerca"]]
    resumen = {
        "metodo_distancia": metodo,
        "hexagonos": int(len(res)),
        "hexagonos_modelados": int(mascara.sum()),
        "area_media_hex_km2": float(res["area_km2"].mean()),
        "residentes": float(total),
        "terminales": float(res["term_total"].sum()),
        "terminales_link": float(res["term_link"].sum()),
        "terminales_banelco": float(res["term_banelco"].sum()),
        "ubicaciones": int(len(cajeros)),
        "ubicaciones_link": int((cajeros["red"] == "LINK").sum()),
        "cuota_link": float(cuota_link),
        "modelo_elegido": ganador,
        "modelo_mejor_aleatoria": ale.sort_values("D² Poisson", ascending=False).index[0],
        "d2_espacial": float(esp.loc[ganador, "D² Poisson"]),
        "d2_espacial_promedio": float(esp.loc[ganador, "D² Poisson (promedio por partición)"]),
        "d2_espacial_desvio": float(esp.loc[ganador, "D² Poisson (desvío entre particiones)"]),
        "d2_aleatoria": float(ale.loc[ganador, "D² Poisson"]),
        "mae_espacial": float(esp.loc[ganador, "MAE"]),
        "r2_espacial": float(esp.loc[ganador, "R²"]),
        "clasificador_elegido": clasificador,
        "auc_clasificador": float(esp_c.loc[clasificador, "AUC"]),
        "auc_clasificador_promedio": float(esp_c.loc[clasificador, "AUC (promedio por partición)"]),
        "pr_auc_clasificador": float(esp_c.loc[clasificador, "PR-AUC"]),
        "precision_clasificador": float(esp_c.loc[clasificador, "Precisión"]),
        "recall_clasificador": float(esp_c.loc[clasificador, "Recall"]),
        "umbral_clasificador": float(umbral),
        "base_positivos_clasificador": float(yc.mean()),
        "zonas_sin_cajero_que_deberian": int((res["deberia_tener_cajero"] & (res["term_total"] == 0)).sum()),
        "residentes_zonas_sin_cajero_que_deberian": float(res.loc[res["deberia_tener_cajero"] & (res["term_total"] == 0), "poblacion"].sum()),
        "zonas_sin_link_que_deberian": int((res["deberia_tener_cajero"] & (res["term_link"] == 0)).sum()),
        "oob_r2_rf": float(oob.iloc[0]["OOB"]),
        "r2_espacial_rf": float(oob.iloc[0]["comunas no vistas"]),
        "oob_auc_rf": float(oob.iloc[1]["OOB"]),
        "auc_espacial_rf": float(oob.iloc[1]["comunas no vistas"]),
        "k_clusters": k,
        "silueta": float(ks.set_index("k").loc[k, "silueta"]),
        "cobertura_link": float(c_usada.loc["Link", "pct_residentes"]),
        "cobertura_banelco": float(c_usada.loc["Banelco", "pct_residentes"]),
        "cobertura_total": float(c_usada.loc["Cualquier red", "pct_residentes"]),
        "cobertura_link_recta": float(c_recta.loc["Link", "pct_residentes"]),
        "residentes_sin_link": float(total - c_usada.loc["Link", "residentes_cubiertos"]),
        "nuevos_10": float(recs["personas_nuevas"].head(10).sum()),
        "nuevos_20": float(recs["personas_nuevas"].head(20).sum()),
        "nuevos_50": float(recs["personas_nuevas"].sum()),
        "nuevos_20_exacto": float(exacto.attrs["objetivo"]),
        "cobertura_link_con_20": float((c_usada.loc["Link", "residentes_cubiertos"] + exacto.attrs["objetivo"]) / total),
        "cajeros_para_igualar_banelco": _cajeros_para_igualar(res, conj, w, recs, c_usada, total),
        "goloso_max_diferencia_pct": float(comp["diferencia_pct"].max()),
        "segundos_exacto_max": float(comp["seg_exacto"].max()),
        "pax_subte_habil": float(curva["habil"].sum()),
        "hora_pico_manana": int(curva.loc[5:11, "habil"].idxmax()),
        "hora_pico_tarde": int(curva.loc[14:20, "habil"].idxmax()),
        "estaciones": int(len(est)),
        "estaciones_sin_link": int(len(sin_link)),
        "pax_estaciones_sin_link": float(sin_link["pax_habil"].sum()),
        "sitios_comunes_con_pax_manana": int(sitios_comunes_manana),
        "estaciones_por_perfil": est["perfil"].value_counts().to_dict(),
        "cajeros_osm_2026": int((osm["categoria"] == "cajero").sum()),
        "cajeros_nuevos_osm": int(len(nuevos)),
        "auc_validacion": float(validacion.iloc[0]["AUC"]),
        "auc_validacion_clasificador": float(validacion.iloc[3]["AUC"]),
        "percentil_cajeros_nuevos": perc.get("percentil_promedio"),
        "p_valor_cajeros_nuevos": perc.get("p_valor"),
        "segundos_pipeline": round(time.time() - t0, 1),
    }
    if con_red:
        resumen["red_km"] = red.largo_total_km
        resumen["red_nodos"] = int(red.n)
        mask_d = np.isfinite(res["dist_link_m_red"]) & (res["dist_link_m"] > 50)
        resumen["factor_desvio_red"] = float(np.median(res.loc[mask_d, "dist_link_m_red"] / res.loc[mask_d, "dist_link_m"]))
        # Sensibilidad: conectar cada punto también a las cuadras a menos de 25 m extra (p. ej. en una esquina)
        red_tol = rp.RedPeatonal.desde_archivos(tolerancia_m=25)
        link = cajeros[cajeros["red"] == "LINK"]
        d_tol = red_tol.distancia_a_mas_cercano(res["lon"], res["lat"], link["lon"], link["lat"])
        resumen["cobertura_link_red_tolerancia_25m"] = float(res.loc[d_tol <= radio, "poblacion"].sum() / total)
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    for k_, v_ in correr().items():
        print(f"{k_:28s} {v_}")
