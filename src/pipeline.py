"""
pipeline.py
-----------
Corre todo el análisis de punta a punta y deja en data/processed/ los archivos que
usan los notebooks de presentación y la app:

    hexagonos.parquet            variables por hexágono (features)
    resultados_hex.parquet       + brechas, tipo de zona y cobertura
    metricas_modelos.csv         comparación de modelos (validación espacial y aleatoria)
    importancia_variables.csv    importancia por permutación del modelo final
    clusters_k.csv               inercia y silueta para elegir k
    clusters_perfil.csv          perfil de cada tipo de zona
    cobertura_redes.csv          % de residentes con un cajero de cada red a < 500 m
    recomendaciones.csv          50 ubicaciones sugeridas, en orden de prioridad
    estaciones_subte.csv         estaciones con pasajeros y coordenadas
    cajeros_nuevos_osm.csv       cajeros en OSM 2026 que no estaban en el listado 2017
    validacion_temporal.csv      ¿la brecha 2017 anticipa dónde aparecieron cajeros?
    resumen.json                 números clave para el informe y la presentación

Uso:
    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from . import features as f
from . import modelos as m

RAIZ = Path(__file__).resolve().parent.parent
DIR_PROC = RAIZ / "data" / "processed"
K_CLUSTERS = None  # None = se elige el k con mejor silueta entre 3 y 7
N_RECOMENDACIONES = 50


def elegir_modelo(metricas: pd.DataFrame) -> str:
    """El de mayor D² de Poisson en validación espacial (la métrica adecuada para conteos)."""
    esp = metricas[metricas["validación"].str.startswith("espacial")]
    return esp.sort_values("D² Poisson", ascending=False).iloc[0]["modelo"]


def correr(radios_path: Path | None = None, salida: Path = DIR_PROC) -> dict:
    salida.mkdir(parents=True, exist_ok=True)

    # 1. Variables por hexágono
    tabla = f.construir(radios_path=radios_path)
    tabla.drop(columns="geometry").to_parquet(salida / "hexagonos.parquet", index=False)

    # 2. Regresión: comparación de modelos y brecha fuera de muestra
    X, y, grupos, mascara = m.preparar(tabla)
    metricas, oof = m.comparar(X, y, grupos)
    metricas.to_csv(salida / "metricas_modelos.csv", index=False)
    ganador = elegir_modelo(metricas)
    res = m.calcular_brechas(tabla, mascara, oof[ganador])

    final = clone(m.candidatos()[ganador]).fit(X, y)
    imp = permutation_importance(final, X, y, n_repeats=10, random_state=m.SEMILLA,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
    importancia = (pd.DataFrame({"variable": X.columns, "importancia": imp.importances_mean,
                                 "desvio": imp.importances_std})
                   .assign(nombre=lambda d: d["variable"].map(m.NOMBRES))
                   .sort_values("importancia", ascending=False))
    importancia.to_csv(salida / "importancia_variables.csv", index=False)

    # 3. Clustering
    Z = m.matriz_cluster(res, mascara)
    ks = m.elegir_k(Z)
    ks.to_csv(salida / "clusters_k.csv", index=False)
    k = K_CLUSTERS or int(ks[ks["k"].between(3, 7)].sort_values("silueta", ascending=False).iloc[0]["k"])
    res.loc[mascara, "cluster"] = m.agrupar(Z, k)
    nombres = m.nombrar_clusters(res.loc[mascara])
    res["tipo_zona"] = res["cluster"].map(nombres)
    perfil = (res.loc[mascara].groupby("tipo_zona")
              .agg(hexagonos=("h3", "count"), residentes=("poblacion", "sum"),
                   pax_subte=("pax_subte_habil", "sum"), terminales=("term_total", "sum"),
                   terminales_link=("term_link", "sum"), brecha_link=("brecha_link", "sum"),
                   **{v: (v, "median") for v in m.VARIABLES_CLUSTER})
              .sort_values("residentes", ascending=False))
    perfil["terminales_cada_10mil"] = perfil["terminales"] / perfil["residentes"] * 1e4
    perfil["link_cada_10mil"] = perfil["terminales_link"] / perfil["residentes"] * 1e4
    perfil.to_csv(salida / "clusters_perfil.csv")

    # 4. Cobertura actual y optimización
    cobertura = m.resumen_cobertura(res, f.RADIO_COBERTURA_M)
    cobertura.to_csv(salida / "cobertura_redes.csv", index=False)
    recs = m.cobertura_golosa(res, N_RECOMENDACIONES, radio_m=f.RADIO_COBERTURA_M)
    h_a_fila = dict(zip(res["h3"], res.index))
    recs["tipo_zona"] = recs["h3"].map(lambda h: res.loc[h_a_fila[h], "tipo_zona"])
    recs["brecha_link"] = recs["h3"].map(lambda h: res.loc[h_a_fila[h], "brecha_link"])
    recs.to_csv(salida / "recomendaciones.csv", index=False)

    # 5. Validación temporal con OpenStreetMap 2026
    osm = pd.read_csv(f.DIR_BASE / "osm-caba-puntos.csv")
    nuevos = m.cajeros_nuevos_osm(osm, f.cargar_cajeros())
    nuevos.to_csv(salida / "cajeros_nuevos_osm.csv", index=False)
    res["cajero_nuevo_osm"] = res["h3"].isin(f.hex_de(nuevos["lat"], nuevos["lon"])).astype(int)
    val = res.loc[mascara]
    etiqueta = val["h3"].isin({v for h in f.hex_de(nuevos["lat"], nuevos["lon"]) for v in f.h3.grid_disk(h, 1)}).astype(int)
    validacion = pd.DataFrame([
        {"puntaje": "Brecha total del modelo (2017)", "AUC": m.auc(val["brecha_total"].to_numpy(), etiqueta.to_numpy())},
        {"puntaje": "Solo residentes", "AUC": m.auc(val["poblacion_k1"].to_numpy(), etiqueta.to_numpy())},
        {"puntaje": "Demanda esperada del modelo", "AUC": m.auc(val["esperado_total_k1"].to_numpy(), etiqueta.to_numpy())},
    ])
    validacion["hexagonos_con_cajero_nuevo"] = int(etiqueta.sum())
    validacion.to_csv(salida / "validacion_temporal.csv", index=False)

    # 6. Estaciones y resultados por hexágono
    f.estaciones_subte().to_csv(salida / "estaciones_subte.csv", index=False)
    res.drop(columns="geometry").to_parquet(salida / "resultados_hex.parquet", index=False)

    esp = metricas[metricas["validación"].str.startswith("espacial")].set_index("modelo")
    ale = metricas[metricas["validación"] == "aleatoria"].set_index("modelo")
    cub = cobertura.set_index("red")
    resumen = {
        "hexagonos": int(len(res)),
        "hexagonos_modelados": int(mascara.sum()),
        "area_media_hex_km2": float(res["area_km2"].mean()),
        "residentes": float(res["poblacion"].sum()),
        "terminales": float(res["term_total"].sum()),
        "terminales_link": float(res["term_link"].sum()),
        "terminales_banelco": float(res["term_banelco"].sum()),
        "ubicaciones": int(res["ubic_link"].sum() + res["ubic_banelco"].sum()),
        "cuota_link": float(res.attrs.get("cuota_link", np.nan)),
        "modelo_elegido": ganador,
        "d2_espacial": float(esp.loc[ganador, "D² Poisson"]),
        "d2_aleatoria": float(ale.loc[ganador, "D² Poisson"]),
        "mae_espacial": float(esp.loc[ganador, "MAE"]),
        "r2_espacial": float(esp.loc[ganador, "R²"]),
        "k_clusters": k,
        "cobertura_link": float(cub.loc["Link", "pct_residentes"]),
        "cobertura_banelco": float(cub.loc["Banelco", "pct_residentes"]),
        "cobertura_total": float(cub.loc["Cualquier red", "pct_residentes"]),
        "residentes_sin_link": float(res["poblacion"].sum() - cub.loc["Link", "residentes_cubiertos"]),
        "nuevos_10": float(recs["personas_nuevas"].head(10).sum()),
        "nuevos_20": float(recs["personas_nuevas"].head(20).sum()),
        "nuevos_50": float(recs["personas_nuevas"].sum()),
        "cajeros_osm_2026": int((osm["categoria"] == "cajero").sum()),
        "cajeros_nuevos_osm": int(len(nuevos)),
        "auc_validacion": float(validacion.iloc[0]["AUC"]),
    }
    (salida / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    for k, v in correr().items():
        print(f"{k:28s} {v}")
