"""
horario.py
----------
Demanda por hora del día a partir de los molinetes del subte (SBASE, 2025).

Los molinetes cuentan ENTRADAS. Por eso el perfil horario de una estación dice qué tipo de
lugar es: si la gente entra sobre todo a la mañana, sale de su casa (estación de ORIGEN,
barrio residencial); si entra a la tarde, vuelve del trabajo o el estudio (estación de DESTINO).
Eso indica a qué hora y para qué pasa la gente por cada zona, y dónde ese flujo no tiene un
cajero Link cerca.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import features as f

HORAS = list(range(5, 24))  # el subte opera aprox. de 5 a 23 h
MANANA = range(6, 10)       # 6:00 a 9:59
TARDE = range(16, 20)       # 16:00 a 19:59
SEMILLA = 42


def perfiles_estaciones(base=f.DIR_BASE, anio: int = f.ANIO_MOLINETES) -> pd.DataFrame:
    """Tabla larga: estación (con coordenadas) x tipo de día x hora -> pasajeros promedio."""
    mol = f.molinetes(base, anio)
    est = f.estaciones_subte(base, anio)
    mol = mol.merge(est[["linea", "base", "estacion", "lat", "lon"]].rename(columns={"estacion": "estacion_osm"}),
                    left_on=["letra", "base"], right_on=["linea", "base"], how="inner", suffixes=("_mol", ""))
    largo = (mol.groupby(["linea", "estacion_osm", "lat", "lon", "tipo_dia", "hora"], as_index=False)["pasajeros_promedio_dia"].sum()
                .rename(columns={"estacion_osm": "estacion", "pasajeros_promedio_dia": "pasajeros"}))
    return largo


def curva_red(largo: pd.DataFrame) -> pd.DataFrame:
    """Pasajeros de toda la red por hora, día hábil y fin de semana."""
    return largo.pivot_table(index="hora", columns="tipo_dia", values="pasajeros", aggfunc="sum").reindex(HORAS).fillna(0)


def matriz_perfiles(largo: pd.DataFrame, tipo_dia: str = "habil") -> pd.DataFrame:
    """Estación x hora, como proporción de las entradas del día (la forma, no el volumen)."""
    m = (largo[largo["tipo_dia"] == tipo_dia]
         .pivot_table(index=["linea", "estacion"], columns="hora", values="pasajeros", aggfunc="sum")
         .reindex(columns=HORAS).fillna(0))
    m = m[m.sum(axis=1) > 0]  # estaciones cerradas todo el período
    return m.div(m.sum(axis=1), axis=0)


def agrupar_estaciones(perfiles: pd.DataFrame, ks=range(2, 6)) -> tuple[pd.Series, pd.DataFrame]:
    """K-Means sobre la forma de la curva horaria; k por silueta."""
    filas = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=30, random_state=SEMILLA).fit(perfiles)
        filas.append({"k": k, "silueta": silhouette_score(perfiles, km.labels_)})
    ks_df = pd.DataFrame(filas)
    k = int(ks_df.sort_values("silueta", ascending=False).iloc[0]["k"])
    etiquetas = KMeans(n_clusters=k, n_init=50, random_state=SEMILLA).fit_predict(perfiles)
    grupos = pd.Series(etiquetas, index=perfiles.index, name="grupo")
    # nombre según cuándo entra la gente
    nombres = {}
    for g in sorted(set(etiquetas)):
        p = perfiles[grupos == g].mean()
        manana, tarde = p[list(MANANA)].sum(), p[list(TARDE)].sum()
        if manana > 1.1 * tarde:
            nombres[g] = "Origen: la gente sale de su casa (pico a la mañana)"
        elif tarde > 1.1 * manana:
            nombres[g] = "Destino: la gente vuelve del trabajo (pico a la tarde)"
        else:
            nombres[g] = "Mixta: entradas repartidas en el día"
    repetidos = {n for n in nombres.values() if list(nombres.values()).count(n) > 1}
    for n in repetidos:
        for i, g in enumerate([g for g in nombres if nombres[g] == n]):
            nombres[g] = f"{n} ({i + 1})"
    return grupos.map(nombres).rename("perfil"), ks_df


def pasajeros_por_franja(largo: pd.DataFrame) -> pd.DataFrame:
    """Por estación: pasajeros de un día hábil en total, a la mañana (6-10 h) y a la tarde (16-20 h)."""
    h = largo[largo["tipo_dia"] == "habil"]
    tot = h.groupby(["linea", "estacion"])["pasajeros"].sum().rename("pax_habil")
    man = h[h["hora"].isin(MANANA)].groupby(["linea", "estacion"])["pasajeros"].sum().rename("pax_manana")
    tar = h[h["hora"].isin(TARDE)].groupby(["linea", "estacion"])["pasajeros"].sum().rename("pax_tarde")
    return pd.concat([tot, man, tar], axis=1).fillna(0).reset_index()
