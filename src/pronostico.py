"""
pronostico.py
-------------
Análisis predictivo en el tiempo: cuánta gente va a entrar al subte cada día en los próximos
meses, a partir de la serie diaria de los molinetes (BA Data; los años que publique el catálogo,
hoy de enero de 2025 a agosto de 2026).

Para Red Link: anticipar la demanda de paso para planificar la carga de efectivo y el
mantenimiento de los cajeros cerca de las estaciones (Gerencia Técnica).

Modelos (clases 6 y 8): Random Forest y XGBoost sobre variables de calendario (día de la
semana, mes, día del año, feriados y feriados puente), comparados con una referencia ingenua
("el mismo día de la semana del año anterior"). Validación HACIA ADELANTE: se entrena con el
pasado y se evalúa en meses posteriores que el modelo no vio; nunca se usan fechas futuras
para entrenar.

Los días con muy pocos pasajeros para su día de la semana (paros, cortes o datos faltantes), con
demasiados (registros repetidos) o con la fecha mal cargada en el archivo de origen se marcan como
anomalías: no se usan para entrenar ni para medir el error, porque no son predecibles con el
calendario.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import _ALIAS_ESTACIONES, DIR_BASE, RAIZ, normalizar

SEMILLA = 42
ARCHIVO_DIARIO = "subte-diario-estacion.parquet"
ARCHIVO_HORA = "subte-hora-red.parquet"
VARIABLES = ["dia_semana", "fin_de_semana", "mes", "dia_mes", "dia_anio_sin", "dia_anio_cos", "semana_mes",
             "feriado", "puente", "no_laborable", "pegado_feriado", "vacaciones_invierno", "anio"]
NOMBRES = {
    "dia_semana": "Día de la semana", "fin_de_semana": "Fin de semana", "mes": "Mes", "dia_mes": "Día del mes",
    "dia_anio_sin": "Época del año (seno)", "dia_anio_cos": "Época del año (coseno)", "semana_mes": "Semana del mes",
    "feriado": "Feriado", "puente": "Feriado puente", "no_laborable": "Día no laborable",
    "pegado_feriado": "Día pegado a un feriado", "vacaciones_invierno": "Vacaciones de invierno", "anio": "Año",
}
# Receso escolar de invierno en CABA (calendario escolar de cada año)
VACACIONES_INVIERNO = {2025: ("2025-07-21", "2025-08-01"), 2026: ("2026-07-20", "2026-07-31")}
# Días que en los datos se comportan como feriado (o que son feriado solo en la Ciudad) y que el paquete `holidays`
# no trae. El 9/11/2026 (visita del Papa León XIV) es feriado nacional (las versiones nuevas del paquete ya lo traen) y el 10
# lo es en CABA.
FERIADOS_EXTRA = {"2025-10-10": "Día del Respeto a la Diversidad Cultural (trasladado al viernes)",
                  "2026-11-09": "Visita del Papa León XIV (feriado nacional)",
                  "2026-11-10": "Visita del Papa León XIV (feriado en la Ciudad de Buenos Aires)"}


def hay_serie(base: Path = DIR_BASE) -> bool:
    return (base / ARCHIVO_DIARIO).exists()


def cargar(base: Path = DIR_BASE) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    diario = pd.read_parquet(base / ARCHIVO_DIARIO)
    hora = pd.read_parquet(base / ARCHIVO_HORA) if (base / ARCHIVO_HORA).exists() else None
    return diario, hora


# --------------------------------------------------------------------------
# Serie y calendario
# --------------------------------------------------------------------------
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


def nombre_mes(fecha) -> str:
    return f"{MESES[fecha.month - 1]} de {fecha.year}"


def _sin_pruebas(diario: pd.DataFrame) -> pd.DataFrame:
    return diario[~diario["linea"].isin(["PRUEBA"]) & ~diario["estacion"].isin(["#N/D", "Prueba", "CochePM"])]


def serie_red(diario: pd.DataFrame) -> pd.Series:
    """Pasajeros por día en toda la red, con todas las fechas (las que faltan quedan vacías)."""
    s = _sin_pruebas(diario).groupby("fecha")["pasajeros"].sum().astype(float)
    s.index = pd.to_datetime(s.index)
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).rename("pasajeros")


def _relacion_con_tipico(s: pd.Series, excluir: np.ndarray) -> pd.Series:
    """
    Pasajeros del día sobre la mediana del mismo día de la semana en las 8 semanas alrededor. Para la mediana no se
    usan los días excluidos (feriados) ni los casi vacíos (menos del 30 % de un día igual).
    """
    dia = s.index.dayofweek
    base = s.where(~excluir)
    mediana_dia = base.groupby(dia).transform("median")
    base = base.where(base > 0.3 * mediana_dia)
    tipico = pd.Series(index=s.index, dtype=float)
    for d in range(7):
        sub = base[dia == d]
        tipico.loc[sub.index] = sub.rolling(9, center=True, min_periods=3).median()
    return s / tipico.fillna(mediana_dia)


def fechas_cruzadas(s: pd.Series, desvio: float = 0.15, minimo: int = 4) -> pd.DatetimeIndex:
    """
    Busca meses en los que una parte de los registros trae la fecha como mes/día y se leyó como día/mes. Si eso le pasa
    al mes M, los días 1 a 12 de M quedan con pasajeros de menos y el día M de los otros meses, con pasajeros de más
    ("08/04/2025" es el 4 de agosto, pero leído como día/mes cae el 8 de abril). Se marca el mes cuando la mediana de los
    días 1 a 12 queda más de un 15 % abajo de lo típico y la del día M de los otros meses, más de un 15 % arriba. El día
    M de M no cambia: se lee igual de las dos formas. En los molinetes de 2025 lo encuentra en agosto (22 días).
    """
    fer = feriados_ar(s.index)["tipo"].reindex(s.index).notna().to_numpy()
    r = _relacion_con_tipico(s, fer).where(~fer)
    fechas = []
    for anio in sorted(set(s.index.year)):
        for m in range(1, 13):
            bajos = pd.DatetimeIndex([pd.Timestamp(anio, m, d) for d in range(1, 13) if d != m])
            altos = pd.DatetimeIndex([pd.Timestamp(anio, d, m) for d in range(1, 13) if d != m])
            rb, ra = r.reindex(bajos).dropna(), r.reindex(altos).dropna()
            if len(rb) >= minimo and len(ra) >= minimo and rb.median() < 1 - desvio and ra.median() > 1 + desvio:
                fechas += list(bajos) + list(altos)
    return pd.DatetimeIndex(sorted(set(fechas)))


def marcar_anomalias(s: pd.Series, bajo: float = 0.5, alto: float = 1.25) -> pd.Series:
    """
    Verdadero en los días sin dato, en los de fecha mal cargada (fechas_cruzadas) y en los que se apartan mucho de un
    día igual: menos de la mitad (paros, cortes, archivos incompletos) o más de 1,25 veces (registros duplicados) que
    la mediana del mismo día de la semana en las 8 semanas alrededor. Los feriados no cuentan como anomalía por tener
    poca gente: el modelo los aprende.
    """
    fer = feriados_ar(s.index)["tipo"].reindex(s.index).notna().to_numpy()
    cruzada = s.index.isin(fechas_cruzadas(s))
    r = _relacion_con_tipico(s, fer | cruzada)
    return s.isna() | cruzada | ((r < bajo) & ~fer) | (r > alto)


def tipico_dia(s: pd.Series, anomalia: pd.Series) -> pd.Series:
    """Mediana de los días normales del mismo día de la semana en las 8 semanas alrededor (para mostrar desvíos)."""
    base = s.where(~anomalia.to_numpy())
    tipico = pd.Series(index=s.index, dtype=float)
    for d in range(7):
        sub = base[s.index.dayofweek == d]
        tipico.loc[sub.index] = sub.rolling(9, center=True, min_periods=3).median()
    return tipico


def feriados_ar(fechas) -> pd.DataFrame:
    """
    Feriados de Argentina (paquete `holidays`) con su tipo: feriado, puente (con fines turísticos) o no laborable
    (Jueves Santo). Se suman el 24 y el 31 de diciembre, que en el subte se comportan como feriado, y los
    traslados que el paquete no trae.
    """
    import holidays

    fechas = pd.DatetimeIndex(fechas)
    anios = list(range(fechas.min().year - 1, fechas.max().year + 2))
    filas = []
    for d, nombre in holidays.Argentina(years=anios).items():
        partes = [p.strip().lower() for p in nombre.split(";")]
        if any("turístic" not in p and "jueves santo" not in p for p in partes):
            tipo = "feriado"
        elif any("turístic" in p for p in partes):
            tipo = "puente"
        else:
            tipo = "no_laborable"
        filas.append({"fecha": pd.Timestamp(d), "nombre": nombre, "tipo": tipo})
    for anio in anios:
        filas += [{"fecha": pd.Timestamp(f"{anio}-12-24"), "nombre": "Nochebuena", "tipo": "feriado"},
                  {"fecha": pd.Timestamp(f"{anio}-12-31"), "nombre": "Fin de año", "tipo": "feriado"}]
    filas += [{"fecha": pd.Timestamp(f), "nombre": n, "tipo": "feriado"} for f, n in FERIADOS_EXTRA.items()]
    return pd.DataFrame(filas).drop_duplicates("fecha").set_index("fecha").sort_index()


def calendario(fechas) -> pd.DataFrame:
    fechas = pd.DatetimeIndex(fechas)
    fer = feriados_ar(fechas)
    tipo = fer["tipo"].reindex(fechas)
    cualquiera = tipo.notna().to_numpy()
    # un día hábil pegado a un feriado (antes o después) suele tener menos gente
    pegado = ((fechas - pd.Timedelta(days=1)).isin(fer.index) | (fechas + pd.Timedelta(days=1)).isin(fer.index)) & ~cualquiera
    vacaciones = np.zeros(len(fechas), dtype=int)
    for desde, hasta in VACACIONES_INVIERNO.values():
        vacaciones |= ((fechas >= pd.Timestamp(desde)) & (fechas <= pd.Timestamp(hasta))).astype(int)
    doy = fechas.dayofyear.to_numpy()
    return pd.DataFrame({
        "dia_semana": fechas.dayofweek,
        "fin_de_semana": (fechas.dayofweek >= 5).astype(int),
        "mes": fechas.month,
        "dia_mes": fechas.day,
        "dia_anio_sin": np.sin(2 * np.pi * doy / 365.25),
        "dia_anio_cos": np.cos(2 * np.pi * doy / 365.25),
        "semana_mes": (fechas.day - 1) // 7 + 1,
        "feriado": (tipo == "feriado").astype(int).to_numpy(),
        "puente": (tipo == "puente").astype(int).to_numpy(),
        "no_laborable": (tipo == "no_laborable").astype(int).to_numpy(),
        "pegado_feriado": (pegado & (fechas.dayofweek < 5)).astype(int),
        "vacaciones_invierno": vacaciones,
        "anio": fechas.year,
    }, index=fechas)


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def _xgb(**kw):
    from xgboost import XGBRegressor

    base = dict(n_estimators=800, learning_rate=0.03, max_depth=4, min_child_weight=3, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=SEMILLA, verbosity=0)
    base.update(kw)
    return XGBRegressor(**base)


def modelos() -> dict:
    return {
        "Random Forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3, max_features=0.6, n_jobs=-1,
                                               random_state=SEMILLA),
        "XGBoost": _xgb(),
    }


def ingenuo_anio_anterior(s: pd.Series, anomalia: pd.Series, fechas) -> np.ndarray:
    """
    El mismo día de la semana de hace 52 semanas (364 días). Si ese día fue anómalo o falta, la mediana de los días
    normales de ese mes y día de la semana del año anterior; si tampoco hay, queda vacío.
    """
    fechas = pd.DatetimeIndex(fechas)
    normal = s.where(~anomalia.to_numpy())
    atras = normal.reindex(fechas - pd.Timedelta(days=364)).to_numpy()
    ref = normal.dropna()
    respaldo = ref.groupby([ref.index.year, ref.index.month, ref.index.dayofweek]).median()
    out = []
    for f, v in zip(fechas, atras):
        if np.isnan(v):
            v = respaldo.get((f.year - 1, f.month, f.dayofweek), np.nan)
        out.append(v)
    return np.array(out, dtype=float)


def entrenar(s: pd.Series, anomalia: pd.Series, hasta, modelo):
    """Entrena con los días normales hasta la fecha indicada (incluida), en escala logarítmica."""
    tr = s.index[(s.index <= pd.Timestamp(hasta)) & ~anomalia.to_numpy()]
    return modelo.fit(calendario(tr), np.log(s.loc[tr].to_numpy()))


def predecir(modelo, fechas) -> np.ndarray:
    return np.exp(modelo.predict(calendario(fechas)))


def metricas(real: np.ndarray, pred: np.ndarray) -> dict:
    ok = ~np.isnan(real) & ~np.isnan(pred)
    r, p = real[ok], pred[ok]
    return {"MAPE (%)": float(np.mean(np.abs(p - r) / r) * 100), "MAE (pasajeros/día)": float(np.mean(np.abs(p - r))),
            "error del total (%)": float((p.sum() - r.sum()) / r.sum() * 100), "días evaluados": int(ok.sum())}


def nombre_periodo(desde, hasta) -> str:
    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    if desde.year == hasta.year:
        return f"{MESES[desde.month - 1]} a {MESES[hasta.month - 1]} de {hasta.year}"
    return f"{nombre_mes(desde)} a {nombre_mes(hasta)}"


def cortes_validacion(s: pd.Series, meses: int = 4, minimo_meses: int = 12, maximo: int = 2) -> list[tuple[str, str, str]]:
    """
    Validación hacia adelante con origen móvil: los períodos de prueba más recientes, de `meses` meses cada uno,
    siempre con al menos `minimo_meses` de historia para entrenar. Devuelve (nombre, último día de entrenamiento,
    último día de prueba), del más viejo al más nuevo.
    """
    datos = s.dropna().index
    inicio, fin_te = datos.min(), datos.max()
    cortes = []
    while len(cortes) < maximo:
        fin_tr = fin_te + pd.Timedelta(days=1) - pd.DateOffset(months=meses) - pd.Timedelta(days=1)
        if fin_tr < inicio + pd.DateOffset(months=minimo_meses) - pd.Timedelta(days=1):
            break
        cortes.append((nombre_periodo(fin_tr + pd.Timedelta(days=1), fin_te), str(fin_tr.date()), str(fin_te.date())))
        fin_te = fin_tr
    return cortes[::-1]


def horizonte(fin, meses: int = 6) -> pd.Timestamp:
    """Último día del pronóstico: seis meses después del último dato."""
    return pd.Timestamp(fin) + pd.DateOffset(months=meses) + pd.offsets.MonthEnd(0)


def validar(s: pd.Series, anomalia: pd.Series, cortes: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validación hacia adelante. Para cada corte se entrena hasta el último día de entrenamiento y se predice el
    período de prueba. Todos los modelos se miden en los mismos días: los normales en los que la referencia
    ingenua tiene valor. Devuelve las métricas por modelo y corte, y las predicciones día por día.
    """
    filas, preds = [], []
    for nombre, fin_tr, fin_te in cortes:
        fechas = s.index[(s.index > pd.Timestamp(fin_tr)) & (s.index <= pd.Timestamp(fin_te))]
        fechas = fechas[~anomalia.reindex(fechas).to_numpy()]
        candidatos = {"Mismo día del año anterior": ingenuo_anio_anterior(s, anomalia, fechas)}
        for m_nombre, m in modelos().items():
            candidatos[m_nombre] = predecir(entrenar(s, anomalia, fin_tr, m), fechas)
        comun = np.all([~np.isnan(p) for p in candidatos.values()], axis=0)
        evaluables = fechas[comun]
        real = s.reindex(evaluables).to_numpy()
        for m_nombre, p in candidatos.items():
            filas.append({"período de prueba": nombre, "modelo": m_nombre, **metricas(real, p[comun])})
            preds.append(pd.DataFrame({"fecha": evaluables, "periodo": nombre, "modelo": m_nombre, "real": real,
                                       "prediccion": p[comun]}))
    return pd.DataFrame(filas), pd.concat(preds, ignore_index=True)


def pronosticar(s: pd.Series, anomalia: pd.Series, hasta: str, modelo_nombre: str = "XGBoost",
                cuantiles=(0.1, 0.9)) -> pd.DataFrame:
    """Pronóstico día por día desde el último dato hasta `hasta`, con un intervalo del 80 %."""
    fechas = pd.date_range(s.dropna().index.max() + pd.Timedelta(days=1), pd.Timestamp(hasta), freq="D")
    fin = s.dropna().index.max()
    punto = predecir(entrenar(s, anomalia, fin, modelos()[modelo_nombre]), fechas)
    out = pd.DataFrame({"fecha": fechas, "pronostico": punto})
    for q in cuantiles:
        m = _xgb(objective="reg:quantileerror", quantile_alpha=q)
        out[f"p{int(q * 100)}"] = predecir(entrenar(s, anomalia, fin, m), fechas)
    out[f"p{int(cuantiles[0] * 100)}"] = out[[f"p{int(cuantiles[0] * 100)}", "pronostico"]].min(axis=1)
    out[f"p{int(cuantiles[1] * 100)}"] = out[[f"p{int(cuantiles[1] * 100)}", "pronostico"]].max(axis=1)
    cal = calendario(fechas)
    out["tipo_dia"] = np.where(cal["feriado"].to_numpy() | cal["puente"].to_numpy(), "feriado",
                               np.where(cal["fin_de_semana"].to_numpy() == 1, "fin de semana", "hábil"))
    return out


def por_mes(s: pd.Series, anomalia: pd.Series, pron: pd.DataFrame) -> pd.DataFrame:
    """Pasajeros por día hábil promedio de cada mes pronosticado, contra el mismo mes del año anterior."""
    hab = pron[pron["tipo_dia"] == "hábil"].assign(mes=lambda d: d["fecha"].dt.to_period("M"))
    out = hab.groupby("mes").agg(pronostico=("pronostico", "mean"), p10=("p10", "mean"), p90=("p90", "mean"))
    real = s[~anomalia]
    cal = calendario(real.index)
    habiles = real[(cal["fin_de_semana"] == 0).to_numpy() & (cal[["feriado", "puente", "no_laborable"]].sum(axis=1) == 0).to_numpy()]
    previo = habiles.groupby(habiles.index.to_period("M")).mean()
    out["anio_anterior"] = [previo.get(p - 12, np.nan) for p in out.index]
    out["variacion_%"] = (out["pronostico"] / out["anio_anterior"] - 1) * 100
    return out


def importancia(s: pd.Series, anomalia: pd.Series, modelo_nombre: str = "Random Forest", hasta=None) -> pd.Series:
    """
    Importancia de cada variable en el modelo entrenado con toda la serie: cuánto baja el error, en promedio, cada
    vez que un árbol corta por esa variable (en Random Forest, la reducción de varianza; en XGBoost, la ganancia).
    """
    hasta = hasta or s.dropna().index.max()
    m = entrenar(s, anomalia, hasta, modelos()[modelo_nombre])
    imp = pd.Series(m.feature_importances_, index=VARIABLES)
    return (imp / imp.sum()).sort_values(ascending=False)


def perfil_horario(hora: pd.DataFrame, meses: int = 6) -> pd.DataFrame:
    """Parte de las entradas del día que ocurre en cada hora (días hábiles y fines de semana, últimos meses)."""
    h = hora.copy()
    h["fecha"] = pd.to_datetime(h["fecha"])
    h = h[h["fecha"] > h["fecha"].max() - pd.DateOffset(months=meses)]
    cal = calendario(pd.DatetimeIndex(h["fecha"].unique()))
    habil = cal.index[(cal["fin_de_semana"] == 0) & (cal[["feriado", "puente", "no_laborable"]].sum(axis=1) == 0)]
    h["tipo_dia"] = np.where(h["fecha"].isin(habil), "hábil", np.where(h["fecha"].dt.dayofweek >= 5, "fin de semana", "otro"))
    h = h[h["tipo_dia"] != "otro"]
    tot = h.groupby(["tipo_dia", "hora"])["pasajeros"].sum()
    return (tot / tot.groupby(level=0).transform("sum")).rename("parte").reset_index()


# --------------------------------------------------------------------------
# Estaciones: cuánta gente va a pasar por cada una
# --------------------------------------------------------------------------
def por_estacion(diario: pd.DataFrame, anomalia: pd.Series, meses: int = 3) -> pd.DataFrame:
    """Entradas por día hábil de cada estación en los últimos meses y su parte del total de la red."""
    d = _sin_pruebas(diario).copy()
    d["fecha"] = pd.to_datetime(d["fecha"])
    d = d[d["fecha"] > d["fecha"].max() - pd.DateOffset(months=meses)]
    cal = calendario(pd.DatetimeIndex(sorted(d["fecha"].unique())))
    habiles = cal.index[(cal["fin_de_semana"] == 0) & (cal[["feriado", "puente", "no_laborable"]].sum(axis=1) == 0)]
    normales = anomalia.index[~anomalia.to_numpy()]
    d = d[d["fecha"].isin(habiles) & d["fecha"].isin(normales)]
    d["letra"] = d["linea"].str.replace("LINEA", "", regex=False).str[0]
    d["base"] = d["estacion"].str.replace(r"\.[A-H]$", "", regex=True).map(normalizar).replace(_ALIAS_ESTACIONES)
    out = (d.groupby(["letra", "base"])["pasajeros"].sum() / d["fecha"].nunique()).rename("pax_habil_reciente").reset_index()
    out["parte"] = out["pax_habil_reciente"] / out["pax_habil_reciente"].sum()
    return out


# --------------------------------------------------------------------------
# Todo junto: python -m src.pronostico
# --------------------------------------------------------------------------
def resultados(base: Path = DIR_BASE, salida: Path = RAIZ / "data" / "processed") -> dict:
    """Valida, pronostica y guarda las tablas que usan el notebook 05, la app y la presentación."""
    diario, hora = cargar(base)
    s = serie_red(diario)
    anomalia = marcar_anomalias(s)
    fin = s.dropna().index.max()
    met, _ = validar(s, anomalia, cortes_validacion(s))
    promedio = met.groupby("modelo")["MAPE (%)"].mean().sort_values()
    elegido = next(m for m in promedio.index if m != "Mismo día del año anterior")
    pron = pronosticar(s, anomalia, horizonte(fin), elegido)
    mensual = por_mes(s, anomalia, pron)
    imp = importancia(s, anomalia, elegido)

    est = por_estacion(diario, anomalia)
    ref = pd.read_csv(salida / "estaciones_subte.csv")
    est = est.merge(ref[["linea", "base", "estacion", "perfil", "link_cerca"]], left_on=["letra", "base"],
                    right_on=["linea", "base"], how="inner").drop(columns="letra")
    primer_mes = mensual.index[0] if mensual.index[0].to_timestamp() > fin else mensual.index[min(1, len(mensual) - 1)]
    est["pax_habil_pronostico"] = est["parte"] * mensual.loc[primer_mes, "pronostico"]

    salida.mkdir(parents=True, exist_ok=True)
    met.to_csv(salida / "pronostico_validacion.csv", index=False)
    pron.to_csv(salida / "pronostico_diario.csv", index=False)
    mensual.assign(mes=mensual.index.astype(str)).to_csv(salida / "pronostico_mensual.csv", index=False)
    imp.rename("importancia").rename_axis("variable").reset_index().assign(
        nombre=lambda d: d["variable"].map(NOMBRES)).to_csv(salida / "pronostico_importancia.csv", index=False)
    est.sort_values("pax_habil_pronostico", ascending=False).to_csv(salida / "pronostico_estaciones.csv", index=False)
    picos = {}
    if hora is not None:
        perfil = perfil_horario(hora)
        perfil.to_csv(salida / "pronostico_perfil_horario.csv", index=False)
        h = perfil[perfil["tipo_dia"] == "hábil"].set_index("hora")["parte"]
        picos = {"parte_pico_manana": float(h.loc[7:9].sum()), "parte_pico_tarde": float(h.loc[17:19].sum())}

    sin_link = est[~est["link_cerca"].astype(bool)]
    ultimo = met[met["período de prueba"] == met["período de prueba"].iloc[-1]].set_index("modelo")
    minimo = mensual["pronostico"].idxmin()
    resumen = {
        "desde": str(s.dropna().index.min().date()), "hasta": str(fin.date()), "dias": int(s.notna().sum()),
        "dias_anomalos": int((anomalia & s.notna()).sum()), "dias_sin_dato": int(s.isna().sum()),
        "dias_fecha_cruzada": int((s.index.isin(fechas_cruzadas(s)) & s.notna()).sum()),
        "modelo_elegido": elegido, "mape_promedio": {k: float(v) for k, v in promedio.items()},
        "periodos_prueba": list(met["período de prueba"].unique()),
        "mape_ultimo_periodo": {k: float(v) for k, v in ultimo["MAPE (%)"].items()},
        "error_total_ultimo_periodo": {k: float(v) for k, v in ultimo["error del total (%)"].items()},
        "pronostico_hasta": str(pron["fecha"].max().date()), "primer_mes": str(primer_mes),
        "primer_mes_nombre": nombre_mes(primer_mes.to_timestamp()),
        "pax_habil_primer_mes": float(mensual.loc[primer_mes, "pronostico"]),
        "variacion_primer_mes_pct": float(mensual.loc[primer_mes, "variacion_%"]),
        "mes_minimo_nombre": nombre_mes(minimo.to_timestamp()),
        "pax_habil_mes_minimo": float(mensual.loc[minimo, "pronostico"]),
        "mensual": [{"mes": str(p), "nombre": nombre_mes(p.to_timestamp()), "pax_habil": float(f["pronostico"]),
                     "p10": float(f["p10"]), "p90": float(f["p90"]),
                     "anio_anterior": None if pd.isna(f["anio_anterior"]) else float(f["anio_anterior"]),
                     "variacion_pct": None if pd.isna(f["variacion_%"]) else float(f["variacion_%"])}
                    for p, f in mensual.iterrows()],
        "estaciones_sin_link": len(sin_link),
        "pax_habil_estaciones_sin_link": float(sin_link["pax_habil_pronostico"].sum()),
        "variable_principal": NOMBRES[imp.index[0]],
        **picos,
    }
    (salida / "pronostico_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    if not hay_serie():
        raise SystemExit("Falta la serie diaria: correr notebooks/00c_serie_diaria_subte.ipynb en Colab y copiar "
                         f"{ARCHIVO_DIARIO} y {ARCHIVO_HORA} a data/base/")
    print(json.dumps(resultados(), ensure_ascii=False, indent=2))
