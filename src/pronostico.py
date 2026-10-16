"""
pronostico.py
-------------
Análisis predictivo en el tiempo: cuánta gente va a entrar al subte cada día en los próximos
meses, a partir de la serie diaria de los molinetes (BA Data, 2022 a 2026).

Para Red Link: anticipar la demanda de paso para planificar la carga de efectivo y el
mantenimiento de los cajeros cerca de las estaciones (Gerencia Técnica).

Modelos (clases 6 y 8): Random Forest y XGBoost sobre variables de calendario (día de la
semana, mes, día del año, feriados y feriados puente), comparados con una referencia ingenua
("el mismo día de la semana del año anterior"). Validación HACIA ADELANTE: se entrena con el
pasado y se evalúa en meses posteriores que el modelo no vio; nunca se usan fechas futuras
para entrenar.

Los días con muy pocos pasajeros para su día de la semana (paros, cortes o datos faltantes) se
marcan como anomalías: no se usan para entrenar ni para medir el error, porque no son
predecibles con el calendario.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .features import DIR_BASE

SEMILLA = 42
ARCHIVO_DIARIO = "subte-diario-estacion.parquet"
ARCHIVO_HORA = "subte-hora-red.parquet"
VARIABLES = ["dia_semana", "fin_de_semana", "mes", "dia_mes", "dia_anio_sin", "dia_anio_cos", "semana_mes",
             "feriado", "puente", "no_laborable", "pegado_feriado", "anio"]
NOMBRES = {
    "dia_semana": "Día de la semana", "fin_de_semana": "Fin de semana", "mes": "Mes", "dia_mes": "Día del mes",
    "dia_anio_sin": "Época del año (seno)", "dia_anio_cos": "Época del año (coseno)", "semana_mes": "Semana del mes",
    "feriado": "Feriado", "puente": "Feriado puente", "no_laborable": "Día no laborable",
    "pegado_feriado": "Día pegado a un feriado", "anio": "Año",
}


def hay_serie(base: Path = DIR_BASE) -> bool:
    return (base / ARCHIVO_DIARIO).exists()


def cargar(base: Path = DIR_BASE) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    diario = pd.read_parquet(base / ARCHIVO_DIARIO)
    hora = pd.read_parquet(base / ARCHIVO_HORA) if (base / ARCHIVO_HORA).exists() else None
    return diario, hora


# --------------------------------------------------------------------------
# Serie y calendario
# --------------------------------------------------------------------------
def serie_red(diario: pd.DataFrame) -> pd.Series:
    """Pasajeros por día en toda la red, con todas las fechas (las que faltan quedan vacías)."""
    s = diario.groupby("fecha")["pasajeros"].sum().astype(float)
    s.index = pd.to_datetime(s.index)
    return s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D")).rename("pasajeros")


def marcar_anomalias(s: pd.Series, umbral: float = 0.5) -> pd.Series:
    """
    Verdadero en los días sin dato o con menos de la mitad de los pasajeros típicos de ese día de la
    semana en las 8 semanas alrededor (paros, cortes de servicio o archivos incompletos).
    """
    tipico = pd.Series(index=s.index, dtype=float)
    for d in range(7):
        sub = s[s.index.dayofweek == d]
        tipico.loc[sub.index] = sub.rolling(9, center=True, min_periods=3).median()
    feriados = feriados_ar(s.index)
    es_feriado = feriados["tipo"].reindex(s.index).notna()
    bajo = s < umbral * tipico
    return s.isna() | (bajo & ~es_feriado)


def feriados_ar(fechas) -> pd.DataFrame:
    """Feriados nacionales de Argentina (paquete `holidays`), con su tipo."""
    import holidays

    anios = sorted({d.year for d in pd.DatetimeIndex(fechas)} | {max(pd.DatetimeIndex(fechas)).year + 1})
    filas = []
    for d, nombre in holidays.Argentina(years=anios).items():
        n = nombre.lower()
        tipo = "puente" if "turístic" in n else "no_laborable" if "jueves santo" in n else "feriado"
        filas.append({"fecha": pd.Timestamp(d), "nombre": nombre, "tipo": tipo})
    return pd.DataFrame(filas).drop_duplicates("fecha").set_index("fecha").sort_index()


def calendario(fechas) -> pd.DataFrame:
    fechas = pd.DatetimeIndex(fechas)
    fer = feriados_ar(fechas)
    tipo = fer["tipo"].reindex(fechas)
    cualquiera = tipo.notna().to_numpy()
    # un día hábil pegado a un feriado (antes o después) suele tener menos gente
    pegado = (pd.Series(cualquiera, index=fechas).shift(1, fill_value=False)
              | pd.Series(cualquiera, index=fechas).shift(-1, fill_value=False)).to_numpy() & ~cualquiera
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


def ingenuo_anio_anterior(s: pd.Series, fechas) -> np.ndarray:
    """El mismo día de la semana de hace 52 semanas (364 días). Si ese día fue anómalo, la mediana de su mes y día."""
    fechas = pd.DatetimeIndex(fechas)
    atras = s.reindex(fechas - pd.Timedelta(days=364)).to_numpy()
    ref = s.dropna()
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


def validar(s: pd.Series, anomalia: pd.Series, cortes: list[tuple[str, str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validación hacia adelante. Cada corte es (nombre, último día de entrenamiento, último día de prueba).
    Devuelve las métricas por modelo y corte, y las predicciones día por día.
    """
    filas, preds = [], []
    for nombre, fin_tr, fin_te in cortes:
        fechas = s.index[(s.index > pd.Timestamp(fin_tr)) & (s.index <= pd.Timestamp(fin_te))]
        evaluables = fechas[~anomalia.reindex(fechas).to_numpy()]
        real = s.reindex(evaluables).to_numpy()
        candidatos = {"Mismo día del año anterior": ingenuo_anio_anterior(s, evaluables)}
        for m_nombre, m in modelos().items():
            candidatos[m_nombre] = predecir(entrenar(s, anomalia, fin_tr, m), evaluables)
        for m_nombre, p in candidatos.items():
            filas.append({"período de prueba": nombre, "modelo": m_nombre, **metricas(real, p)})
            preds.append(pd.DataFrame({"fecha": evaluables, "periodo": nombre, "modelo": m_nombre, "real": real,
                                       "prediccion": p}))
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


def importancia(s: pd.Series, anomalia: pd.Series, hasta=None) -> pd.Series:
    hasta = hasta or s.dropna().index.max()
    m = entrenar(s, anomalia, hasta, modelos()["XGBoost"])
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
