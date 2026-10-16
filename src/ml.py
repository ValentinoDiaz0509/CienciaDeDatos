"""
ml.py
-----
Machine learning predictivo (clases 6 y 8: árboles de decisión, Random Forest y boosting con XGBoost).

Dos preguntas de negocio, dos modelos supervisados:

1. Regresión. ¿Cuántas terminales le corresponden a cada zona según su demanda?
   Objetivo: terminales (todas las redes) en el área de influencia de cada hexágono.
2. Clasificación. ¿A esta zona le corresponde al menos un cajero?
   Objetivo: el hexágono tiene al menos una terminal (cualquier red).

Validación cruzada ESPACIAL ANIDADA: el ajuste de hiperparámetros (GridSearchCV) se hace
adentro de cada partición de entrenamiento, también dejando comunas afuera, y el desempeño
se mide siempre en comunas que el modelo no vio. Así la métrica no queda inflada por haber
elegido los hiperparámetros mirando los mismos datos con los que se evalúa.

Explicación por zona: el modelo elegido es aditivo en escala logarítmica, así que la
predicción de cada zona se descompone exactamente en el aporte de cada variable (valores
SHAP de XGBoost, que el propio XGBoost calcula con TreeSHAP, y coeficiente × valor en la
regresión de Poisson).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import modelos as m

SEMILLA = m.SEMILLA
PLIEGUES = 5            # particiones externas (evaluación)
PLIEGUES_INTERNOS = 4   # particiones internas (ajuste de hiperparámetros)


def _log1p(X):
    return np.log1p(np.clip(X, 0, None))


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def _xgb_regresor(**kw):
    from xgboost import XGBRegressor

    base = dict(objective="count:poisson", n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=5,
                reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=1,
                random_state=SEMILLA, verbosity=0)
    base.update(kw)
    return XGBRegressor(**base)


def _xgb_clasificador(**kw):
    from xgboost import XGBClassifier

    base = dict(n_estimators=400, learning_rate=0.05, max_depth=3, min_child_weight=5, reg_lambda=10.0,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=SEMILLA,
                verbosity=0, eval_metric="logloss")
    base.update(kw)
    return XGBClassifier(**base)


def _glm(alpha: float = 1e-3):
    return make_pipeline(FunctionTransformer(_log1p), StandardScaler(), PoissonRegressor(alpha=alpha, max_iter=3000))


class EnsambleGeometrico(RegressorMixin, BaseEstimator):
    """
    Promedio, en escala logarítmica, de una regresión de Poisson y un XGBoost con pérdida de
    Poisson: log(predicción) = (log(GLM) + log(XGBoost)) / 2. El GLM aporta extrapolación
    suave (el microcentro tiene valores que los árboles no vieron) y XGBoost aporta las no
    linealidades e interacciones. Al ser un promedio de logaritmos, la predicción sigue
    siendo la suma de los aportes de cada variable.
    """

    def __init__(self, glm=None, xgb=None):
        self.glm = glm
        self.xgb = xgb

    def fit(self, X, y):
        self.glm_ = clone(self.glm if self.glm is not None else _glm()).fit(X, y)
        self.xgb_ = clone(self.xgb if self.xgb is not None else _xgb_regresor()).fit(X, y)
        return self

    def predict(self, X):
        a = np.clip(self.glm_.predict(X), 1e-9, None)
        b = np.clip(self.xgb_.predict(X), 1e-9, None)
        return np.exp((np.log(a) + np.log(b)) / 2)


def especificaciones_regresion() -> dict:
    """Modelo y grilla de hiperparámetros (None = sin ajuste). Del más simple al más complejo."""
    xgb_grilla = {"max_depth": [2, 3, 4], "min_child_weight": [1, 5, 20], "reg_lambda": [1.0, 10.0]}
    return {
        "Línea de base (promedio)": (DummyRegressor(strategy="mean"), None),
        "Poisson solo con población": (make_pipeline(
            FunctionTransformer(lambda X: _log1p(X[["poblacion_k1"]])), StandardScaler(),
            PoissonRegressor(alpha=1e-4, max_iter=1000)), None),
        "Regresión de Poisson (GLM)": (_glm(), {"poissonregressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
        "Árbol de decisión (CART)": (DecisionTreeRegressor(criterion="poisson", random_state=SEMILLA),
                                     {"max_depth": [3, 4, 5, 6, 8, 10], "min_samples_leaf": [10, 20, 40]}),
        "Random Forest": (RandomForestRegressor(n_estimators=300, criterion="poisson", n_jobs=-1, random_state=SEMILLA),
                          {"min_samples_leaf": [5, 10, 20], "max_features": [0.33, 0.5]}),
        "XGBoost": (_xgb_regresor(), xgb_grilla),
        "Ensamble (Poisson + XGBoost)": (EnsambleGeometrico(glm=_glm(), xgb=_xgb_regresor()),
                                         {f"xgb__{k}": v for k, v in xgb_grilla.items()}),
    }


# --------------------------------------------------------------------------
# Validación cruzada espacial anidada
# --------------------------------------------------------------------------
def _ajustar(est, grilla, X, y, grupos, scoring):
    if not grilla:
        return clone(est).fit(X, y), {}
    gs = GridSearchCV(clone(est), grilla, cv=GroupKFold(n_splits=PLIEGUES_INTERNOS), scoring=scoring, n_jobs=-1,
                      refit=True)
    gs.fit(X, y, groups=grupos)
    return gs.best_estimator_, gs.best_params_


def validacion_anidada(X, y, grupos, especs: dict, scoring: str, predecir, metricas_fn):
    """
    Devuelve:
      - tabla de métricas (agregadas sobre todas las predicciones fuera de muestra, y promedio y
        desvío entre particiones),
      - predicciones fuera de muestra por modelo,
      - hiperparámetros elegidos en cada partición,
      - los modelos ajustados en cada partición con los índices de prueba (para explicarlos).
    """
    externos = list(GroupKFold(n_splits=PLIEGUES).split(X, y, grupos))
    filas, oof, params, ajustes = [], {}, {}, {}
    for nombre, (est, grilla) in especs.items():
        pred = np.zeros(len(y))
        por_pliegue, params[nombre], ajustes[nombre] = [], [], []
        for k, (tr, te) in enumerate(externos):
            modelo, p = _ajustar(est, grilla, X.iloc[tr], y.iloc[tr], grupos.iloc[tr], scoring)
            pred[te] = predecir(modelo, X.iloc[te])
            por_pliegue.append(metricas_fn(y.iloc[te].to_numpy(), pred[te]))
            params[nombre].append(p)
            ajustes[nombre].append((te, modelo))
        oof[nombre] = pred
        agregada = metricas_fn(y.to_numpy(), pred)
        pp = pd.DataFrame(por_pliegue)
        fila = {"modelo": nombre, "validación": "espacial (por comuna)", **agregada}
        for c in pp.columns:
            fila[f"{c} (promedio por partición)"] = pp[c].mean()
            fila[f"{c} (desvío entre particiones)"] = pp[c].std()
        filas.append(fila)
    return pd.DataFrame(filas), oof, params, ajustes, externos


def params_mas_frecuentes(lista: list[dict]) -> dict:
    if not lista or not lista[0]:
        return {}
    clave = Counter(tuple(sorted(p.items())) for p in lista).most_common(1)[0][0]
    return dict(clave)


def validacion_aleatoria(X, y, especs: dict, params: dict, predecir, metricas_fn) -> pd.DataFrame:
    """
    La misma comparación con particiones al azar (KFold), con los hiperparámetros más elegidos.
    Sirve para mostrar cuánto se infla el desempeño cuando las zonas vecinas quedan de los dos
    lados de la partición (autocorrelación espacial).
    """
    filas = []
    for nombre, (est, _) in especs.items():
        modelo = clone(est).set_params(**params_mas_frecuentes(params[nombre]))
        pred, por_pliegue = np.zeros(len(y)), []
        for tr, te in KFold(n_splits=PLIEGUES, shuffle=True, random_state=SEMILLA).split(X):
            pred[te] = predecir(clone(modelo).fit(X.iloc[tr], y.iloc[tr]), X.iloc[te])
            por_pliegue.append(metricas_fn(y.iloc[te].to_numpy(), pred[te]))
        fila = {"modelo": nombre, "validación": "aleatoria", **metricas_fn(y.to_numpy(), pred)}
        pp = pd.DataFrame(por_pliegue)
        for c in pp.columns:
            fila[f"{c} (promedio por partición)"] = pp[c].mean()
            fila[f"{c} (desvío entre particiones)"] = pp[c].std()
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_regresion(X, y, grupos):
    especs = especificaciones_regresion()
    esp, oof, params, ajustes, externos = validacion_anidada(
        X, y, grupos, especs, "neg_mean_poisson_deviance", lambda mod, Z: mod.predict(Z), m.metricas)
    ale = validacion_aleatoria(X, y, especs, params, lambda mod, Z: mod.predict(Z), m.metricas)
    return pd.concat([esp, ale], ignore_index=True), oof, params, ajustes, externos


def elegir_regresion(tabla: pd.DataFrame) -> str:
    """Regla fijada de antemano: mayor D² de Poisson promedio entre las particiones espaciales."""
    esp = tabla[tabla["validación"].str.startswith("espacial")]
    return esp.sort_values("D² Poisson (promedio por partición)", ascending=False).iloc[0]["modelo"]


# --------------------------------------------------------------------------
# Sobreajuste y OOB (clases 6 y 8)
# --------------------------------------------------------------------------
def curva_profundidad(X, y, grupos, profundidades=range(1, 16)) -> pd.DataFrame:
    """D² en entrenamiento y en comunas no vistas según la profundidad del árbol."""
    filas = []
    for d in profundidades:
        arbol = DecisionTreeRegressor(criterion="poisson", max_depth=d, min_samples_leaf=2, random_state=SEMILLA)
        pred = cross_val_predict(arbol, X, y, groups=grupos, cv=GroupKFold(n_splits=PLIEGUES))
        entren = clone(arbol).fit(X, y).predict(X)
        filas.append({"profundidad": d, "entrenamiento": m.metricas(y.to_numpy(), entren)["D² Poisson"],
                      "validación espacial": m.metricas(y.to_numpy(), pred)["D² Poisson"]})
    return pd.DataFrame(filas)


def oob_vs_validacion(X, y, grupos, Xc, yc, params_rf: dict, params_rfc: dict) -> pd.DataFrame:
    """
    El error OOB (out-of-bag) del Random Forest es una validación "gratis": cada árbol se evalúa
    con las filas que no entraron en su muestra bootstrap. Pero esas filas son vecinas de las que
    sí entraron, así que con datos espaciales es optimista. Lo comparamos con dejar comunas afuera.
    """
    rf = RandomForestRegressor(n_estimators=300, criterion="poisson", oob_score=True, n_jobs=-1,
                               random_state=SEMILLA, **params_rf).fit(X, y)
    from sklearn.metrics import r2_score

    r2_esp = r2_score(y, cross_val_predict(clone(rf).set_params(oob_score=False), X, y, groups=grupos,
                                           cv=GroupKFold(n_splits=PLIEGUES)))
    rfc = RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, random_state=SEMILLA,
                                 **params_rfc).fit(Xc, yc)
    auc_oob = roc_auc_score(yc, rfc.oob_decision_function_[:, 1])
    auc_esp = roc_auc_score(yc, cross_val_predict(clone(rfc).set_params(oob_score=False), Xc, yc, groups=grupos,
                                                  cv=GroupKFold(n_splits=PLIEGUES), method="predict_proba")[:, 1])
    return pd.DataFrame([
        {"modelo": "Random Forest (regresión)", "métrica": "R²", "OOB": rf.oob_score_, "comunas no vistas": r2_esp},
        {"modelo": "Random Forest (clasificación)", "métrica": "AUC", "OOB": auc_oob, "comunas no vistas": auc_esp},
    ])


def importancia_gini(X, y, params_rf: dict) -> pd.Series:
    """Importancia por reducción de impureza (MDI) del Random Forest, como en la clase 8."""
    rf = RandomForestRegressor(n_estimators=300, criterion="poisson", n_jobs=-1, random_state=SEMILLA,
                               **params_rf).fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------
# Explicación por zona (aportes en escala logarítmica)
# --------------------------------------------------------------------------
def _aportes_glm(glm, X):
    """Coeficiente × variable estandarizada: aporte de cada variable a log(predicción)."""
    z = glm[:-1].transform(X)
    reg = glm[-1]
    return pd.DataFrame(z * reg.coef_, index=X.index, columns=X.columns), float(reg.intercept_)


def _aportes_xgb(xgb, X):
    """Valores SHAP de XGBoost (TreeSHAP) en la escala del margen, que con Poisson es log(predicción)."""
    from xgboost import DMatrix

    c = xgb.get_booster().predict(DMatrix(X), pred_contribs=True)
    return pd.DataFrame(c[:, :-1], index=X.index, columns=X.columns), float(np.mean(c[:, -1]))


def aportes(modelo, X) -> tuple[pd.DataFrame, float]:
    """Aportes por variable y valor base, tales que base + suma de aportes = log(predicción)."""
    if isinstance(modelo, EnsambleGeometrico):
        a, ba = _aportes_glm(modelo.glm_, X)
        b, bb = _aportes_xgb(modelo.xgb_, X)
        return (a + b) / 2, (ba + bb) / 2
    if hasattr(modelo, "get_booster"):
        return _aportes_xgb(modelo, X)
    return _aportes_glm(modelo, X)


def aportes_fuera_de_muestra(ajustes: list, X) -> pd.DataFrame:
    """Aportes de cada zona según el modelo que NO la vio al entrenar (los mismos de la brecha)."""
    partes = []
    for te, modelo in ajustes:
        a, base = aportes(modelo, X.iloc[te])
        partes.append(a.assign(_base=base))
    return pd.concat(partes).loc[X.index]


# --------------------------------------------------------------------------
# 2. Clasificación
# --------------------------------------------------------------------------
VARIABLES_HEX = {
    "poblacion": "Residentes (hexágono)",
    "actividad_osm": "Comercios (hexágono)",
    "parcelas_comercio": "Parcelas comerciales (hexágono)",
    "parcelas_oficinas": "Parcelas de oficinas (hexágono)",
    "pax_subte_habil": "Pasajeros de subte (hexágono)",
    "estaciones_tren": "Estaciones de tren (hexágono)",
}
NOMBRES = {**m.NOMBRES, **VARIABLES_HEX}


def preparar_clasificacion(tabla: pd.DataFrame, X: pd.DataFrame, mascara: pd.Series):
    t = tabla.loc[mascara]
    hexv = [c for c in VARIABLES_HEX if c in t.columns]
    Xc = pd.concat([X, t[hexv].fillna(0)], axis=1)
    yc = (t["term_total"] > 0).astype(int)
    return Xc, yc


def especificaciones_clasificacion() -> dict:
    return {
        "Regresión logística": (make_pipeline(FunctionTransformer(_log1p), StandardScaler(),
                                              LogisticRegression(max_iter=3000)),
                                {"logisticregression__C": [0.1, 1.0, 10.0]}),
        "Árbol de decisión": (DecisionTreeClassifier(random_state=SEMILLA),
                              {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [20, 50]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEMILLA),
                          {"min_samples_leaf": [5, 10, 20], "max_features": [0.33, 0.5]}),
        "XGBoost": (_xgb_clasificador(), {"max_depth": [2, 3, 4], "min_child_weight": [1, 5], "reg_lambda": [1.0, 10.0]}),
    }


def umbral_tasa_base(y: np.ndarray, p: np.ndarray) -> float:
    """Umbral con el que el modelo marca tantas zonas como las que hoy tienen cajero."""
    return float(np.quantile(p, 1 - y.mean()))


def metricas_clasificacion(y: np.ndarray, p: np.ndarray, umbral: float | None = None) -> dict:
    umbral = umbral_tasa_base(y, p) if umbral is None else umbral
    pred = (p >= umbral).astype(int)
    return {"AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Precisión": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "Exactitud": float((pred == y).mean()), "umbral": umbral}


def comparar_clasificacion(Xc, yc, grupos):
    especs = especificaciones_clasificacion()
    proba = lambda mod, Z: mod.predict_proba(Z)[:, 1]  # noqa: E731
    met = lambda y_, p_: {k: v for k, v in metricas_clasificacion(y_, p_).items()}  # noqa: E731
    esp, oof, params, ajustes, externos = validacion_anidada(Xc, yc, grupos, especs, "roc_auc", proba, met)
    ale = validacion_aleatoria(Xc, yc, especs, params, proba, met)
    return pd.concat([esp, ale], ignore_index=True), oof, params, ajustes


def elegir_clasificador(tabla: pd.DataFrame) -> str:
    esp = tabla[tabla["validación"].str.startswith("espacial")]
    return esp.sort_values("AUC (promedio por partición)", ascending=False).iloc[0]["modelo"]


def matriz(y: np.ndarray, p: np.ndarray, umbral: float) -> pd.DataFrame:
    cm = confusion_matrix(y, (p >= umbral).astype(int), labels=[0, 1])
    return pd.DataFrame(cm, index=["Real: sin cajero", "Real: con cajero"],
                        columns=["Predicho: sin cajero", "Predicho: con cajero"])


def arbol_explicativo(Xc, yc, profundidad: int = 3) -> DecisionTreeClassifier:
    """Árbol chico, entrenado con todo, solo para leer las reglas que aprende (no para predecir)."""
    return DecisionTreeClassifier(max_depth=profundidad, min_samples_leaf=50, random_state=SEMILLA).fit(Xc, yc)


# --------------------------------------------------------------------------
# Resultado de negocio: cajeros que le corresponden a cada barrio
# --------------------------------------------------------------------------
def por_barrio(res: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada barrio: terminales que tiene y terminales que le corresponden según su demanda
    (predicción fuera de muestra). La predicción es para el área de influencia de cada
    hexágono (él y sus 6 vecinos), así que se reparte en partes iguales entre los 7 para no
    contar cada terminal siete veces.
    """
    r = res[res["esperado_total_k1"].notna()]
    b = (r.groupby("barrio")
         .agg(comuna=("comuna", "first"), residentes=("poblacion", "sum"),
              pasajeros_subte=("pax_subte_habil", "sum"), comercios=("actividad_osm", "sum"),
              tiene_total=("term_total", "sum"), tiene_link=("term_link", "sum"), tiene_banelco=("term_banelco", "sum"),
              corresponden_total=("esperado_total_k1", lambda s: s.sum() / 7),
              corresponden_link=("esperado_link_k1", lambda s: s.sum() / 7)))
    b["diferencia_link"] = b["corresponden_link"] - b["tiene_link"]
    b["cuota_link"] = b["tiene_link"] / b["tiene_total"].replace(0, np.nan)
    return b.sort_values("diferencia_link", ascending=False)


def explicar_zona(aportes_hex: pd.DataFrame, X: pd.DataFrame, idx, n: int = 5) -> pd.DataFrame:
    """
    Aportes de las n variables más influyentes en una zona, como factores multiplicativos sobre una
    zona típica (×2 = duplica lo esperado; ×0,5 = lo reduce a la mitad). El resto se agrupa.
    """
    a = aportes_hex.loc[idx].drop(labels=[c for c in aportes_hex.columns if c.startswith("_")])
    orden = a.abs().sort_values(ascending=False).index
    top, resto = orden[:n], orden[n:]
    filas = [{"variable": v, "nombre": NOMBRES.get(v, v), "valor": float(X.loc[idx, v]), "aporte_log": float(a[v]),
              "factor": float(np.exp(a[v]))} for v in top]
    if len(resto):
        filas.append({"variable": "resto", "nombre": f"Otras {len(resto)} variables", "valor": np.nan,
                      "aporte_log": float(a[resto].sum()), "factor": float(np.exp(a[resto].sum()))})
    out = pd.DataFrame(filas)
    out.attrs["base"] = float(np.exp(aportes_hex.loc[idx, "_base"]))
    out.attrs["prediccion"] = float(np.exp(aportes_hex.loc[idx, "_base"] + a.sum()))
    return out


def frase_barrio(fila: pd.Series) -> str:
    """Resumen de negocio de un barrio: cuánta demanda tiene, cuánto le corresponde y cuánto tiene."""
    n = lambda v: f"{v:,.0f}".replace(",", ".")  # noqa: E731
    return (f"{n(fila['residentes'])} vecinos, {n(fila['comercios'])} comercios y {n(fila['pasajeros_subte'])} pasajeros de "
            f"subte por día hábil: le corresponden {n(fila['corresponden_total'])} terminales ({n(fila['corresponden_link'])} "
            f"de Link) y tiene {n(fila['tiene_total'])} ({n(fila['tiene_link'])} de Link)")
