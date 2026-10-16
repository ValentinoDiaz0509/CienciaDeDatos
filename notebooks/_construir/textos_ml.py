# Textos del notebook 02 (machine learning). Se ejecuta desde nb02_ml.py con RAIZ definido; los números salen de data/processed.
import json

import numpy as np
import pandas as pd

from src import estilo as _e
from src import ml as _ml
from src import modelos as _m

_n = _e.num_es
_P = RAIZ / "data/processed"
_R = json.loads((_P / "resumen.json").read_text(encoding="utf-8"))
_met = pd.read_csv(_P / "metricas_modelos.csv")
_esp = _met[_met["validación"].str.startswith("espacial")].set_index("modelo")
_ale = _met[_met["validación"] == "aleatoria"].set_index("modelo")
_metc = pd.read_csv(_P / "metricas_clasificacion.csv")
_espc = _metc[_metc["validación"].str.startswith("espacial")].set_index("modelo")
_hp = pd.read_csv(_P / "hiperparametros.csv")
_curva = pd.read_csv(_P / "curva_profundidad.csv")
_oob = pd.read_csv(_P / "oob_vs_validacion.csv").set_index("modelo")
_imp = pd.read_csv(_P / "importancia_variables.csv")
_gini = pd.read_csv(_P / "importancia_gini.csv")
_coef = pd.read_csv(_P / "coeficientes_glm.csv").set_index("variable")
_bp = pd.read_csv(_P / "barrios_prediccion.csv", index_col="barrio")
_cm = pd.read_csv(_P / "matriz_confusion.csv", index_col=0)
_val = pd.read_csv(_P / "validacion_temporal.csv").set_index("puntaje")
_res = pd.read_parquet(_P / "resultados_hex.parquet")
_ap = pd.read_parquet(_P / "aportes_hex.parquet").set_index("h3")
_tab = pd.read_parquet(_P / "hexagonos.parquet")
_est = pd.read_csv(_P / "estaciones_subte.csv")

_g = _R["modelo_elegido"]
_c = _R["clasificador_elegido"]
_d2 = lambda mod, df=_esp: df.loc[mod, "D² Poisson (promedio por partición)"]  # noqa: E731
_best_ale = _ale.sort_values("D² Poisson (promedio por partición)", ascending=False).index[0]
_glm, _xgb = "Regresión de Poisson (GLM)", "XGBoost"
_prof = _hp[_hp["modelo"] == _xgb]["max_depth"].dropna().astype(int)

comparacion = f"""
**Lectura: por qué elegimos este modelo y no otro.**

- Con la validación **aleatoria** el mejor es {_best_ale} (D² = {_n(_d2(_best_ale, _ale), 2)}): los modelos de árboles parecen
  imbatibles. Con la validación **espacial** el ranking cambia: {_best_ale} baja a {_n(_d2(_best_ale), 2)}. Los árboles aprenden la
  geografía de cada zona y, en una comuna que no vieron, esa memoria no les sirve (autocorrelación espacial). Si hubiéramos
  validado al azar, habríamos elegido mal.
- Gana el **{_g}**: D² = {_n(_R["d2_espacial_promedio"], 2)} ± {_n(_R["d2_espacial_desvio"], 2)} entre grupos de comunas (el más alto y
  el más estable) y el menor error típico: erra en promedio **{_n(_R["mae_espacial"], 1)} terminales por zona**, contra
  {_n(_esp.loc[_glm, "MAE"], 1)} de la regresión de Poisson sola y {_n(_esp.loc[_xgb, "MAE"], 1)} de XGBoost solo. Combina lo que cada parte hace
  bien: la regresión de Poisson extrapola mejor al microcentro, que tiene valores que los árboles no vieron, y XGBoost capta
  relaciones no lineales e interacciones entre variables.
- La regla simple "más vecinos, más cajeros" (Poisson solo con población) es **peor que predecir el promedio**
  (D² = {_n(_d2("Poisson solo con población"), 2)}): los cajeros siguen al comercio, no a los vecinos.
- XGBoost eligió árboles poco profundos (profundidad {_prof.min()} a {_prof.max()}) en las 5 particiones: con
  {_n(_R["hexagonos_modelados"])} zonas, los modelos simples generalizan mejor que los complejos."""

_mejor = _curva.loc[_curva["validación espacial"].idxmax()]
_ultima = _curva.iloc[-1]
sobreajuste = f"""
**Lectura.** El árbol mejora siempre en entrenamiento (D² = {_n(_ultima["entrenamiento"], 2)} con {int(_ultima["profundidad"])} niveles), pero
en comunas no vistas el mejor resultado está en **{int(_mejor["profundidad"])} niveles** (D² = {_n(_mejor["validación espacial"], 2)}) y después
empeora hasta {_n(_ultima["validación espacial"], 2)}: memoriza ruido. Es el sobreajuste de la clase 6, y es la razón por la que el
Random Forest y el boosting combinan muchos árboles en lugar de uno solo profundo.

El **OOB** del Random Forest da R² = {_n(_oob.loc["Random Forest (regresión)", "OOB"], 2)}, pero en comunas no vistas el R² es
{_n(_oob.loc["Random Forest (regresión)", "comunas no vistas"], 2)}. El OOB deja afuera filas al azar, que tienen vecinas parecidas
adentro de la muestra, así que con datos espaciales es optimista: es la misma trampa que la validación aleatoria. En
clasificación la diferencia es chica (AUC {_n(_oob.loc["Random Forest (clasificación)", "OOB"], 2)} OOB contra
{_n(_oob.loc["Random Forest (clasificación)", "comunas no vistas"], 2)}), porque si una zona tiene o no cajero depende sobre todo de lo
que hay en el propio hexágono."""

_ef = lambda v: _n(_coef.loc[v, "efecto"], 2)  # noqa: E731
_top_shap = ", ".join(_imp["nombre"].head(3).str.lower())
_top_gini = ", ".join(_gini["nombre"].head(3).str.lower())
importancia = f"""
**Lectura.** Las tres miradas coinciden en lo principal: **el comercio y el tipo de tejido urbano pesan más que la cantidad
de vecinos**. Según SHAP, lo que más mueve la predicción es: {_top_shap}. Según Gini del Random Forest: {_top_gini} (el
Random Forest le da más peso al subte porque, al no usar escala logarítmica, los cortes en pasajeros separan muy bien el
centro del resto).

En la regresión de Poisson, con lo demás fijo, subir un desvío las **parcelas comerciales** multiplica lo esperado por
{_ef("parcelas_comercio_k1")} y los **comercios** por {_ef("actividad_osm_k1")}; el **% de 65 años o más** por {_ef("pct_65_k1")}, coherente con un uso
más intensivo del efectivo en la población mayor; los **pasajeros de subte** por {_ef("pax_subte_habil_k1")}. En cambio, las **parcelas
residenciales** lo multiplican por {_ef("parcelas_residencial_k1")} (barrios de casas bajas) y el **% de hogares con NBI** por {_ef("pct_nbi_k1")}: a
igual demanda, las zonas vulnerables tienen menos cajeros. Algunas variables están correlacionadas entre sí (residentes y
parcelas residenciales, por ejemplo), así que conviene leerlas en conjunto."""

_falta = _bp[_bp["diferencia_link"] > 0].head(5)
_sin_link = _bp[(_bp["tiene_link"] == 0) & (_bp["corresponden_link"] >= 1.5)].sort_values("corresponden_link", ascending=False)
_sobra = _bp.sort_values("diferencia_link").head(3)
_lista = lambda d: "; ".join(f"**{b}** (le corresponden {_n(r['corresponden_link'])}, tiene {_n(r['tiene_link'])})" for b, r in d.iterrows())  # noqa: E731
_dom = int((_falta["cuota_link"] < 0.5).sum())
_dom_txt = ("En todos domina Banelco." if _dom == len(_falta) else f"En {_dom} de ellos domina Banelco." if _dom else "")
barrios = f"""
**Lectura.** Donde a Link más terminales le faltan según la demanda: {_lista(_falta)}. {_dom_txt}

Hay barrios donde Link **no tiene ninguna terminal** y por su demanda le corresponderían: {", ".join(f"**{b}** ({_n(r['corresponden_link'])})" for b, r in _sin_link.iterrows())}.

En el otro extremo está el microcentro: {_lista(_sobra)}. Ahí están las casas centrales de los bancos públicos de la red, y
el modelo no ve bien a la gente que trabaja en el centro (mide residentes, comercios y pasajeros de subte, pero no los
oficinistas que llegan en colectivo o tren), así que lo leemos como **sobreoferta a revisar**, no como cajeros que sobran."""

# Zonas de ejemplo: las mismas reglas que el notebook
_r = _res[_res["esperado_total_k1"].notna()].set_index("h3")
_zona_alta = _r[_r["term_link_k1"] == 0]["esperado_link_k1"].idxmax()
_uno = _r[(_r["term_total_k1"] == 0) & _r["esperado_link_k1"].between(0.5, 1.5) & (_r["tipo_zona"] == "Residencial denso")]
_zona_uno = _uno["poblacion_k1"].idxmax()
_zona_baja = _r[_r["poblacion_k1"] >= 15_000]["esperado_total_k1"].idxmin()
_X, _y, _gr, _mask = _m.preparar(_tab)
_Xh = _X.set_index(_tab.loc[_mask, "h3"])


def _cerca(h):
    d = np.hypot((_est["lat"] - _r.loc[h, "lat"]) * 111_000, (_est["lon"] - _r.loc[h, "lon"]) * 92_000)
    i = d.idxmin()
    return f"a unos {_n(round(d[i], -2))} m de la estación {_est.loc[i, 'estacion']} (línea {_est.loc[i, 'linea']})"


def _factores(h, signo):
    t = _ml.explicar_zona(_ap, _Xh, h, n=5)
    t = t[(t["variable"] != "resto") & ((t["aporte_log"] > 0) if signo > 0 else (t["aporte_log"] < 0))]
    return " y ".join(f"{r['nombre'].lower()} (×{_n(r['factor'], 2)})" for _, r in t.head(2).iterrows())


_za, _zu, _zb = _r.loc[_zona_alta], _r.loc[_zona_uno], _r.loc[_zona_baja]
zonas = f"""
**Lectura: tres zonas, tres decisiones.**

1. **Competir.** En {_za["barrio"]}, {_cerca(_zona_alta)}, hay {_n(_za["actividad_osm_k1"])} comercios en el área. Lo que más suma
   es {_factores(_zona_alta, 1)}. Le corresponden **{_n(_za["esperado_total_k1"])} terminales ({_n(_za["esperado_link_k1"])} de Link)** y
   tiene {_n(_za["term_total_k1"])}, **ninguna de Link**.
2. **Alcanza con uno.** En una zona residencial de {_zu["barrio"]}, {_cerca(_zona_uno)}, viven {_n(_zu["poblacion_k1"])} personas pero
   hay solo {_n(_zu["actividad_osm_k1"])} comercios y no pasa el subte: el flujo de gente es bajo. Le corresponden
   {_n(_zu["esperado_total_k1"], 1)} terminales, o sea **un solo cajero Link**, y hoy no tiene ninguno de ninguna red.
3. **Mucha gente, poco flujo.** En {_zb["barrio"]}, {_cerca(_zona_baja)}, viven {_n(_zb["poblacion_k1"])} personas pero hay solo
   {_n(_zb["actividad_osm_k1"])} comercios: lo que más resta es {_factores(_zona_baja, -1)}. Le corresponden
   {_n(_zb["esperado_total_k1"], 1)} terminales. Por demanda comercial no le corresponde un cajero; si el objetivo es acercar el
   efectivo a quienes más lo usan, entra por la estrategia de **cubrir** (notebook 03), no por la de competir."""

_tp, _fn = _cm.loc["Real: con cajero", "Predicho: con cajero"], _cm.loc["Real: con cajero", "Predicho: sin cajero"]
_fp, _tn = _cm.loc["Real: sin cajero", "Predicho: con cajero"], _cm.loc["Real: sin cajero", "Predicho: sin cajero"]
_op = _res[_res["deberia_tener_cajero"].fillna(False).astype(bool) & (_res["term_total"] == 0)]
_op_barrios = ", ".join(_op["barrio"].value_counts().head(4).index)
_arbol = _ml.arbol_explicativo(*_ml.preparar_clasificacion(_tab, _X, _mask))
_raiz = _ml.NOMBRES.get(_arbol.feature_names_in_[_arbol.tree_.feature[0]], "")
clasificacion = f"""
**Lectura.** Los clasificadores quedan muy cerca (AUC en comunas no vistas entre {_n(_espc["AUC"].min(), 2)} y {_n(_espc["AUC"].max(), 2)}) y el
árbol solo queda último, como se espera. Por la regla fijada (mayor AUC promedio por grupo de comunas) elegimos
**{_c}**: AUC = {_n(_R["auc_clasificador"], 2)}. Con el umbral de la tasa base acierta {_n(10 * _R["precision_clasificador"])} de cada 10 zonas que marca.

La matriz de confusión, en negocio: {_n(_tp)} zonas con cajero bien detectadas, {_n(_tn)} sin cajero bien detectadas,
{_n(_fn)} zonas con cajero y menos demanda de la esperada, y **{_n(_fp)} oportunidades**: zonas sin ningún cajero de ninguna red
que, por su demanda, deberían tener uno. En ellas viven **{_n(_R["residentes_zonas_sin_cajero_que_deberian"])} personas** y se concentran en
{_op_barrios}.

El árbol de 3 niveles, que no usamos para predecir sino para leer las reglas, empieza preguntando por
**{_raiz.lower()}**: es la variable que mejor separa zonas con y sin cajero, coherente con la importancia de la sección 4."""

_vb = _val.loc["Brecha total del modelo (2017)", "AUC"]
_vc = _val.loc["Probabilidad de cajero (clasificador)", "AUC"]
_vr = _val.loc["Solo residentes", "AUC"]
validacion = f"""
**Lectura.** De los {_R["cajeros_osm_2026"]} cajeros de OpenStreetMap, {_R["cajeros_nuevos_osm"]} no tienen ningún cajero del listado 2017 a
menos de 150 m: muy probablemente son nuevos. Cayeron, en promedio, en el **percentil {_n(100 * _R["percentil_cajeros_nuevos"])}** de la
brecha calculada con datos de 2017 (al azar se esperaría el 50; p = {_n(_R["p_valor_cajeros_nuevos"], 3)}). El AUC de la brecha es
{_n(_vb, 2)} y el de la probabilidad del clasificador, {_n(_vc, 2)}, contra {_n(_vr, 2)} de mirar solo los residentes.

Es una **evidencia a favor, pero modesta**: son pocos cajeros y OpenStreetMap no está completo. La leemos así: el mercado
tendió a instalar cajeros donde los modelos marcaban faltante, y no al azar."""

conclusion = f"""
- **Machine learning predictivo:** comparamos 7 modelos de regresión y 4 de clasificación con validación cruzada espacial
  anidada. Ganan el **{_g}** (D² = {_n(_R["d2_espacial_promedio"], 2)}, MAE = {_n(_R["mae_espacial"], 1)} terminales por zona) y
  **{_c}** (AUC = {_n(_R["auc_clasificador"], 2)}).
- **Diagnóstico:** los cajeros siguen al comercio y al flujo de gente, no a la cantidad de vecinos; las zonas vulnerables tienen
  menos cajeros que los que les corresponden por su demanda.
- **Resultado de negocio:** a Link le faltan terminales sobre todo en {", ".join(_falta.index[:3])}; hay barrios sin ningún Link a los
  que les corresponde alguno ({", ".join(_sin_link.index[:3])}); y {_n(_fp)} zonas sin ningún cajero deberían tener uno.
- **Validación:** los cajeros instalados después de 2017 aparecieron donde los modelos marcaban faltante (p = {_n(_R["p_valor_cajeros_nuevos"], 3)}).
- **Siguiente paso:** el notebook 03 convierte esto en una decisión: cuántos cajeros sumar y dónde, para llegar a más gente."""
