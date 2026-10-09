# Textos del notebook 02. Se ejecuta desde nb02_modelos.py con RAIZ definido; los números salen de data/processed.
import json

import pandas as pd

from src import estilo as _e

_n = _e.num_es
_P = RAIZ / "data/processed"
_R = json.loads((_P / "resumen.json").read_text(encoding="utf-8"))
_met = pd.read_csv(_P / "metricas_modelos.csv")
_esp = _met[_met["validación"].str.startswith("espacial")].set_index("modelo")
_ale = _met[_met["validación"] == "aleatoria"].set_index("modelo")
_coef = pd.read_csv(_P / "coeficientes_glm.csv").set_index("variable")
_perfil = pd.read_csv(_P / "clusters_perfil.csv").set_index("tipo_zona")
_recs = pd.read_csv(_P / "recomendaciones.csv")
_ex = pd.read_csv(_P / "recomendaciones_exactas_20.csv")
_comp = pd.read_csv(_P / "goloso_vs_exacto.csv")
_res = pd.read_parquet(_P / "resultados_hex.parquet")
_cam = _R["metodo_distancia"] == "caminando"
_dist = "caminando por las calles" if _cam else "en línea recta"

_g, _best_ale = _R["modelo_elegido"], _R["modelo_mejor_aleatoria"]
_pb = (_res.groupby("barrio").agg(r=("poblacion", "sum"), l=("term_link", "sum"), b=("term_banelco", "sum"),
                                  f=("brecha_link", lambda s: s.clip(lower=0).sum() / 7))
       .sort_values("f", ascending=False))
_top_brecha = _pb.head(4)
_cuota = lambda b: _pb.loc[b, "l"] / (_pb.loc[b, "l"] + _pb.loc[b, "b"])
_ef = lambda v: _n(_coef.loc[v, "efecto"], 2)
_zonas_ex = _ex["tipo_zona"].value_counts()
_barrios_ex = _ex["barrio"].value_counts()


def _p(zona, col):
    return _perfil.loc[zona, col] if zona in _perfil.index else float("nan")


_res_d, _corr, _vul = "Residencial denso", "Corredores comerciales con subte", "Mayor vulnerabilidad social"

comparacion = f"""
**Lectura: por qué elegimos este modelo y no otro.**

- Con la validación **aleatoria** el ranking lo ganan los modelos complejos: {_best_ale} llega a D² =
  {_n(_ale.loc[_best_ale, "D² Poisson"], 2)}. Con la validación **espacial** el orden se da vuelta: {_best_ale} cae a
  {_n(_esp.loc[_best_ale, "D² Poisson"], 2)} y gana la **{_g}** (D² = {_n(_esp.loc[_g, "D² Poisson"], 2)}). Los modelos de
  árboles aprenden la geografía de cada zona; cuando les toca una comuna que no vieron, esa memoria no les sirve.
- El **ensamble** definido de antemano (GLM + Boosting) queda en {_n(_esp.loc["Ensamble (Poisson GLM + Boosting)", "D² Poisson"], 2)}:
  no mejora al GLM solo, así que por parsimonia nos quedamos con el modelo simple.
- **Solo con población** el modelo es peor que el promedio (D² = {_n(_esp.loc["Poisson solo con población", "D² Poisson"], 2)}):
  confirma el hallazgo del EDA de que los cajeros no siguen a los vecinos.
- Además del desempeño, el GLM tiene una ventaja de negocio: sus coeficientes se explican en una frase a la gerencia.

El D² espacial de {_n(_R["d2_espacial"], 2)} significa que el modelo explica cerca de dos tercios de la variación de las
terminales entre zonas que nunca vio, solo con variables de demanda."""

interpretacion = f"""
**Lectura.** Con las demás variables fijas, subir un desvío estándar en:
- **parcelas comerciales** multiplica las terminales esperadas por {_ef("parcelas_comercio_k1")}, y **comercios de OSM** por
  {_ef("actividad_osm_k1")}: el comercio es el principal imán de cajeros;
- **% de 65 años o más** las multiplica por {_ef("pct_65_k1")}, algo coherente con un uso más intensivo del efectivo en la
  población mayor;
- **pasajeros de subte** por {_ef("pax_subte_habil_k1")} y **oficinas** por {_ef("parcelas_oficinas_k1")};
- **parcelas residenciales** por {_ef("parcelas_residencial_k1")}: los barrios de casas bajas tienen menos cajeros;
- **% de hogares con NBI** por {_ef("pct_nbi_k1")}: a igual comercio y población, las zonas vulnerables tienen menos oferta.

Ojo al leerlos: algunas variables están correlacionadas entre sí (residentes y parcelas residenciales, por ejemplo), así
que conviene interpretarlas en conjunto. La importancia por permutación coincide en lo principal: tipo de tejido urbano y
comercio primero, transporte y perfil de la población después."""

_desc = []
for _b in _top_brecha.index[:3]:
    _c = _cuota(_b)
    _desc.append(f"**{_b}** (faltan unas {_n(_top_brecha.loc[_b, 'f'])}; Link tiene el {_n(100 * _c)} % de las terminales"
                 + (", domina Banelco)" if _c < 0.4 else ", reparto parejo pero mucha demanda)"))
brecha = f"""
**Lectura.** La brecha se concentra en los **corredores comerciales del norte y el centro**, donde la demanda predice más
terminales Link de las que hay: {"; ".join(_desc)}. En azul aparece el microcentro, donde Link tiene más de lo que
predice la demanda (allí están las casas centrales de los bancos públicos de la red).

Esta es la oportunidad **competitiva**: zonas con mucha demanda donde Link tiene menos presencia de la que le corresponde."""

clusters = f"""
**Lectura.** Con k = {_R["k_clusters"]} (silueta = {_n(_R["silueta"], 2)}) aparecen cuatro ciudades distintas:
- **{_corr}:** {_n(_p(_corr, "residentes"))} vecinos y casi todo el flujo del subte; {_n(_p(_corr, "terminales_cada_10mil"), 1)}
  terminales cada 10.000 vecinos ({_n(_p(_corr, "link_cada_10mil"), 1)} de Link).
- **{_res_d}:** la mayor parte de la ciudad, con {_n(_p(_res_d, "residentes"))} vecinos pero solo
  {_n(_p(_res_d, "terminales_cada_10mil"), 1)} terminales cada 10.000 ({_n(_p(_res_d, "link_cada_10mil"), 1)} de Link).
- **{_vul}:** el sur, con NBI mediano de {_n(100 * _p(_vul, "pct_nbi_k1"))} % y {_n(_p(_vul, "terminales_cada_10mil"), 1)}
  terminales cada 10.000 vecinos, la menor oferta por habitante.
- **Zonas sin vecinos:** parques, puerto y playones; quedan como fondo.

Un vecino de un barrio residencial tiene, por habitante, menos de un tercio de las terminales que uno de un corredor comercial."""

_n20 = _R["nuevos_20_exacto"]
optimizacion = f"""
**Lectura.** Medido {_dist}, hoy el **{_n(100 * _R["cobertura_link"])} %** de los vecinos tiene un Link a menos de 500 m,
contra el **{_n(100 * _R["cobertura_banelco"])} %** de Banelco: unas **{_n(_R["residentes_sin_link"])} personas** no tienen
un Link a distancia caminable.

- Con **20 cajeros** en las ubicaciones óptimas, Link suma **{_n(_n20)} vecinos** y pasa al
  **{_n(100 * _R["cobertura_link_con_20"], 1)} %**. La curva muestra rendimientos decrecientes: los primeros 10 cajeros suman
  {_n(_R["nuevos_10"])} vecinos y los siguientes 40, {_n(_R["nuevos_50"] - _R["nuevos_10"])}.
- El **goloso** queda a menos del **{_n(_R["goloso_max_diferencia_pct"], 1)} %** del óptimo exacto en todos los casos, y el
  exacto se resuelve en menos de {_n(max(1, _R["segundos_exacto_max"]), 0)} segundo: se puede usar en vivo en la app.
- Las ubicaciones óptimas están en zonas **residenciales densas** ({_n(_zonas_ex.get(_res_d, 0))} de 20) y de **mayor
  vulnerabilidad social** en el sur ({_n(_zonas_ex.get(_vul, 0))} de 20), lejos del centro: {", ".join(_barrios_ex.index[:5])}.

Esta es la oportunidad de **cobertura**, distinta de la competitiva: no se trata de competir donde ya hay cajeros sino de
llegar adonde no hay ninguno cerca."""

validacion = f"""
**Lectura.** De los {_R["cajeros_osm_2026"]} cajeros mapeados en OpenStreetMap, {_R["cajeros_nuevos_osm"]} no tienen
ningún cajero del listado 2017 a menos de 150 m: muy probablemente son **nuevos**. Cayeron en zonas que en 2017 tenían,
en promedio, más brecha que el {_n(100 * _R["percentil_cajeros_nuevos"])} % de la ciudad (p = {_n(_R["p_valor_cajeros_nuevos"], 3)}
frente a zonas elegidas al azar). El AUC de la brecha es {_n(_R["auc_validacion"], 2)}, contra 0,51 de mirar solo la población.

Es una **evidencia a favor, pero modesta**: son pocos cajeros y OpenStreetMap no está completo. La leemos así: el mercado
tendió a instalar cajeros donde el modelo marcaba faltante, y no al azar."""

conclusion = f"""
**Respuesta a la hipótesis.** Sí: en CABA hay zonas con demanda y pocos cajeros Link, y se pueden detectar y priorizar con
datos públicos. Aparecen **dos oportunidades distintas**:

1. **Competir** en los corredores comerciales del norte y el centro ({", ".join(_top_brecha.index[:3])}), donde la
   demanda justifica más terminales Link de las que hay.
2. **Cubrir** los barrios residenciales del oeste y del sur, donde {_n(_R["residentes_sin_link"])} vecinos no tienen un
   Link a distancia caminable. Con 20 cajeros bien ubicados, Link pasa del {_n(100 * _R["cobertura_link"])} % al
   {_n(100 * _R["cobertura_link_con_20"], 1)} % de cobertura y la diferencia con Banelco baja de
   {_n(100 * (_R["cobertura_banelco"] - _R["cobertura_link"]))} a {_n(100 * (_R["cobertura_banelco"] - _R["cobertura_link_con_20"]))} puntos.

**Cómo usarlo:** la gerencia elige la estrategia (competir o cubrir), el simulador de la app devuelve las ubicaciones y su
impacto, y con el listado actual de cajeros se recalcula todo en segundos.

**Limitaciones:** la oferta es de ≈2017; la demanda se estima sin transacciones reales; la brecha compara contra el
promedio del mercado (si todo el mercado desatiende una zona, la brecha no lo ve; por eso la optimización mide cobertura y
no brecha)."""
