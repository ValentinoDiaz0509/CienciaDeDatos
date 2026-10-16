# Textos del notebook 03 (segmentación y optimización). Se ejecuta desde nb03_optimizacion.py con RAIZ definido.
import json

import pandas as pd

from src import estilo as _e

_n = _e.num_es
_P = RAIZ / "data/processed"
_R = json.loads((_P / "resumen.json").read_text(encoding="utf-8"))
_perfil = pd.read_csv(_P / "clusters_perfil.csv").set_index("tipo_zona")
_ex = pd.read_csv(_P / "recomendaciones_exactas_20.csv")
_cam = _R["metodo_distancia"] == "caminando"
_dist = "caminando por las calles" if _cam else "en línea recta"
_zonas_ex = _ex["tipo_zona"].value_counts()
_bar_tipo = lambda z: ", ".join(_ex[_ex["tipo_zona"] == z]["barrio"].value_counts().index[:3])  # noqa: E731
_res_d, _corr, _vul = "Residencial denso", "Corredores comerciales con subte", "Mayor vulnerabilidad social"


def _p(zona, col):
    return _perfil.loc[zona, col] if zona in _perfil.index else float("nan")


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
_igualar = (f"Con **{_R['cajeros_para_igualar_banelco']} cajeros** bien ubicados, Link igualaría la cobertura que Banelco tiene hoy."
            if _R.get("cajeros_para_igualar_banelco") else "")
optimizacion = f"""
**Lectura.** Medido {_dist}, hoy el **{_n(100 * _R["cobertura_link"])} %** de los vecinos tiene un Link a menos de 500 m,
contra el **{_n(100 * _R["cobertura_banelco"])} %** de Banelco: unas **{_n(_R["residentes_sin_link"])} personas** no tienen
un Link a distancia caminable.

- Con **20 cajeros** en las ubicaciones óptimas, Link suma **{_n(_n20)} vecinos** y pasa al
  **{_n(100 * _R["cobertura_link_con_20"], 1)} %**. Los primeros 10 cajeros suman {_n(_R["nuevos_10"])} vecinos y los siguientes 40,
  {_n(_R["nuevos_50"] - _R["nuevos_10"])}: cada cajero suma un poco menos que el anterior, pero la curva se aplana despacio porque
  hay mucha ciudad sin un Link cerca. {_igualar}
- El **goloso** queda a menos del **{_n(_R["goloso_max_diferencia_pct"], 1)} %** del óptimo exacto en todos los casos, y el
  exacto se resuelve en menos de {_n(max(1, _R["segundos_exacto_max"]), 0)} segundo: se puede usar en vivo en la app.
- Las ubicaciones óptimas están en zonas **residenciales densas** ({_n(_zonas_ex.get(_res_d, 0))} de 20; sobre todo
  {_bar_tipo(_res_d)}) y de **mayor vulnerabilidad social** en el sur ({_n(_zonas_ex.get(_vul, 0))} de 20;
  {_bar_tipo(_vul)}). Solo {_n(_zonas_ex.get(_corr, 0))} cae en un corredor comercial.

Esta es la oportunidad de **cobertura**, distinta de la competitiva del notebook 02: no se trata de competir donde ya hay
cajeros sino de llegar adonde no hay ninguno cerca."""

conclusion = f"""
**Qué conviene hacer (nivel prescriptivo).** Dos estrategias que se pueden combinar:

1. **Competir** donde la demanda justifica más terminales Link de las que hay (la brecha del notebook 02): sobre todo
   corredores comerciales del norte, dominados por Banelco.
2. **Cubrir** los barrios residenciales y el sur, donde {_n(_R["residentes_sin_link"])} vecinos no tienen un Link a distancia
   caminable. Con 20 cajeros bien ubicados, Link pasa del {_n(100 * _R["cobertura_link"])} % al
   {_n(100 * _R["cobertura_link_con_20"], 1)} % de cobertura y la diferencia con Banelco baja de
   {_n(100 * (_R["cobertura_banelco"] - _R["cobertura_link"]))} a {_n(100 * (_R["cobertura_banelco"] - _R["cobertura_link_con_20"]))} puntos.

**Cómo usarlo:** la gerencia elige la estrategia, el simulador de la app devuelve las ubicaciones y su impacto, y con el
listado actual de cajeros se recalcula todo en segundos."""
