# Textos del notebook 01. Se ejecuta desde nb01_eda.py con RAIZ definido; los números salen de los datos.
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from src import estilo as _e
from src import features as _f

_n = _e.num_es
_B, _P = RAIZ / "data/base", RAIZ / "data/processed"
_caj = _f.cargar_cajeros(_B)
_hex = pd.read_parquet(_P / "resultados_hex.parquet")
_R = json.loads((_P / "resumen.json").read_text(encoding="utf-8"))
_mod = _hex[_hex["esperado_total_k1"].notna()]
_censo = pd.read_parquet(_B / "censo2022-caba-largo.parquet")
_pob = _censo[_censo["codigo_variable"] == "PERSONA_P02"]["conteo"].sum()
_osm = pd.read_csv(_B / "osm-caba-puntos.csv")
_osm_atm = _osm[_osm["categoria"] == "cajero"]
_mol = pd.read_csv(_B / "molinetes-2025-estacion-hora.csv")
_dias = _mol.groupby("tipo_dia")["dias"].first().sum()
_viejos = _caj["banco"].isin(["CitiBank", "BBVA Banco Francés", "Banco Santander Río", "HSBC Bank Argentina"])
_sp = _mod[["term_total_k1", "actividad_osm_k1", "poblacion_k1", "pax_subte_habil_k1", "pct_nbi_k1"]].corr(method="spearman")["term_total_k1"]
_com = (_hex.groupby("comuna").agg(r=("poblacion", "sum"), l=("term_link", "sum"), b=("term_banelco", "sum"))
        .assign(l10=lambda d: d["l"] / d["r"] * 1e4, b10=lambda d: d["b"] / d["r"] * 1e4, cuota=lambda d: d["l"] / (d["l"] + d["b"])))
_ceros = (_mod["term_total"] == 0).mean()
_ceros_link = (_mod["term_link"] == 0).mean()
_centro = _mod["dist_centro_km"] < 3
_pct_term_centro = _mod.loc[_centro, "term_total"].sum() / _mod["term_total"].sum()
_pct_pob_centro = _mod.loc[_centro, "poblacion"].sum() / _mod["poblacion"].sum()

oportunidad = f"""
**Lectura.** La métrica de *oportunidad* es fecha de uso − fecha del dato: usamos el listado en octubre de 2026 y el
relevamiento es de alrededor de 2017, **unos nueve años**. {_n(_viejos.sum())} de {_n(len(_caj))} ubicaciones
({_n(100 * _viejos.mean())} %) están a nombre de entidades que ya no operan con ese nombre. El portal informa una fecha de
actualización reciente porque se actualizan los metadatos, no el relevamiento.

**Qué hacemos con esto:** (1) leemos la oferta como una foto de ≈2017 y lo decimos en la defensa; (2) validamos contra los
cajeros mapeados en OpenStreetMap en 2026 (notebook 02, §7); (3) la app permite cargar el listado actual, que Red Link sí
tiene. El resto de las dimensiones del listado da bien: está completo, es válido y no tiene duplicados."""

calidad_otras = f"""
**Lectura.**
- **Censo:** la base por radio suma {_n(_pob)} habitantes, el {_n(100 * _pob / 3_121_707, 1)} % del total oficial de CABA
  (3.121.707). La diferencia es chica; posiblemente corresponde a población que no se asigna a un radio (por ejemplo, en
  situación de calle).
- **Molinetes:** hay datos para {_n(_dias)} de los 365 días de 2025 ({_n(100 * _dias / 365, 1)} %). Además se descartan
  filas de prueba y estaciones cargadas en líneas que no les corresponden. Los feriados que caen en día de semana cuentan
  como días hábiles: el promedio hábil queda levemente subestimado.
- **Usos del suelo:** el CSV oficial **no trae coordenadas**; las tomamos del shapefile (99,99 % de las parcelas ubicadas).
- **OpenStreetMap:** tiene {_n(len(_osm_atm))} cajeros contra {_n(len(_caj))} del listado oficial: está **incompleto**,
  así que no sirve para medir la oferta, pero sí como referencia actual de dónde aparecieron cajeros nuevos."""

tipo1 = f"""
**Lectura.** Los conteos son muy asimétricos: el **{_n(100 * _ceros)} %** de los hexágonos no tiene ninguna terminal y
el **{_n(100 * _ceros_link)} %** no tiene ninguna Link, pero el máximo llega a {_n(_mod['term_total'].max())}. La media
describe una zona que casi no existe. Dos consecuencias para el modelado: medir en el **área de influencia** (hexágono +
vecinos) para suavizar, y usar modelos de **conteo** (Poisson) en lugar de una regresión lineal común."""

tipo2 = f"""
**Lectura.** Los residentes se reparten de forma bastante pareja (la ciudad es densa casi en todas partes), mientras que
las terminales tienen una cola larga: pocas zonas concentran muchísimas. Por ubicación, las dos redes tienen una mediana
de {_n(_caj.groupby("red")["terminales"].median().max())} terminales; Link tiene más casos extremos (hasta
{_n(_caj.loc[_caj["red"] == "LINK", "terminales"].max())} terminales en un mismo punto, típicamente sucursales grandes de
bancos públicos). La caja muestra la mediana y el rango intercuartil; los puntos sueltos son los casos extremos."""

tipo3 = f"""
**Lectura.** Las terminales se asocian más con el **comercio** (ρ = {_n(_sp['actividad_osm_k1'], 2)}) que con los
**residentes** (ρ = {_n(_sp['poblacion_k1'], 2)}) o los **pasajeros de subte** (ρ = {_n(_sp['pax_subte_habil_k1'], 2)}),
y prácticamente nada con el % de hogares con NBI (ρ = {_n(_sp['pct_nbi_k1'], 2)}).

Por comuna, la diferencia es enorme: la Comuna 1 (centro) tiene {_n(_com.loc[1, 'l10'], 1)} terminales Link cada 10.000
vecinos, y la Comuna {_com['l10'].idxmin()} apenas {_n(_com['l10'].min(), 2)}. Llama la atención que la participación de
Link es mayor justamente en el sur (Comuna {_com['cuota'].idxmax()}: {_n(100 * _com['cuota'].max())} % de las terminales),
donde pesan los bancos públicos."""

tipo4 = f"""
**Lectura.** En los dos paneles la relación es positiva, pero es más fuerte con el comercio. El centro (naranja) es
un caso aparte: concentra el {_n(100 * _pct_term_centro)} % de las terminales con solo el {_n(100 * _pct_pob_centro)} %
de los vecinos, porque la gente que usa esos cajeros trabaja ahí pero no vive ahí.

En clase vimos que una correlación global puede invertirse al separar grupos (paradoja de Simpson). Lo verificamos: acá
**no** pasa. Dentro de cada grupo la relación mantiene el signo, así que la lectura global es confiable."""

mapas = """
**Lectura.** La población se reparte por toda la ciudad, con más densidad en el eje norte (Palermo, Belgrano), en el
centro geográfico (Balvanera, Almagro, Caballito) y en algunos barrios populares del sur.
Los cajeros, en cambio, se amontonan en el microcentro y sobre las avenidas comerciales. Esa diferencia entre dónde vive
la gente y dónde están los cajeros es justamente lo que el proyecto busca cuantificar."""

_solo_link = _com[_com["l"] > _com["b"]].index.tolist()
_pb = _hex.groupby("barrio").agg(r=("poblacion", "sum"), l=("term_link", "sum"), pax=("pax_subte_habil", "sum"))
_pb["l10"] = _pb["l"] / _pb["r"] * 1e4
_lug = _pb.loc["Villa Lugano"] if "Villa Lugano" in _pb.index else None
tipo5 = f"""
**Lectura.** En el mapa de calor, Banelco tiene más terminales por vecino que Link en todas las comunas salvo la
{", ".join(map(str, _solo_link)) or "—"}. En el gráfico de burbujas, los barrios más poblados (a la derecha) quedan cerca
o por debajo de la mediana de terminales Link por vecino ({_n(_pb["l10"].median(), 1)}). Villa Lugano, uno de los más
grandes y sin subte, tiene apenas {_n(_lug["l10"], 1) if _lug is not None else "—"}. Esos barrios grandes y lejos del
centro son los candidatos naturales para sumar cobertura."""

hallazgos = f"""
1. **La oferta es una foto de ≈2017.** La dimensión *oportunidad* falla; el método vale igual, pero los resultados se leen
   como diagnóstico de esa oferta y la app permite cargar el listado actual.
2. **Los cajeros siguen al comercio, no a los vecinos.** La demanda hay que medirla con varias fuentes, no solo con
   población: por eso cruzamos censo, usos del suelo, OpenStreetMap y subte.
3. **Los conteos tienen muchos ceros y cola larga** ({_n(100 * _ceros)} % de hexágonos sin terminales): modelos de Poisson y
   áreas de influencia en lugar de hexágonos sueltos.
4. **Link es relativamente más fuerte en el sur y más débil en el norte comercial**, donde Banelco domina: hay dos
   oportunidades distintas, competir donde hay demanda y cubrir donde no hay ningún cajero cerca.
5. **OpenStreetMap está incompleto para cajeros**, pero sirve como validación independiente de dónde aparecieron
   cajeros nuevos después de 2017."""
