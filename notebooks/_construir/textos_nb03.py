# Textos del notebook 03. Se ejecuta desde nb03_horario.py con RAIZ definido.
import json

import pandas as pd

from src import estilo as _e
from src import horario as _ho

_n = _e.num_es
_P = RAIZ / "data/processed"
_R = json.loads((_P / "resumen.json").read_text(encoding="utf-8"))
_hor = pd.read_csv(_P / "subte_horario.csv")
_est = pd.read_csv(_P / "estaciones_subte.csv")
_curva = _hor.pivot_table(index="hora", columns="tipo_dia", values="pasajeros", aggfunc="sum").reindex(_ho.HORAS).fillna(0)
_hab = _curva["habil"]
_pm, _pt = _R["hora_pico_manana"], _R["hora_pico_tarde"]
_pct_pico = (_hab.loc[[7, 8, 17, 18]].sum()) / _hab.sum()
_finde_ratio = _curva["fin_de_semana"].sum() / _hab.sum()
_perf = _est["perfil"].value_counts()
_orig = _est[_est["perfil"].str.startswith("Origen", na=False)]
_dest = _est[_est["perfil"].str.startswith("Destino", na=False)]
_sin = _est[~_est["link_cerca"]]
_sin_orig = int(_sin["perfil"].str.startswith("Origen", na=False).sum())
_sin_lin = _sin["linea"].value_counts()
_con = len(_est) - len(_sin)
_casi = "Casi todas" if _con / len(_est) >= 0.95 else "La gran mayoría de"
_comunes = _R.get("sitios_comunes_con_pax_manana")
_cam = _R["metodo_distancia"] == "caminando"

curva = f"""
**Lectura.** Un día hábil promedio entran {_n(_hab.sum())} personas al subte, con dos picos: **{_pm}:00** a la mañana y
**{_pt}:00** a la tarde. Las cuatro horas pico (7, 8, 17 y 18 h) concentran el {_n(100 * _pct_pico)} % de las entradas del
día. El fin de semana la curva es plana y el volumen cae al {_n(100 * _finde_ratio)} % del de un día hábil."""

calor = """
**Lectura.** Cada fila es una estación y cada columna una hora: el color es qué parte de las entradas del día ocurre en
esa hora. Se ven dos dibujos: estaciones que se "encienden" temprano (la gente sale de su casa) y estaciones que se
encienden a la tarde (la gente vuelve del trabajo, típicamente en el centro)."""

tipos = f"""
**Lectura.** El K-Means separa {len(_perf)} tipos de estación (la silueta más alta se da con k = {len(_perf)}):
- **{_perf.index[0]}**: {_perf.iloc[0]} estaciones;
- **{_perf.index[1]}**: {_perf.iloc[1]} estaciones.

Las estaciones de destino se concentran en el centro y los corredores de oficinas; las de origen están en los barrios.
Es la misma división de la ciudad que aparece en los tipos de zona del notebook 02, pero obtenida con otra fuente y otra
técnica: un buen chequeo de consistencia."""

cobertura = f"""
**Lectura.** {_casi} las estaciones ya tienen un Link a menos de 500 m (medido {"caminando" if _cam else "en línea recta"}):
{_con} de {len(_est)}. De las {len(_sin)} que no, {_sin_orig} son estaciones de **origen** (barrios), y
{_sin_lin.iloc[0]} están sobre la línea {_sin_lin.index[0]}. Suman {_n(_sin["pax_habil"].sum())} entradas por día hábil,
el {_n(100 * _sin["pax_habil"].sum() / _est["pax_habil"].sum(), 1)} % del subte. El subte, entonces, **no** es donde más
falta Link: la mayoría de la gente que viaja ya pasa cerca de un cajero. El faltante grande está en los barrios."""

_cambian = 20 - (_comunes or 0)
optimizacion = f"""
**Lectura.** Sumar a los pasajeros de la hora pico de la mañana deja **{_comunes} de los 20 sitios** iguales: lo que
define dónde poner cajeros son los vecinos sin un Link cerca, porque la mayoría de las estaciones ya está cubierta.
{"El sitio que cambia está" if _cambian == 1 else "Los sitios que cambian están"} en la tabla de arriba. La recomendación
es robusta: no depende de cuánto pesemos a los pasajeros."""

conclusion = f"""
- La demanda de paso tiene horario: **{_pm}:00** y **{_pt}:00** en día hábil. Un cajero cerca de una estación de origen ve
  su demanda a primera hora; uno en el centro, al final de la tarde. Es un dato útil para la **Gerencia Técnica**:
  programar la carga de efectivo y el mantenimiento **antes** de esos picos (hipótesis a validar con las transacciones
  por hora que Red Link sí tiene).
- {_casi} las estaciones de subte ya tienen un Link cerca ({_con} de {len(_est)}): la oportunidad grande de cobertura está
  en los **barrios**, no en el transporte.
- La división origen/destino del subte confirma, con otra fuente, la ciudad de dos velocidades del análisis de zonas."""
