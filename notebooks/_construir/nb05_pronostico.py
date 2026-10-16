"""Genera notebooks/05_pronostico_subte.ipynb (correr: python notebooks/_construir/nb05_pronostico.py)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SALIDA = Path(__file__).resolve().parents[1] / "05_pronostico_subte.ipynb"

cells = [
    md("""# 05 · Pronóstico: ¿cuánta gente va a pasar por el subte en los próximos meses?

**TPO Ciencia de Datos · ¿Dónde faltan cajeros Link en CABA?**

El notebook 02 predice **dónde**: cuántos cajeros le corresponden a cada zona. Este predice **cuándo**: cuánta gente va a
entrar al subte cada día de los próximos meses. Son las dos caras del análisis predictivo:

| | Notebook 02 | Notebook 05 |
|---|---|---|
| Pregunta | ¿Cuántos cajeros le corresponden a cada zona? | ¿Cuánta gente va a pasar por las estaciones cada día? |
| Datos | una foto de la ciudad (censo, usos del suelo, OpenStreetMap) | la serie diaria de los molinetes, 2022 a 2026 (SBASE) |
| Modelos | regresión de conteos: árboles, Random Forest, XGBoost | regresión sobre el calendario: Random Forest y XGBoost |
| Validación | espacial: comunas que el modelo no vio | hacia adelante: meses que el modelo no vio |

**Para qué le sirve a Red Link**

- **Gerencia Técnica:** planificar la carga de efectivo y el mantenimiento de los cajeros cerca de las estaciones: cuándo
  va a haber más gente y en qué días y meses conviene hacer los trabajos.
- **Gerencia Comercial:** cuánta gente va a pasar por las estaciones que hoy no tienen un Link a distancia caminable.

**Cómo correrlo.** Necesita la serie diaria en `data/base/` (la arma el notebook 00c). En Colab no hace falta nada más:
si no está, este notebook la baja (tarda entre 20 y 40 minutos la primera vez)."""),
    code("""# Preparar el entorno: en Colab clona el repo y, si falta, baja la serie diaria
import json, subprocess, sys, shutil, zipfile
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    RAIZ = Path("/content/CienciaDeDatos")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/ValentinoDiaz0509/CienciaDeDatos.git", str(RAIZ)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "holidays", "xgboost", "h3"], check=True)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import estilo as e, pronostico as pr
from src.features import DIR_BASE

import matplotlib.dates as mdates

e.mpl_estilo()
FIG = RAIZ / "docs/figuras"; FIG.mkdir(parents=True, exist_ok=True)
PROC = RAIZ / "data/processed"
MES_CORTO = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def eje_meses(ax, cada=1):   # marcas mensuales en castellano: "abr 26"
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=cada))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{MES_CORTO[mdates.num2date(x).month - 1]} {mdates.num2date(x):%y}"))

if not pr.hay_serie(DIR_BASE):
    zip_serie = RAIZ / "data/serie_subte.zip"
    if not zip_serie.exists() and EN_COLAB:
        from src import descarga
        zip_serie = descarga.descargar_serie_subte()
    if not zip_serie.exists():
        raise FileNotFoundError("Falta la serie diaria del subte: correr notebooks/00c_serie_diaria_subte.ipynb en Colab "
                                "y dejar serie_subte.zip en data/ (o sus dos archivos en data/base/).")
    with zipfile.ZipFile(zip_serie) as z:
        for nombre in (pr.ARCHIVO_DIARIO, pr.ARCHIVO_HORA):
            if nombre in z.namelist():
                z.extract(nombre, DIR_BASE)
print("Serie diaria:", DIR_BASE / pr.ARCHIVO_DIARIO)"""),
    md("""## 1. La serie

Sumamos las entradas de todas las estaciones para tener **pasajeros por día en toda la red**. Antes de modelar marcamos los
**días anómalos**: los que tienen menos de la mitad de los pasajeros de un día igual en las semanas de alrededor (paros,
cortes de servicio o archivos incompletos). No se pueden predecir con el calendario, así que no se usan para entrenar ni
para medir el error. Los feriados no cuentan como anomalía: son justamente lo que el modelo tiene que aprender."""),
    code("""diario, hora = pr.cargar(DIR_BASE)
s = pr.serie_red(diario)
anomalia = pr.marcar_anomalias(s)
fin = s.dropna().index.max()
print(f"Desde {s.dropna().index.min():%d/%m/%Y} hasta {fin:%d/%m/%Y}: {e.num_es(s.notna().sum())} días con datos, "
      f"{e.num_es(s.isna().sum())} sin datos y {e.num_es((anomalia & s.notna()).sum())} anómalos")
reciente = s[(s.index > fin - pd.DateOffset(years=1)) & (s.index.dayofweek < 5)]
print(f"Estaciones: {diario.groupby(['linea', 'estacion']).ngroups} · "
      f"día hábil típico del último año: {e.num_es(reciente.median())} entradas")
pd.DataFrame({"pasajeros": s[anomalia & s.notna()]}).assign(dia=lambda d: d.index.day_name()).tail(10)"""),
    code("""fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(s.index, s / 1e3, color=e.EJE, linewidth=0.6, label="Pasajeros por día")
semanal = s.where(~anomalia).rolling(28, center=True, min_periods=14).mean()
ax.plot(semanal.index, semanal / 1e3, color=e.LINK, linewidth=2.2, label="Promedio de 4 semanas")
an = s[anomalia & s.notna()]
ax.scatter(an.index, an / 1e3, s=16, color=e.DIVERGENTE[-2], zorder=3, label="Días anómalos (paros, cortes)")
ax.set(title="Entradas al subte por día, toda la red", ylabel="miles de pasajeros")
ax.legend(loc="upper left"); e.ejes_es(ax, y=0); fig.tight_layout(); fig.savefig(FIG / "pron_serie.png"); plt.show()"""),
    md("""### Los patrones que tiene que aprender el modelo"""),
    code("""cal = pr.calendario(s.index)
normal = s.where(~anomalia)
habil = (cal["fin_de_semana"] == 0) & (cal[["feriado", "puente", "no_laborable"]].sum(axis=1) == 0)
ultimo_anio = s.index > fin - pd.DateOffset(years=1)
dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1.4, 1]})
sin_feriado = ultimo_anio & (cal[["feriado", "puente"]].sum(axis=1) == 0).to_numpy()
por_dia = normal[sin_feriado].groupby(s.index[sin_feriado].dayofweek).mean()
ax[0].bar(dias, por_dia.values / 1e3, color=[e.LINK] * 5 + [e.TINTA_SUAVE] * 2, width=0.65)
ax[0].set(title="Por día de la semana (último año)", ylabel="miles de pasajeros")

mensual = normal[habil.to_numpy()].groupby([s.index[habil.to_numpy()].year, s.index[habil.to_numpy()].month]).mean().unstack(0)
anios = list(mensual.columns)
tonos = np.linspace(1, 6, len(anios)).round().astype(int)   # de claro (años viejos) a oscuro (el último)
for k, anio in enumerate(anios):
    ultimo = anio == anios[-1]
    ax[1].plot(mensual.index, mensual[anio] / 1e3, marker="o", markersize=3.5, linewidth=2.4 if ultimo else 1.3,
               color=e.SECUENCIAL[tonos[k]], label=str(anio))
ax[1].set_xticks(range(1, 13), ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
ax[1].set(title="Día hábil promedio de cada mes", ylabel="miles de pasajeros")
ax[1].legend(ncol=len(anios), loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8, frameon=False)

tipico = normal.groupby([s.index.year, s.index.month, s.index.dayofweek]).transform("median")
efecto = (normal / tipico)[(cal["dia_semana"] < 5).to_numpy()]
grupos = {"Feriado": cal["feriado"] == 1, "Feriado puente": cal["puente"] == 1,
          "Pegado a un feriado": cal["pegado_feriado"] == 1}
valores = {k: float(efecto[v[(cal["dia_semana"] < 5)].to_numpy()].median() * 100) for k, v in grupos.items()}
ax[2].barh(list(valores), list(valores.values()), color=e.DIVERGENTE[1], height=0.55)
for i, v in enumerate(valores.values()):
    ax[2].text(v + 2, i, f"{e.num_es(v)} %", va="center", fontsize=9, color=e.TINTA)
ax[2].axvline(100, color=e.TINTA_SUAVE, linewidth=1)
ax[2].set(title="Feriados", xlim=(0, 125), xlabel="% de los pasajeros de un día hábil normal")
for a in ax[:2]:
    e.ejes_es(a, y=0)
ax[2].grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(FIG / "pron_patrones.png"); plt.show()
print(f"Un sábado tiene el {e.num_es(100 * por_dia[5] / por_dia[:5].mean())} % de un día hábil y un domingo el "
      f"{e.num_es(100 * por_dia[6] / por_dia[:5].mean())} %. Enero tiene el "
      f"{e.num_es(100 * mensual.loc[1].mean() / mensual.mean(axis=1).max())} % del mes de más pasajeros.")"""),
    md("""**Lectura.** La demanda depende sobre todo del **calendario**: los días hábiles tienen mucha más gente que los fines de
semana, los feriados se parecen a un domingo, el verano (enero y febrero) cae fuerte y se recupera en marzo. Por eso el
modelo usa como variables solo el calendario: se conocen con meses de anticipación, así que se puede pronosticar
cualquier fecha futura.

## 2. Modelo y validación

**Variables (todas del calendario):**"""),
    code("""pd.DataFrame({"variable": pr.VARIABLES, "qué mide": [pr.NOMBRES[v] for v in pr.VARIABLES]})"""),
    md("""- **Objetivo:** el logaritmo de los pasajeros del día (así el error es relativo: equivocarse en 50 mil pasajeros pesa
  más un domingo que un día hábil).
- **Modelos (clases 6 y 8):** **Random Forest** (500 árboles) y **XGBoost** (boosting de 800 árboles poco profundos).
- **Referencia ingenua:** "el mismo día de la semana del año anterior". Un modelo que no le gane a esto no sirve.
- **Validación hacia adelante:** se entrena solo con el pasado y se mide en los meses siguientes, que el modelo no vio.
  Lo hacemos dos veces, con los dos semestres más recientes. Nunca se usan fechas futuras para entrenar: sería hacer
  trampa, porque en la realidad el futuro no se conoce.
- **Métricas:** MAPE (error porcentual medio por día), MAE (pasajeros por día) y error del total del período."""),
    code("""cortes = pr.cortes_validacion(fin)
for nombre, fin_tr, fin_te in cortes:
    print(f"Prueba {nombre}: entrena hasta el {pd.Timestamp(fin_tr):%d/%m/%Y}, mide hasta el {pd.Timestamp(fin_te):%d/%m/%Y}")
met, preds = pr.validar(s, anomalia, cortes)
promedio = met.groupby("modelo")["MAPE (%)"].mean().sort_values()
elegido = next(m for m in promedio.index if m != "Mismo día del año anterior")
met.set_index(["período de prueba", "modelo"]).round(2)"""),
    code("""ultimo = cortes[-1][0]
p = preds[preds["periodo"] == ultimo].pivot_table(index="fecha", columns="modelo", values=["real", "prediccion"])
fig, ax = plt.subplots(figsize=(13, 4.6))
ax.plot(p.index, p[("real", elegido)] / 1e3, color=e.TINTA, linewidth=1.4, label="Real")
ax.plot(p.index, p[("prediccion", "Mismo día del año anterior")] / 1e3, color=e.EJE, linewidth=1.2,
        label=f"Mismo día del año anterior (MAPE {e.num_es(met.set_index(['período de prueba', 'modelo']).loc[(ultimo, 'Mismo día del año anterior'), 'MAPE (%)'], 1)} %)")
ax.plot(p.index, p[("prediccion", elegido)] / 1e3, color=e.LINK, linewidth=1.6,
        label=f"{elegido} (MAPE {e.num_es(met.set_index(['período de prueba', 'modelo']).loc[(ultimo, elegido), 'MAPE (%)'], 1)} %)")
ax.set(title=f"Validación hacia adelante: {ultimo}, meses que el modelo no vio", ylabel="miles de pasajeros por día")
eje_meses(ax); ax.legend(loc="lower right"); e.ejes_es(ax, y=0); fig.tight_layout(); fig.savefig(FIG / "pron_validacion.png"); plt.show()
m_el = met[met["modelo"] == elegido]["MAPE (%)"].mean()
m_in = met[met["modelo"] == "Mismo día del año anterior"]["MAPE (%)"].mean()
print(f"Modelo elegido: {elegido}. Erra en promedio un {e.num_es(m_el, 1)} % por día, contra {e.num_es(m_in, 1)} % de la "
      f"referencia ingenua: reduce el error a {e.num_es(100 * m_el / m_in)} %.")"""),
    md("""**Lectura.** Si el modelo erra poco **día por día** y casi nada en el **total del período**, sirve para planificar: el
error de un día puntual se compensa a lo largo del mes. Los días que más erra suelen ser los que tienen algo que el
calendario no sabe (lluvia fuerte, un evento, un paro parcial).

## 3. Qué aprendió el modelo

Importancia de cada variable en XGBoost (cuánto mejora la predicción cada vez que el modelo la usa para cortar)."""),
    code("""imp = pr.importancia(s, anomalia)
fig, ax = plt.subplots(figsize=(9, 4.4))
d = imp.sort_values()
ax.barh([pr.NOMBRES[v] for v in d.index], d.values, color=e.LINK, height=0.62)
ax.set(title="XGBoost: qué variables del calendario pesan más", xlabel="importancia (proporción)")
ax.grid(axis="y", visible=False); e.ejes_es(ax, x=2); fig.tight_layout(); fig.savefig(FIG / "pron_importancia.png"); plt.show()"""),
    md("""## 4. Pronóstico

Entrenamos con **toda** la serie y pronosticamos día por día hasta fin de año (como mínimo, tres meses). La banda es un
**intervalo del 80 %**: lo estimamos con dos modelos XGBoost de cuantiles (10 y 90), que aprenden el rango en el que
suelen caer los días de cada tipo."""),
    code("""pron = pr.pronosticar(s, anomalia, pr.horizonte(fin), elegido)
desde = fin - pd.DateOffset(months=9)
fig, ax = plt.subplots(figsize=(13, 4.8))
reciente = s[s.index > desde]
ax.plot(reciente.index, reciente / 1e3, color=e.TINTA_SUAVE, linewidth=1, label="Real")
ax.fill_between(pron["fecha"], pron["p10"] / 1e3, pron["p90"] / 1e3, color=e.SECUENCIAL[0], label="Intervalo del 80 %")
ax.plot(pron["fecha"], pron["pronostico"] / 1e3, color=e.LINK, linewidth=1.6, label=f"Pronóstico ({elegido})")
ax.axvline(fin, color=e.TINTA_SUAVE, linewidth=1, linestyle="--")
ax.text(fin, ax.get_ylim()[1], " hoy", va="top", fontsize=9, color=e.TINTA_2)
ax.set(title="Pronóstico de entradas al subte por día", ylabel="miles de pasajeros por día")
eje_meses(ax); ax.legend(loc="lower left"); e.ejes_es(ax, y=0); fig.tight_layout(); fig.savefig(FIG / "pron_pronostico.png"); plt.show()
mensual_p = pr.por_mes(s, anomalia, pron)
(mensual_p.rename(columns={"pronostico": "día hábil promedio", "p10": "mínimo probable", "p90": "máximo probable",
                           "anio_anterior": "mismo mes, año anterior", "variacion_%": "variación (%)"})
 .rename(index=lambda p_: pr.nombre_mes(p_.to_timestamp())).round(0))"""),
    code("""habiles = pron[pron["tipo_dia"] == "hábil"]
pico = habiles.loc[habiles["pronostico"].idxmax()]
valle = habiles.loc[habiles["pronostico"].idxmin()]
largos = pron[pron["tipo_dia"] == "feriado"]
print(f"Día hábil con más gente: {pico['fecha']:%d/%m/%Y} ({e.num_es(pico['pronostico'])} entradas). "
      f"Con menos: {valle['fecha']:%d/%m/%Y} ({e.num_es(valle['pronostico'])}).")
domingos = pron[pron["fecha"].dt.dayofweek == 6]["pronostico"].mean()
if len(largos):
    print("Feriados en el período:", ", ".join(f"{f:%d/%m}" for f in largos["fecha"]),
          f"· en promedio {e.num_es(largos['pronostico'].mean())} entradas (un domingo: {e.num_es(domingos)}).")"""),
    md("""## 5. Por estación y por hora

Para bajar el pronóstico a cada estación usamos su **parte** de las entradas de la red en los últimos tres meses (las
estaciones cambian de peso muy despacio). Así sale cuánta gente va a entrar por día hábil a cada una, y en particular a
las que hoy **no tienen un Link** a distancia caminable."""),
    code("""res = pr.resultados(DIR_BASE, PROC)   # guarda las tablas para la app y la presentación
est = pd.read_csv(PROC / "pronostico_estaciones.csv")
print(f"Pronóstico para {res['primer_mes_nombre']}: {e.num_es(res['pax_habil_primer_mes'])} entradas por día hábil en "
      f"toda la red ({'+' if res['variacion_primer_mes_pct'] >= 0 else ''}{e.num_es(res['variacion_primer_mes_pct'], 1)} % "
      "contra el mismo mes del año anterior).")
print(f"Las {res['estaciones_sin_link']} estaciones sin un Link cerca van a recibir unas "
      f"{e.num_es(res['pax_habil_estaciones_sin_link'])} entradas por día hábil.")
(est[~est["link_cerca"].astype(bool)][["linea", "estacion", "perfil", "pax_habil_reciente", "pax_habil_pronostico"]]
 .rename(columns={"pax_habil_reciente": "por día hábil (últimos 3 meses)", "pax_habil_pronostico": f"por día hábil ({res['primer_mes_nombre']})"})
 .round(0))"""),
    code("""if hora is not None:
    perfil = pr.perfil_horario(hora)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for tipo, color in (("hábil", e.LINK), ("fin de semana", e.TINTA_SUAVE)):
        d = perfil[perfil["tipo_dia"] == tipo]
        ax.plot(d["hora"], 100 * d["parte"], marker="o", markersize=4, linewidth=2, color=color, label=f"Día {tipo}")
    ax.set(title="Qué parte de las entradas del día ocurre en cada hora (últimos 6 meses)", xlabel="hora",
           ylabel="% de las entradas del día")
    ax.set_xticks(range(5, 24, 2))
    ax.legend(); e.ejes_es(ax, y=0); fig.tight_layout(); fig.savefig(FIG / "pron_horario.png"); plt.show()
    h = perfil[perfil["tipo_dia"] == "hábil"].set_index("hora")["parte"]
    print(f"En un día hábil, entre las 7 y las 10 entra el {e.num_es(100 * h.loc[7:9].sum())} % de la gente y entre las 17 y "
          f"las 20, el {e.num_es(100 * h.loc[17:19].sum())} %. Con el pronóstico de {res['primer_mes_nombre']}, son unas "
          f"{e.num_es(res['pax_habil_primer_mes'] * h.loc[7:9].sum())} personas en el pico de la mañana.")"""),
    md("""## 6. Conclusión

- **Técnica:** con variables de calendario, XGBoost y Random Forest pronostican la demanda diaria con un error muy chico
  frente a la referencia ingenua, medido hacia adelante en meses que no vieron.
- **Negocio:** la demanda de paso se puede **anticipar**. La Gerencia Técnica puede planificar la carga de efectivo
  antes de los días de más gente y concentrar el mantenimiento y las instalaciones en los días y meses de menos flujo
  (fines de semana, feriados y el verano). La Comercial sabe cuánta gente va a pasar por las estaciones sin un Link cerca.
- **Limitación:** los molinetes cuentan gente que entra al subte, no extracciones. Es un indicador del flujo de personas
  cerca de las estaciones, no de la demanda de efectivo, que es un dato privado de las redes."""),
    code("""if EN_COLAB:   # para pasarle la serie al grupo
    zip_serie = RAIZ / "data/serie_subte.zip"
    if zip_serie.exists():
        from google.colab import files
        files.download(str(zip_serie))"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.write(nb, SALIDA)
print("ok", SALIDA)
