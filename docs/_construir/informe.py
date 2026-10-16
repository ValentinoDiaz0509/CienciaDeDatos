"""Genera docs/informe.md a partir de los resultados (python docs/_construir/informe.py)."""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.estilo import num_es as n  # noqa: E402

P = RAIZ / "data/processed"
R = json.loads((P / "resumen.json").read_text(encoding="utf-8"))
met = pd.read_csv(P / "metricas_modelos.csv")
esp = met[met["validación"].str.startswith("espacial")].set_index("modelo")
ale = met[met["validación"] == "aleatoria"].set_index("modelo")
metc = pd.read_csv(P / "metricas_clasificacion.csv")
espc = metc[metc["validación"].str.startswith("espacial")].set_index("modelo")
alec = metc[metc["validación"] == "aleatoria"].set_index("modelo")
coef = pd.read_csv(P / "coeficientes_glm.csv").set_index("variable")
perfil = pd.read_csv(P / "clusters_perfil.csv").set_index("tipo_zona")
ex = pd.read_csv(P / "recomendaciones_exactas_20.csv")
comp = pd.read_csv(P / "goloso_vs_exacto.csv")
res = pd.read_parquet(P / "resultados_hex.parquet")
bp = pd.read_csv(P / "barrios_prediccion.csv", index_col="barrio")
curva = pd.read_csv(P / "curva_profundidad.csv")
oob = pd.read_csv(P / "oob_vs_validacion.csv").set_index("modelo")
imp = pd.read_csv(P / "importancia_variables.csv")
gini = pd.read_csv(P / "importancia_gini.csv")
cm = pd.read_csv(P / "matriz_confusion.csv", index_col=0)
val = pd.read_csv(P / "validacion_temporal.csv").set_index("puntaje")
cam = R["metodo_distancia"] == "caminando"
dist = "caminando por las calles" if cam else "en línea recta"

D2 = "D² Poisson (promedio por partición)"
top = bp[bp["diferencia_link"] > 0].head(5)
sin_link = bp[(bp["tiene_link"] == 0) & (bp["corresponden_link"] >= 1.5)].sort_values("corresponden_link", ascending=False)
sobra = bp.sort_values("diferencia_link").head(3)
orden = esp.sort_values(D2, ascending=False).index
tabla_modelos = "\n".join(
    f"| {mo} | {n(esp.loc[mo, D2], 2)} ± {n(esp.loc[mo, 'D² Poisson (desvío entre particiones)'], 2)} | "
    f"{n(ale.loc[mo, D2], 2)} | {n(esp.loc[mo, 'MAE'], 2)} |" for mo in orden)
orden_c = espc.sort_values("AUC (promedio por partición)", ascending=False).index
tabla_clasif = "\n".join(
    f"| {mo} | {n(espc.loc[mo, 'AUC (promedio por partición)'], 3)} | {n(espc.loc[mo, 'PR-AUC'], 3)} | "
    f"{n(espc.loc[mo, 'Precisión'], 3)} | {n(alec.loc[mo, 'AUC'], 3)} |" for mo in orden_c)
tabla_comp = "\n".join(
    f"| {int(r.cajeros)} | {n(r.goloso)} | {n(r.exacto)} | {n(r.diferencia_pct, 2)} % | {int(r.sitios_en_comun)} | {n(r.seg_exacto, 2)} |"
    for r in comp.itertuples())
zonas = [z for z in perfil.index if not z.startswith("Zonas sin")]
tabla_zonas = "\n".join(
    f"| {z} | {n(perfil.loc[z, 'residentes'])} | {n(perfil.loc[z, 'pax_subte'])} | {n(perfil.loc[z, 'terminales_cada_10mil'], 1)} | "
    f"{n(perfil.loc[z, 'link_cada_10mil'], 1)} | {n(100 * perfil.loc[z, 'pct_nbi_k1'], 1)} % |" for z in zonas)
tabla_top = "\n".join(f"| {b} | {n(r.residentes)} | {n(r.tiene_link)} | {n(r.corresponden_link)} | **+{n(r.diferencia_link)}** | "
                       f"{n(100 * r.cuota_link)} % |" for b, r in top.iterrows())
tabla_ex = "\n".join(f"| {int(r.orden)} | {r.barrio} | {int(r.comuna)} | {r.tipo_zona} | {n(r.personas_nuevas)} |" for r in ex.itertuples())

# Zonas de ejemplo (las mismas reglas que el notebook 02)
rr = res[res["esperado_total_k1"].notna()].set_index("h3")
za = rr.loc[rr[rr["term_link_k1"] == 0]["esperado_link_k1"].idxmax()]
_uno = rr[(rr["term_total_k1"] == 0) & rr["esperado_link_k1"].between(0.5, 1.5) & (rr["tipo_zona"] == "Residencial denso")]
zu = rr.loc[_uno["poblacion_k1"].idxmax()]
zb = rr.loc[rr[rr["poblacion_k1"] >= 15_000]["esperado_total_k1"].idxmin()]
mejor_prof = curva.loc[curva["validación espacial"].idxmax()]
op = res[res["deberia_tener_cajero"].fillna(False).astype(bool) & (res["term_total"] == 0)]
op_barrios = ", ".join(op["barrio"].value_counts().head(4).index)
est = pd.read_csv(P / "estaciones_subte.csv")
minus = lambda t: t[0].lower() + t[1:]  # noqa: E731


def cerca(z):
    d = ((est["lat"] - z["lat"]) * 111_000) ** 2 + ((est["lon"] - z["lon"]) * 92_000) ** 2
    i = d.idxmin()
    return f"a {n(round(d[i] ** 0.5, -2))} m de la estación {est.loc[i, 'estacion']} ({est.loc[i, 'linea']})"
ef = lambda v: n(coef.loc[v, "efecto"], 2)
bar_tipo = lambda z: ", ".join(ex[ex["tipo_zona"] == z]["barrio"].value_counts().index[:3])
cubrir_txt = f"los barrios residenciales ({bar_tipo('Residencial denso')}) y el sur ({bar_tipo('Mayor vulnerabilidad social')})"
comunes = R.get("sitios_comunes_con_pax_manana", 20)
robustez_txt = (f"sumar a los pasajeros de la hora pico de la mañana deja {comunes} de los 20 sitios óptimos iguales"
                if comunes < 20 else "sumar a los pasajeros de la hora pico de la mañana no cambia ninguno de los 20 sitios óptimos")
recta_txt = (f"Medir caminando cambia la respuesta: en línea recta, el {n(100 * R['cobertura_link_recta'])} % de los vecinos "
             f"tenía un Link a menos de 500 m; caminando por las calles, el {n(100 * R['cobertura_link'])} %. Uno de cada "
             f"{n(R['cobertura_link_recta'] / (R['cobertura_link_recta'] - R['cobertura_link']))} vecinos que la línea recta daba por "
             f"cubiertos en realidad tiene que caminar más." if cam else "")
igualar_txt = (f"; con {R['cajeros_para_igualar_banelco']} cajeros, Link igualaría la cobertura actual de Banelco"
               if R.get("cajeros_para_igualar_banelco") else "")
dif_ini = 100 * (R["cobertura_banelco"] - R["cobertura_link"])
dif_fin = 100 * (R["cobertura_banelco"] - R["cobertura_link_con_20"])
red_txt = (f"Las distancias se miden **caminando** por la red peatonal de OpenStreetMap ({n(R['red_km'])} km de calles y sendas, "
           f"{n(R['red_nodos'])} esquinas), con el algoritmo de Dijkstra. En la mediana, caminar hasta el Link más cercano es "
           f"{n(R['factor_desvio_red'], 2)} veces la distancia en línea recta. Cada punto se proyecta sobre la cuadra más cercana "
           "(no sobre la esquina, que sumaría hasta media cuadra en cada punta) y los cajeros candidatos se ubican sobre la vereda. "
           + (f"Como análisis de sensibilidad, si cada punto se conecta también con las cuadras a menos de 25 m extra (una esquina, "
              f"por ejemplo), la cobertura de Link pasa del {n(100 * R['cobertura_link'], 1)} % al "
              f"{n(100 * R['cobertura_link_red_tolerancia_25m'], 1)} %: el resultado no depende de ese detalle."
              if "cobertura_link_red_tolerancia_25m" in R else "") if cam else
           "Las distancias se miden en línea recta (la red peatonal de OpenStreetMap se incorpora con `notebooks/00b_red_peatonal.ipynb`).")

# Pronóstico del subte (notebook 05): textos con los números de data/processed/pronostico_*.
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre",
         "diciembre"]
_fl = lambda iso: f"{MESES[int(iso[5:7]) - 1]} de {iso[:4]}"  # noqa: E731
_sg = lambda v, d=1: ("+" if v >= 0 else "") + n(v, d)  # noqa: E731
PR = json.loads((P / "pronostico_resumen.json").read_text(encoding="utf-8")) if (P / "pronostico_resumen.json").exists() else None
pron_resumen = pron_seccion = pron_eval = pron_datos = ""
if PR:
    _el, _ing = PR["modelo_elegido"], "Mismo día del año anterior"
    _otro = next(m for m in PR["mape_promedio"] if m not in (_el, _ing))
    _m, _mi, _mo = PR["mape_promedio"][_el], PR["mape_promedio"][_ing], PR["mape_promedio"][_otro]
    _tot = PR["error_total_ultimo_periodo"][_el]
    _min_pct = 100 * (PR["pax_habil_mes_minimo"] / PR["pax_habil_primer_mes"] - 1)
    _periodo = f"{_fl(PR['desde'])} a {_fl(PR['hasta'])}"
    pv = pd.read_csv(P / "pronostico_validacion.csv")
    imp_p = pd.read_csv(P / "pronostico_importancia.csv")
    _tops = [minus(v) for v in imp_p["nombre"].head(3)]
    _top3 = ", ".join(_tops[:-1]) + " y " + _tops[-1]
    _tot_el = sorted(pv[pv["modelo"] == _el]["error del total (%)"])
    tabla_pron = "\n".join(
        f"| {r['período de prueba']} | {r['modelo']} | {n(r['MAPE (%)'], 1)} % | {n(r['MAE (pasajeros/día)'])} | "
        f"{_sg(r['error del total (%)'])} % |" for _, r in pv.iterrows())
    tabla_mens = "\n".join(
        f"| {m['nombre']} | {n(m['pax_habil'])} | {n(m['p10'])} a {n(m['p90'])} | "
        f"{n(m['anio_anterior']) if m['anio_anterior'] else 'sin dato'} | "
        f"{_sg(m['variacion_pct']) + ' %' if m['variacion_pct'] is not None else 'sin dato'} |" for m in PR["mensual"])
    pron_resumen = (
        f"- Un segundo modelo **pronostica** cuánta gente va a entrar al subte cada día: **{_el}** sobre variables del "
        f"calendario erra un **{n(_m, 1)} %** por día en meses que no vio, contra {n(_mi, 1)} % de repetir el año anterior. "
        f"Espera **{n(PR['pax_habil_primer_mes'] / 1e3)} mil entradas por día hábil** en {PR['primer_mes_nombre']} y "
        f"{n(PR['pax_habil_mes_minimo'] / 1e3)} mil en {PR['mes_minimo_nombre']}: la Gerencia Técnica puede planificar cuándo "
        "reforzar la carga de efectivo y cuándo hacer el mantenimiento.\n")
    pron_datos = (f"| Molinetes del subte, serie diaria | SBASE · BA Data | {_periodo} | Entradas por día y estación, y por hora "
                  "en toda la red (pronóstico) |\n")
    pron_eval = (f"- **Pronóstico (validación hacia adelante):** MAPE de {n(_m, 1)} % por día con {_el}, contra {n(_mi, 1)} % de "
                 f"repetir el año anterior; en el último período ({PR['periodos_prueba'][-1]}), {n(PR['mape_ultimo_periodo'][_el], 1)} % "
                 f"y {'sobreestima' if _tot >= 0 else 'subestima'} el total en {n(abs(_tot), 1)} %.\n")
    pron_seccion = f"""
### 4.7 Pronóstico: cuánta gente va a pasar por el subte (`notebooks/05_pronostico_subte.ipynb`)
Las secciones anteriores predicen **dónde**; esta predice **cuándo**. Con la serie diaria de los molinetes ({_periodo};
{n(PR['dias'])} días con datos) pronosticamos las entradas al subte de cada día de los seis meses siguientes al último dato,
para que la Gerencia Técnica planifique la carga de efectivo y el mantenimiento de los cajeros cerca de las estaciones.

- **Limpieza:** {PR['dias_anomalos']} días anómalos quedan afuera del entrenamiento y de la medición: paros y cortes (menos de la
  mitad de un día igual de las semanas de alrededor), registros duplicados (más de 1,25 veces) y {PR['dias_fecha_cruzada']} días con
  la **fecha mal cargada** en el archivo de 2025. Una parte de los registros del 1 al 12 de agosto trae la fecha como mes/día
  y, leída como día/mes, cae en el día 8 de otro mes: el 8 de cada mes de 2025 tiene un tercio más de pasajeros que un día
  típico y del 1 al 12 de agosto falta lo mismo, mientras que en 2026 no pasa. La prueba más clara: marzo de 2025 no está en el
  archivo, pero el 8 de marzo tiene entradas, que sumadas a las del 3 de agosto dan un domingo normal.
- **Variables (13, todas del calendario):** día de la semana, fin de semana, mes, día del mes, época del año (seno y coseno
  del día del año), semana del mes, feriado, feriado puente, día no laborable, día pegado a un feriado, vacaciones de invierno
  y año. Se conocen de antemano, incluso los feriados que se decretan con poco tiempo (como el del 9 de noviembre de 2026 por
  la visita del Papa), así que se puede pronosticar cualquier fecha futura.
- **Objetivo:** el logaritmo de las entradas del día, para que el error sea relativo.
- **Modelos (clases 6 y 8):** Random Forest (500 árboles) y XGBoost (800 árboles de profundidad 4), contra una referencia
  ingenua: el mismo día de la semana del año anterior.
- **Validación hacia adelante con origen móvil:** se entrena con el pasado y se mide en los cuatro meses siguientes, que el
  modelo no vio. Dos veces ({" y ".join(PR['periodos_prueba'])}), siempre con al menos un año de historia; los tres se
  miden en los mismos días. Nunca se usan fechas futuras para entrenar.

| Período de prueba | Modelo | MAPE | MAE (entradas por día) | Error del total |
|---|---|---|---|---|
{{tabla_pron}}

**Elección: {_el}**, el de menor error promedio: MAPE de {n(_m, 1)} %, contra {n(_mo, 1)} % de {_otro} y {n(_mi, 1)} % de la
referencia (su error es el {n(100 * _m / _mi)} % del de la referencia). El error del total del período va de
{" a ".join(_sg(v) + " %" for v in _tot_el)} según el período: con menos de dos años de historia (y sin marzo de 2025), el
modelo capta solo en parte los cambios de nivel de un año a otro, así que conviene leer cada mes como un rango. Lo que más pesa
es {_top3}. El intervalo del 80 % sale de dos modelos XGBoost de cuantiles (10 y 90).

| Mes | Día hábil promedio | Rango probable (80 %) | Mismo mes, un año antes | Variación |
|---|---|---|---|---|
{{tabla_mens}}

**Lectura para el negocio.** El día hábil promedio baja de {n(PR['pax_habil_primer_mes'] / 1e3)} mil en {PR['primer_mes_nombre']} a
{n(PR['pax_habil_mes_minimo'] / 1e3)} mil en {PR['mes_minimo_nombre']} ({n(_min_pct)} %): el verano es el momento para el mantenimiento
y las instalaciones, y los días previos a los de más gente, para reforzar la carga de efectivo. Las {PR['estaciones_sin_link']}
estaciones sin un Link a distancia caminable van a recibir unas {n(PR['pax_habil_estaciones_sin_link'])} entradas por día hábil.
{f"Entre las 7 y las 10 de un día hábil entra el {n(100 * PR['parte_pico_manana'])} % de la gente del día. " if "parte_pico_manana" in PR else ""}Los
molinetes cuentan entradas al subte, no extracciones: son un indicador del flujo de gente cerca de las estaciones, no de la
demanda de efectivo.
"""
    pron_seccion = pron_seccion.replace("{tabla_pron}", tabla_pron).replace("{tabla_mens}", tabla_mens)

# Técnicas de la clase 11 (notebook 06)
_RT = P / "tecnicas_resumen.json"
RT = json.loads(_RT.read_text(encoding="utf-8")) if _RT.exists() else None
tec_seccion = tec_eval = ""
if RT:
    est_t = pd.read_csv(P / "tecnicas_estratificada.csv")
    km_t = pd.read_csv(P / "tecnicas_kmeans_vs_optimo.csv")
    bol_t = pd.read_csv(P / "tecnicas_bolsones.csv")
    tabla_est = "\n".join(f"| {r['modelo']} | {n(r['AUC por comuna'], 3)} | {n(r['AUC estratificada'], 3)} |" for _, r in est_t.iterrows())
    tabla_km = "\n".join(
        f"| {int(r['cajeros'])} | {n(r['óptimo exacto'])} | {n(r['K-Means'])} | {n(r['K-Means / óptimo (%)'])} % | "
        f"{n(r['distancia media con el óptimo (m)'])} | {n(r['distancia media con K-Means (m)'])} |"
        for _, r in km_t[km_t["cajeros"].isin([5, 10, 20, 30, 50])].iterrows())
    tabla_bol = "\n".join(f"| {int(r['bolsón'])} | {r['barrios']} | {n(r['vecinos sin Link'])} | "
                           f"{'sí' if r['le toca un cajero óptimo'] else 'no'} |" for _, r in bol_t.head(5).iterrows())
    tec_seccion = f"""
### 4.8 Técnicas de la clase 11 (`notebooks/06_tecnicas_avanzadas.ipynb`)
La clase 11 (técnicas avanzadas de minería de datos) repasó la preparación de variables y el escalado, la validación cruzada
estratificada y tres técnicas no supervisadas. Las aplicamos donde responden una pregunta del negocio.

**Escalado sin fuga de datos.** El logaritmo de los conteos y el `StandardScaler` son pasos del pipeline de la regresión de
Poisson y de la logística, así que en cada partición se ajustan solo con los datos de entrenamiento. Los árboles, Random
Forest y XGBoost no se escalan: no lo necesitan.

**Validación cruzada estratificada por grupos.** GroupKFold deja comunas enteras afuera, pero no controla la proporción de
zonas con cajero: en las particiones de prueba va de {n(RT['tasa_por_comuna_min'])} % a {n(RT['tasa_por_comuna_max'])} %.
Con `StratifiedGroupKFold` (también anidada) va de {n(RT['tasa_estratificada_min'])} % a {n(RT['tasa_estratificada_max'])} %, y
el resultado no cambia: {RT['mejor_estratificada']} sigue siendo el mejor clasificador.

| Clasificador | AUC por comuna | AUC por comuna y estratificada |
|---|---|---|
{{tabla_est}}

**PCA.** Sobre las 14 variables de demanda (conteos en logaritmo, todo estandarizado), dos componentes guardan el
{n(RT['pca_varianza_2'])} % de la varianza y cuatro, el {n(RT['pca_varianza_4'])} %; el criterio de Kaiser también se queda con
{RT['pca_kaiser']}. El primero es la intensidad urbana (comercio, oficinas, vecinos), el segundo la lejanía del centro y del
subte, el tercero el nivel socioeconómico y el cuarto, casi solo las estaciones de tren. Repetido sobre cuatro componentes,
el K-Means de tipos de zona da casi la misma agrupación (índice de Rand ajustado {n(RT['ari_4_componentes'], 2)}). No lo usamos
como entrada de los modelos: son pocas variables, los árboles no lo necesitan y se perdería la explicación por variable (SHAP).

**K-Means como ubicador de cajeros.** En la clase se propuso usar K-Means para ubicar cajeros: cada centro de grupo es un
cajero. Lo aplicamos a la gente sin un Link a 500 m (cada hexágono pesa por sus vecinos) y llevamos cada centro al sitio con
comercio más cercano. Frente al óptimo exacto:

| Cajeros | Óptimo exacto (vecinos) | K-Means (vecinos) | K-Means / óptimo | Distancia media, óptimo (m) | Distancia media, K-Means (m) |
|---|---|---|---|---|---|
{{tabla_km}}

K-Means minimiza la distancia de cada persona al centro de su grupo, pero no sabe dónde están los cajeros actuales ni que la
meta es quedar a 500 m: suma la mitad de vecinos y tampoco gana en distancia media (hoy {n(RT['dist_media_hoy'])} m para la gente
sin Link). Con 20 cajeros, solo {RT['sitios_en_comun_20']} sitios coinciden.

**DBSCAN.** Sobre los hexágonos con vecinos sin un Link a distancia caminable, pesados por su gente (radio de 320 m, mínimo
de {n(RT['minimo_bolson'])} vecinos), encuentra **{RT['bolsones']} bolsones** densos que juntan el {n(RT['pct_en_bolsones'])} % de
esa gente; el resto vive disperso. Los 20 cajeros óptimos caen en {RT['bolsones_con_cajero_optimo']} bolsones, así que los otros
{RT['bolsones'] - RT['bolsones_con_cajero_optimo']} son la lista para una segunda tanda.

| Bolsón | Barrios | Vecinos sin Link | Le toca un cajero óptimo |
|---|---|---|---|
{{tabla_bol}}
"""
    tec_seccion = (tec_seccion.replace("{tabla_est}", tabla_est).replace("{tabla_km}", tabla_km)
                   .replace("{tabla_bol}", tabla_bol))
    tec_eval = (f"- **Robustez (clase 11):** con validación estratificada por grupos el clasificador elegido no cambia (AUC "
                f"{n(RT['auc_estratificada_rf'], 2)}); los tipos de zona se mantienen sobre componentes principales (ARI "
                f"{n(RT['ari_4_componentes'], 2)}); ubicar los cajeros con K-Means suma el {n(RT['kmeans_pct_optimo_20'])} % de los "
                "vecinos del óptimo exacto.\n")

fig_red = ("\n![Ejemplo en San Cristóbal: el Link más cercano está a unos 400 m en línea recta y a más de 600 m caminando](figuras/ejemplo_caminando.png)\n" if cam and (RAIZ / "docs/figuras/ejemplo_caminando.png").exists() else "")

texto = f"""# Informe técnico · ¿Dónde faltan cajeros Link en CABA?

TPO de Ciencia de Datos · UADE · 2º cuatrimestre 2026 · Docente: Santiago Martín

Metodología: CRISP-DM. Código, datos y resultados: [github.com/ValentinoDiaz0509/CienciaDeDatos](https://github.com/ValentinoDiaz0509/CienciaDeDatos).
Este informe se genera a partir de los resultados del pipeline (`python docs/_construir/informe.py`); todos los números salen de `data/processed/`.

---

## Resumen ejecutivo

- Hoy el **{n(100 * R['cobertura_link'])} %** de los vecinos de CABA tiene un cajero Link a menos de 500 m ({dist}), contra el
  **{n(100 * R['cobertura_banelco'])} %** de Banelco. Unas **{n(R['residentes_sin_link'])} personas** no tienen un Link a distancia caminable.
  {recta_txt}
- Los cajeros **siguen al comercio, no a los vecinos**: en los corredores comerciales hay
  {n(perfil.loc['Corredores comerciales con subte', 'terminales_cada_10mil'], 1)} terminales cada 10.000 vecinos; en los barrios
  residenciales, {n(perfil.loc['Residencial denso', 'terminales_cada_10mil'], 1)}.
- Un modelo de **machine learning**, un ensamble de regresión de Poisson y XGBoost elegido entre 7 con validación cruzada
  espacial anidada (D² = {n(R['d2_espacial_promedio'], 2)}; error medio de {n(R['mae_espacial'], 1)} terminales por zona), estima cuántos
  cajeros le corresponden a cada zona según su demanda. A Link le faltan terminales sobre todo en **{top.index[0]}** (le corresponden
  {n(top.iloc[0]['corresponden_link'])}, tiene {n(top.iloc[0]['tiene_link'])}) y **{top.index[1]}** ({n(top.iloc[1]['corresponden_link'])} contra
  {n(top.iloc[1]['tiene_link'])}). Un clasificador **{R['clasificador_elegido']}** (AUC = {n(R['auc_clasificador'], 2)}) encuentra
  **{n(R['zonas_sin_cajero_que_deberian'])} zonas sin ningún cajero** que, por su demanda, deberían tener uno: viven ahí
  {n(R['residentes_zonas_sin_cajero_que_deberian'])} personas.
- Hay **dos estrategias**. **Competir** donde la demanda justifica más terminales Link de las que hay ({", ".join(top.index[:3])}).
  **Cubrir** {cubrir_txt}, donde no hay un Link a distancia caminable.
- Con **20 cajeros** en las ubicaciones óptimas, Link suma **{n(R['nuevos_20_exacto'])} vecinos** y pasa al
  **{n(100 * R['cobertura_link_con_20'], 1)} %**. La diferencia con Banelco baja de {n(dif_ini)} a {n(dif_fin)} puntos{igualar_txt}. Al promedio
  nacional de {n(3242)} extracciones por terminal por mes (BCRA, 2025), son unas **{n(20 * 3242)} operaciones mensuales**.
{pron_resumen}- El modelo se entrega como una **app** que la gerencia puede usar con su listado actual de cajeros.

---

## 1. Comprensión del negocio

### 1.1 Dominio
En la Argentina operan dos grandes redes de cajeros automáticos: **Red Link** y **Banelco**. Los bancos compran e instalan
las terminales; la red las conecta y procesa las operaciones. Según el BCRA, en julio de 2025 se hicieron **57,2 millones de
extracciones** en **17.643 cajeros** del país (unas 3.242 por terminal por mes), con una caída interanual cercana al 40 %, en un
contexto de avance de los pagos digitales. Con menos uso del efectivo, **dónde** está cada terminal importa más: una terminal mal
ubicada es un costo fijo con poco uso.

### 1.2 Problema e hipótesis
> En CABA hay zonas con mucha demanda potencial y pocos cajeros Link; detectarlas con datos públicos le permite a la
> Gerencia Comercial y Técnica de Red Link proponerles a sus bancos dónde instalar las próximas terminales con un
> argumento medible, en vez de a ojo.

| Pregunta | Respuesta |
|---|---|
| ¿Qué resolvemos? | Dónde faltan terminales Link y dónde conviene sumarlas |
| ¿Para quién? | Gerencia Comercial y Técnica de Red Link (audiencia no especialista en datos) |
| ¿Cómo entrega valor? | Más vecinos con un Link cerca, más operaciones en la red y una priorización que reemplaza la decisión a ojo |

### 1.3 Supuestos y alcance
- Cada banco decide la instalación; Red Link puede proponer ubicaciones a los bancos de su red.
- Solo datos públicos: no se usan transacciones ni ningún dato interno de Red Link.
- Un integrante del grupo trabaja en Red Link. Por eso elegimos este dominio; el trabajo no usa información interna de la
  empresa y sus conclusiones no la representan.
- Alcance: Ciudad Autónoma de Buenos Aires.
- Criterio de éxito: un modelo de demanda que generalice a zonas no vistas (validación espacial) y un conjunto de
  ubicaciones cuyo impacto se pueda medir en vecinos cubiertos.

## 2. Comprensión de los datos

| Fuente | Organismo | Período | Aporta |
|---|---|---|---|
| Cajeros automáticos ({n(R['ubicaciones'])} ubicaciones, {n(R['terminales'])} terminales) | GCBA · BA Data | ≈2017 | Oferta por red y banco |
| Censo 2022 por radio censal (3.820 radios) | INDEC | 2022 | Residentes, NBI, edad, educación, internet |
| Molinetes del subte ({R['estaciones']} estaciones) | SBASE · BA Data | 2025 | Entradas por estación y hora |
{pron_datos}| Relevamiento de usos del suelo (417.764 parcelas) | GCBA · BA Data | 2022-2024 | Comercio, oficinas, vivienda |
| OpenStreetMap (≈29 mil puntos y red peatonal) | Colaboradores de OSM | 2026 | Comercios, estaciones, calles |
| Barrios | GCBA · BA Data | vigente | Límites |

**Calidad de datos (DAMA).** El listado de cajeros da bien en completitud (97 %), validez (100 %), unicidad (98,9 %: 14 duplicados por ubicación y banco) y consistencia (96,9 %), pero falla en **oportunidad**: el 31 %
de las ubicaciones está a nombre de bancos que dejaron de operar con ese nombre entre 2017 y 2024 (Citibank, BBVA Banco
Francés, Santander Río, HSBC). Lo tratamos como una foto de ≈2017. El censo por radio cubre el 99,2 % de la población oficial.
A los molinetes de 2025 les falta marzo y, del 1 al 12 de agosto, una parte de los registros trae la fecha invertida (ver 4.7). El CSV de usos del suelo no trae coordenadas, así que se tomaron del shapefile.
OpenStreetMap está incompleto para cajeros ({R['cajeros_osm_2026']} contra {n(R['ubicaciones'])}).

**Hallazgos del EDA** (cinco tipos de análisis exploratorio, `notebooks/01_calidad_y_eda.ipynb`):
1. El 78 % de los hexágonos no tiene ninguna terminal y la distribución tiene cola larga, lo que justifica usar modelos de conteo.
2. Las terminales se asocian más con el comercio (ρ = 0,68) que con los residentes (ρ = 0,47).
3. El centro concentra el 40 % de las terminales con el 12 % de los vecinos.
4. Link tiene mayor participación en el sur, donde pesan los bancos públicos, y menor en el norte comercial.

## 3. Preparación de los datos

- **Unidad de análisis:** {n(R['hexagonos'])} hexágonos H3 de resolución 9 (≈{n(R['area_media_hex_km2'], 2)} km², unas 3 × 3 manzanas).
  Todos tienen el mismo tamaño y los mismos vecinos, lo que reduce el problema de la unidad de área modificable frente a
  unidades administrativas desiguales.
- **Residentes por hexágono:** interpolación areal desde los radios censales (se conservan los totales). Los porcentajes se
  calculan agregando numeradores y denominadores, nunca promediando porcentajes.
- **Área de influencia:** variables y objetivo se suman sobre el hexágono y sus 6 vecinos (≈400 m), la escala de una caminata.
- **Variables de demanda (14):** residentes, comercio (OSM y parcelas), oficinas, parcelas residenciales, pasajeros de
  subte, estaciones de tren, distancia al centro y al subte, % NBI, % sin internet, % de 65+, % universitarios, tasa de empleo.
  Se excluyen deliberadamente sucursales y cajeros: son oferta, no demanda.
- **Distancias:** {red_txt}
{fig_red}

## 4. Modelado

Seguimos los cuatro niveles de análisis de la materia: el EDA es el **descriptivo**; los modelos de esta sección son el
**predictivo** (4.1 y 4.2 en el espacio, 4.7 en el tiempo) y el **diagnóstico** que lo explica (4.3); la segmentación y la optimización son el
**prescriptivo** (4.4 a 4.6).

### 4.1 Regresión: ¿cuántas terminales le corresponden a cada zona? (`notebooks/02_machine_learning.ipynb`)
Objetivo: terminales de todas las redes en el área de influencia. Es un conteo con muchos ceros y cola larga, así que
usamos pérdida de Poisson y medimos con la **devianza de Poisson (D²)**, el equivalente del R² para conteos, y el **error
absoluto medio (MAE)**.

**Validación cruzada espacial anidada.** Las zonas vecinas se parecen y comparten área de influencia: si se mezclan al azar
entrenamiento y prueba, el modelo "ve" la respuesta en los vecinos. Por eso se mide siempre en **comunas que el modelo no vio**
(GroupKFold por comuna, 5 particiones), y los hiperparámetros se eligen con GridSearchCV **adentro** de cada partición de
entrenamiento, también dejando comunas afuera (4 particiones internas). La regla de elección se fijó antes de ver los
resultados: el mayor D² promedio entre particiones.

| Modelo | D² espacial (promedio ± desvío) | D² aleatoria | MAE espacial |
|---|---|---|---|
{tabla_modelos}

**Elección: {R['modelo_elegido']}** (promedio geométrico de la regresión de Poisson y XGBoost): el D² más alto y más estable, y
el menor error. Con validación aleatoria habría ganado {R['modelo_mejor_aleatoria']} (D² = {n(ale.loc[R['modelo_mejor_aleatoria'], D2], 2)}),
que en comunas no vistas cae a {n(esp.loc[R['modelo_mejor_aleatoria'], D2], 2)}: los árboles memorizan la geografía
(autocorrelación espacial). La regla "más vecinos, más cajeros" (Poisson solo con población) es peor que predecir el promedio.

**Sobreajuste y OOB (clases 6 y 8).** Un árbol de decisión mejora siempre en entrenamiento (D² = {n(curva.iloc[-1]['entrenamiento'], 2)}
con {int(curva.iloc[-1]['profundidad'])} niveles), pero en comunas no vistas el mejor resultado está en {int(mejor_prof['profundidad'])} niveles
(D² = {n(mejor_prof['validación espacial'], 2)}) y después empeora. El error OOB del Random Forest da R² = {n(oob.loc['Random Forest (regresión)', 'OOB'], 2)},
pero en comunas no vistas el R² es {n(oob.loc['Random Forest (regresión)', 'comunas no vistas'], 2)}: con datos espaciales el OOB es
optimista, por la misma razón que la validación aleatoria.

**Brecha** = terminales que le corresponden (predicción fuera de muestra) − terminales que tiene. Para Link, lo que le
corresponde es la predicción por la participación de Link en la ciudad ({n(100 * R['cuota_link'], 1)} %). Por barrio, la predicción
de cada área de influencia se reparte entre sus 7 hexágonos para no contar cada terminal siete veces.

| Barrio | Residentes | Terminales Link | Le corresponden | Faltan | Participación Link |
|---|---|---|---|---|---|
{tabla_top}

Hay barrios sin ninguna terminal Link a los que les corresponde alguna: {", ".join(f"{b} ({n(r.corresponden_link)})" for b, r in sin_link.iterrows())}.
En el otro extremo, el microcentro ({"; ".join(f"{b}: tiene {n(r.tiene_link)} terminales Link y le corresponden {n(r.corresponden_link)}" for b, r in sobra.iterrows())})
concentra las casas centrales de los bancos públicos de la red y gente que llega en colectivo y tren, que el modelo ve solo
en parte: lo leemos como sobreoferta a revisar, no como cajeros que sobran.

### 4.2 Clasificación: ¿a esta zona le corresponde al menos un cajero?
Objetivo: el hexágono tiene o no al menos una terminal de cualquier red. Misma validación espacial anidada; el umbral marca
tantas zonas como las que hoy tienen cajero (tasa base), así que precisión y recall coinciden.

| Modelo | AUC espacial (promedio) | PR-AUC | Precisión = recall | AUC aleatoria |
|---|---|---|---|---|
{tabla_clasif}

Elegimos **{R['clasificador_elegido']}**: AUC de {n(R['auc_clasificador_promedio'], 2)} en promedio por grupo de comunas
({n(R['auc_clasificador'], 2)} con todas las zonas juntas). En la matriz de confusión, los falsos positivos son
**oportunidades**: {n(cm.loc['Real: sin cajero', 'Predicho: con cajero'])} zonas sin ningún cajero que, por su demanda, deberían tener uno,
donde viven {n(R['residentes_zonas_sin_cajero_que_deberian'])} personas (sobre todo en {op_barrios}). Un árbol de 3 niveles, entrenado solo
para leer las reglas, empieza preguntando por la cantidad de comercios del hexágono.

### 4.3 Diagnóstico: por qué el modelo predice lo que predice
El modelo elegido es aditivo en escala logarítmica, así que la predicción de cada zona se descompone **exactamente** en el
aporte de cada variable (valores **SHAP**: TreeSHAP en XGBoost y coeficiente × valor en la regresión de Poisson). Según SHAP lo
que más pesa es {", ".join(minus(v) for v in imp["nombre"].head(3))}; según la importancia de **Gini** del Random Forest,
{", ".join(minus(v) for v in gini["nombre"].head(3))}. En la regresión de Poisson, subir un desvío las parcelas comerciales multiplica
las terminales esperadas por {ef('parcelas_comercio_k1')}; los comercios de OSM, por {ef('actividad_osm_k1')}; el % de 65+, por
{ef('pct_65_k1')}; los pasajeros de subte, por {ef('pax_subte_habil_k1')}; las parcelas residenciales, por {ef('parcelas_residencial_k1')};
y el % de hogares con NBI, por {ef('pct_nbi_k1')}.

Tres zonas, tres decisiones:

| Zona | Demanda en ≈400 m | Le corresponden | Tiene | Decisión |
|---|---|---|---|---|
| {za['barrio']}, {cerca(za)} | {n(za['poblacion_k1'])} vecinos, {n(za['actividad_osm_k1'])} comercios | {n(za['esperado_total_k1'], 1)} ({n(za['esperado_link_k1'], 1)} Link) | {n(za['term_total_k1'])} (ninguna Link) | **Competir** |
| {zu['barrio']}, {cerca(zu)} | {n(zu['poblacion_k1'])} vecinos, {n(zu['actividad_osm_k1'])} comercios, sin subte | {n(zu['esperado_total_k1'], 1)} ({n(zu['esperado_link_k1'], 1)} Link) | {n(zu['term_total_k1'])} | **Un solo cajero Link** |
| {zb['barrio']} | {n(zb['poblacion_k1'])} vecinos, {n(zb['actividad_osm_k1'])} comercios | {n(zb['esperado_total_k1'], 1)} | {n(zb['term_total_k1'])} | **Cubrir**, no competir |

En {zu['barrio']} viven muchas personas pero el flujo de gente es bajo (pocos comercios y sin subte): le corresponde un solo
cajero Link ({n(zu['esperado_link_k1'], 1)} terminales), y hoy no tiene ninguno de ninguna red. En {zb['barrio']} el mercado no pondría un cajero; si el objetivo es acercar el
efectivo a quienes más lo usan, entra por la estrategia de cobertura.

### 4.4 K-Means: tipos de zona (`notebooks/03_optimizacion.ipynb`)
k = {R['k_clusters']} por silueta ({n(R['silueta'], 2)}) y codo, sobre variables de demanda estandarizadas (conteos en escala log).

| Tipo de zona | Residentes | Pasajeros de subte/día | Terminales cada 10 mil | Link cada 10 mil | NBI (mediana) |
|---|---|---|---|---|---|
{tabla_zonas}

### 4.5 Optimización: dónde sumar cajeros
**Problema de cobertura máxima** (MCLP; Church y ReVelle, 1974): elegir N sitios (hexágonos con comercio) que maximicen los
vecinos que pasan a tener un Link a menos de 500 m. Se resuelve con el **algoritmo goloso** (garantía de 1 − 1/e del óptimo por
submodularidad; Nemhauser, Wolsey y Fisher, 1978) y de forma **exacta** como programa lineal entero con HiGHS.

| Cajeros | Goloso (vecinos) | Exacto (vecinos) | Diferencia | Sitios en común | Segundos (exacto) |
|---|---|---|---|---|---|
{tabla_comp}

El goloso queda a menos del {n(R['goloso_max_diferencia_pct'], 1)} % del óptimo. Las 20 ubicaciones óptimas:

| # | Barrio | Comuna | Tipo de zona | Vecinos que suma |
|---|---|---|---|---|
{tabla_ex}

### 4.6 Demanda por hora (`notebooks/04_demanda_por_hora.ipynb`)
Con los molinetes 2025 ({n(R['pax_subte_habil'])} entradas por día hábil), los picos son a las **{R['hora_pico_manana']}:00** y las
**{R['hora_pico_tarde']}:00**. Un K-Means sobre la forma de la curva horaria de cada estación separa estaciones de **origen** (pico a la
mañana, barrios) y de **destino** (pico a la tarde, centro). {R['estaciones'] - R['estaciones_sin_link']} de {R['estaciones']}
estaciones ya tienen un Link a menos de 500 m; {robustez_txt}: el faltante grande está en los barrios, no en el transporte.
{pron_seccion}{tec_seccion}
## 5. Evaluación

- **Generalización (regresión):** D² espacial = {n(R['d2_espacial_promedio'], 2)} ± {n(R['d2_espacial_desvio'], 2)} entre grupos de
  comunas; MAE = {n(R['mae_espacial'], 1)} terminales por área de influencia.
- **Generalización (clasificación):** AUC espacial = {n(R['auc_clasificador'], 2)}; PR-AUC = {n(R['pr_auc_clasificador'], 2)} (la tasa base
  es {n(100 * R['base_positivos_clasificador'])} %).
- **Validación temporal:** de los {R['cajeros_osm_2026']} cajeros que OpenStreetMap tiene mapeados en 2026, {R['cajeros_nuevos_osm']} no
  estaban en el listado 2017. Cayeron, en promedio, en zonas con más brecha que el {n(100 * R['percentil_cajeros_nuevos'])} % de la
  ciudad (p = {n(R['p_valor_cajeros_nuevos'], 3)}, Monte Carlo). El AUC de la brecha es {n(R['auc_validacion'], 2)} y el de la probabilidad
  del clasificador, {n(val.loc['Probabilidad de cajero (clasificador)', 'AUC'], 2)}, contra {n(val.loc['Solo residentes', 'AUC'], 2)} de usar solo la
  población. Es una evidencia a favor, pero modesta, por la cantidad de casos.
{pron_eval}{tec_eval}- **Robustez:** {robustez_txt}. El goloso queda a menos del {n(R['goloso_max_diferencia_pct'], 1)} % del óptimo exacto, y medir
  con otra regla de conexión a la red cambia la cobertura en poco más de un punto.

**Limitaciones.**
- La oferta es de ≈2017.
- No hay transacciones: la demanda se aproxima con residentes, comercio y pasajeros.
- Lo que le "corresponde" a una zona compara contra cómo está repartido hoy el mercado, así que no detecta zonas que el
  mercado entero desatiende (la optimización por cobertura cubre ese caso).
- OpenStreetMap varía en completitud entre barrios.
- Los feriados en día de semana cuentan como hábiles en el promedio horario.
- La serie diaria del subte tiene menos de dos años (sin marzo de 2025): con más años, el pronóstico aprendería mejor la
  estacionalidad. No anticipa paros, lluvias ni eventos.
- Las distancias se miden desde el centro de cada hexágono y la red peatonal no tiene en cuenta semáforos, pendientes ni
  la percepción de seguridad de cada cuadra.

## 6. Despliegue

App en Streamlit con mapas de deck.gl (`streamlit run app/app.py`). Funciona sin internet. Incluye:
- mapa 3D de la brecha;
- predicción por zona: para cada barrio, cuántas terminales tiene y cuántas le corresponden, el porqué de cada zona (SHAP)
  y las oportunidades del clasificador;
- simulador de N cajeros (exacto, goloso o K-Means, distancia caminando o en línea recta, con o sin pasajeros por franja
  horaria);
- demanda por hora y pronóstico de entradas al subte para los próximos seis meses;
- tipos de zona;
- carga del listado propio de cajeros, que recalcula cobertura y ubicaciones.

Una terminal nueva que alcance el promedio nacional suma unas 3.242 extracciones por mes, por unos 243 millones de pesos de efectivo (monto
promedio de julio 2025). El ingreso por operación es un dato interno: la app lo recibe como parámetro.

**Próximos pasos:**
- sumar más años de molinetes al pronóstico (el notebook 00c los baja cuando el catálogo de BA Data responde) y variables
  de clima y eventos;
- reportar a BA Data el error de fechas del archivo de molinetes de 2025;
- correr el modelo con el listado actual y las transacciones por terminal;
- sumar restricciones de costo y de seguridad por sitio a la optimización;
- validar la hipótesis horaria (carga de efectivo antes de los picos) con transacciones por hora.

## Uso de herramientas de IA

Usamos asistentes de inteligencia artificial como apoyo para programar, revisar el código y redactar. El grupo definió el
problema, eligió las fuentes y los métodos, y revisó el código y los resultados.

## Referencias

- BCRA (2025). *Informe Mensual de Pagos Minoristas*, agosto de 2025 (datos de julio de 2025).
- Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126-139.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Breiman, L., Friedman, J., Olshen, R. y Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785-794.
- Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*.
- Church, R. y ReVelle, C. (1974). The maximal covering location problem. *Papers of the Regional Science Association*, 32, 101-118.
- Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269-271.
- Ester, M., Kriegel, H.-P., Sander, J. y Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *Proceedings of KDD-96*, 226-231.
- Fayyad, U., Piatetsky-Shapiro, G. y Smyth, P. (1996). From data mining to knowledge discovery in databases. *AI Magazine*, 17(3).
- Friedman, J. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5).
- Hubert, L. y Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2, 193-218.
- Huangfu, Q. y Hall, J. (2018). Parallelizing the dual revised simplex method. *Mathematical Programming Computation*, 10, 119-142.
- Hyndman, R. y Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3.ª ed.). OTexts.
- INDEC (2024). *Censo Nacional de Población, Hogares y Viviendas 2022*, base por radio censal.
- Jolliffe, I. y Cadima, J. (2016). Principal component analysis: a review and recent developments. *Philosophical Transactions of the Royal Society A*, 374(2065), 20150202.
- Kaiser, H. (1960). The application of electronic computers to factor analysis. *Educational and Psychological Measurement*, 20(1), 141-151.
- Koenker, R. y Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33-50.
- Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.
- Lundberg, S. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
- Lundberg, S. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.
- McCullagh, P. y Nelder, J. (1989). *Generalized Linear Models* (2.ª ed.). Chapman & Hall.
- Nemhauser, G., Wolsey, L. y Fisher, M. (1978). An analysis of approximations for maximizing submodular set functions. *Mathematical Programming*, 14, 265-294.
- Openshaw, S. (1984). *The Modifiable Areal Unit Problem*. Geo Books.
- Roberts, D. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913-929.
- Uber Technologies (2018). *H3: Uber's Hexagonal Hierarchical Spatial Index*.
- Varma, S. y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*, 7, 91.
- Wang, R. y Strong, D. (1996). Beyond accuracy: what data quality means to data consumers. *Journal of Management Information Systems*, 12(4).
- Fuentes de datos: GCBA (BA Data), INDEC, SBASE, © colaboradores de OpenStreetMap.
"""
(RAIZ / "docs" / "informe.md").write_text(texto, encoding="utf-8")
print("ok docs/informe.md")
