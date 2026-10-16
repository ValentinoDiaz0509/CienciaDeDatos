"""Genera README.md a partir de los resultados (python docs/_construir/readme.py)."""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.estilo import num_es as n  # noqa: E402

R = json.loads((RAIZ / "data/processed/resumen.json").read_text(encoding="utf-8"))
import pandas as pd  # noqa: E402

BP = pd.read_csv(RAIZ / "data/processed/barrios_prediccion.csv", index_col="barrio")
TOP = BP[BP["diferencia_link"] > 0].head(3)
faltan_txt = ", ".join(f"{b} ({n(r.diferencia_link)})" for b, r in TOP.iterrows())

EX = pd.read_csv(RAIZ / "data/processed/recomendaciones_exactas_20.csv")
bar_tipo = lambda z: ", ".join(EX[EX["tipo_zona"] == z]["barrio"].value_counts().index[:3])  # noqa: E731
cam = R["metodo_distancia"] == "caminando"
COLAB = "https://colab.research.google.com/github/ValentinoDiaz0509/CienciaDeDatos/blob/main/notebooks"

fila_recta = (f"\n| Link con la misma regla, medida en línea recta | {n(100 * R['cobertura_link_recta'])} % "
              "(la línea recta sobreestima la cobertura) |" if cam else "")

_PR = RAIZ / "data/processed/pronostico_resumen.json"
PR = json.loads(_PR.read_text(encoding="utf-8")) if _PR.exists() else None
fila_pron = (f"\n| Pronóstico del subte ({PR['modelo_elegido']}, validado en meses que no vio) | erra un "
             f"{n(PR['mape_promedio'][PR['modelo_elegido']], 1)} % por día (repetir el año anterior: "
             f"{n(PR['mape_promedio']['Mismo día del año anterior'], 1)} %); {n(PR['pax_habil_primer_mes'] / 1e3)} mil entradas por "
             f"día hábil en {PR['primer_mes_nombre']} y {n(PR['pax_habil_mes_minimo'] / 1e3)} mil en {PR['mes_minimo_nombre']} |"
             if PR else "")
texto_pron = (" Un segundo modelo pronostica cuánta gente va a entrar al subte cada día de los próximos seis meses, para "
              "planificar la carga de efectivo y el mantenimiento." if PR else "")

fila_igualar = (f"\n| Cajeros nuevos para igualar la cobertura de Banelco | **{R['cajeros_para_igualar_banelco']}** |"
                if R.get("cajeros_para_igualar_banelco") else "")

texto = f"""# ¿Dónde faltan cajeros Link en CABA?

**TPO de Ciencia de Datos · UADE · 2º cuatrimestre 2026**

Cruzamos dónde vive y circula la gente con dónde están los cajeros automáticos para responder una pregunta de negocio:
**¿en qué zonas de la Ciudad le conviene a Red Link sumar terminales?**

## Resultados en una mirada

| | |
|---|---|
| Vecinos con un **Link** a menos de 500 m ({"caminando" if cam else "en línea recta"}) | **{n(100 * R["cobertura_link"])} %** |
| Vecinos con un **Banelco** a menos de 500 m | **{n(100 * R["cobertura_banelco"])} %** |{fila_recta}
| Vecinos sin un Link a distancia caminable | **{n(R["residentes_sin_link"])}** |
| Con **20 cajeros** en las ubicaciones óptimas | Link pasa al **{n(100 * R["cobertura_link_con_20"], 1)} %** (+{n(R["nuevos_20_exacto"])} vecinos) |{fila_igualar}
| Terminales Link que faltan según su demanda (machine learning) | {faltan_txt} |
| Zonas sin ningún cajero que deberían tener uno (clasificador) | **{n(R["zonas_sin_cajero_que_deberian"])}**, con {n(R["residentes_zonas_sin_cajero_que_deberian"])} vecinos |
| Modelo de regresión elegido (validación espacial anidada) | {R["modelo_elegido"]}: D² = {n(R["d2_espacial_promedio"], 2)}, error medio de {n(R["mae_espacial"], 1)} terminales por zona |
| Clasificador elegido | {R["clasificador_elegido"]}: AUC = {n(R["auc_clasificador"], 2)} |{fila_pron}

Los cajeros **siguen al comercio, no a los vecinos**. Un modelo de machine learning (árboles, Random Forest y XGBoost,
validados en comunas que no vieron) estima cuántos cajeros le corresponden a cada zona según su demanda. Hay dos
oportunidades: **competir** donde la demanda justifica más terminales Link de las que hay ({", ".join(TOP.index)}), y
**cubrir** los barrios residenciales ({bar_tipo("Residencial denso")}) y el sur ({bar_tipo("Mayor vulnerabilidad social")}),
donde no hay un Link a distancia caminable.{texto_pron} El detalle está en el [informe técnico](docs/informe.md).

## Cómo ver la app

### Opción 1 · En tu computadora (la que hay que usar en la defensa: funciona sin internet)

**1. Instalar Python (una sola vez).** Hace falta Python 3.10 o superior; recomendamos **3.12**. En Windows, lo más simple es
abrir una terminal y escribir:

```bat
winget install -e --id Python.Python.3.12
```

También se puede bajar de [python.org](https://www.python.org/downloads/windows/): en el instalador hay que marcar
**"Add python.exe to PATH"**. Después de instalarlo, cerrar la terminal y abrir una nueva.

**2. Bajar el repositorio.** Con [Git](https://git-scm.com/downloads):
`git clone https://github.com/ValentinoDiaz0509/CienciaDeDatos.git`, o desde GitHub con *Code → Download ZIP*.

**3. Abrir la app.**
- **Windows:** doble clic en **`abrir_app.bat`**.
- **macOS / Linux:** en una terminal, dentro de la carpeta, `./abrir_app.sh`.

La primera vez crea el entorno e instala lo necesario (tarda unos minutos); las siguientes abre en segundos. La app queda en
`http://localhost:8501` y el navegador se abre solo. Para cerrarla, cerrar la ventana de la terminal.

<details>
<summary>Paso a paso a mano (si el archivo no funciona)</summary>

En Windows (cmd), dentro de la carpeta del repositorio:

```bat
py -3.12 -m venv .venv
.venv\\Scripts\\activate
pip install -r app\\requirements.txt
streamlit run app\\app.py
```

En macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/app.py
```

Después, abrir `http://localhost:8501` en el navegador. Para correr también los notebooks, instalar el archivo completo:
`pip install -r requirements.txt`.
</details>

### Opción 2 · Publicada en internet (para compartirla con el grupo o el profe)

1. Entrá a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
2. *Create app* → elegí el repositorio `ValentinoDiaz0509/CienciaDeDatos`, la rama `main` y el archivo `app/app.py`.
3. *Deploy*. En unos minutos queda una dirección pública para abrir desde cualquier navegador.

La app tiene siete secciones: panorama (mapa 3D de la brecha), predicción por zona (cuántas terminales tiene cada barrio,
cuántas le corresponden y por qué), simulador de cajeros nuevos, demanda por hora y pronóstico del subte, tipos de zona,
"usá tus datos" (cargar el listado actual de cajeros) y cómo funciona.

![Predicción por zona: Belgrano tiene 30 terminales Link y le corresponden 52; el mapa marca sus zonas y a la derecha está el porqué de la zona elegida](docs/figuras/app_prediccion.png)

![Simulador de expansión: elegís cuántos cajeros sumar y a qué distancia, y la app devuelve dónde ponerlos](docs/figuras/app_simulador.png)

| Panorama | Demanda por hora |
|---|---|
| ![Mapa 3D de la brecha Link por hexágono](docs/figuras/app_panorama.png) | ![Pasajeros del subte por hora y tipo de estación](docs/figuras/app_demanda_hora.png) |

![Pronóstico de entradas al subte para los próximos seis meses, con su intervalo del 80 % y el día hábil promedio de cada mes](docs/figuras/app_pronostico.png)

## Entregables de la consigna

| # | Entregable | Dónde está |
|---|---|---|
| 1 | Dominio y problema | [informe §1](docs/informe.md#1-comprensión-del-negocio) |
| 2 | Arquitectura de la solución | [diagrama](docs/figuras/arquitectura.png) e [informe §3](docs/informe.md#3-preparación-de-los-datos) |
| 3 | Análisis exploratorio | [`notebooks/01_calidad_y_eda.ipynb`](notebooks/01_calidad_y_eda.ipynb) |
| 4 | Técnica de minería y por qué | [`02_machine_learning.ipynb`](notebooks/02_machine_learning.ipynb), [`03_optimizacion.ipynb`](notebooks/03_optimizacion.ipynb), [`04_demanda_por_hora.ipynb`](notebooks/04_demanda_por_hora.ipynb) y [`05_pronostico_subte.ipynb`](notebooks/05_pronostico_subte.ipynb) |
| 5 | Conclusión con visualización | [informe](docs/informe.md#resumen-ejecutivo), notebooks 02 §9 y 03, y la presentación |
| 6 | Aplicación funcional | [`app/app.py`](app/app.py) |

La [guía de defensa](docs/defensa.md) repasa la consigna punto por punto e incluye roles, guion de 15 minutos, checklist y
preguntas probables.

## Metodología (CRISP-DM)

![Arquitectura](docs/figuras/arquitectura.png)

1. **Datos:** seis fuentes públicas: BA Data (GCBA), Censo 2022 (INDEC), molinetes del subte (SBASE) y OpenStreetMap.
   Calidad medida con las seis dimensiones de DAMA.
2. **Preparación:** grilla de {n(R["hexagonos"])} hexágonos H3 (≈3 × 3 manzanas), interpolación areal del censo y áreas de
   influencia de ≈400 m. {"Distancias caminando por la red peatonal de OpenStreetMap (Dijkstra)." if cam else "Distancias en línea recta."}
3. **Modelos**, siguiendo los cuatro niveles de análisis:
   - **predictivo (machine learning):** regresión (7 modelos: de una línea de base a árbol de decisión, Random Forest,
     XGBoost y un ensamble) y clasificación (4 modelos) con **validación cruzada espacial anidada**: GroupKFold por comuna
     y GridSearchCV adentro; curva de sobreajuste y error OOB;
   - **diagnóstico:** valores SHAP, importancia de Gini y coeficientes de la regresión de Poisson;
   - **prescriptivo:** K-Means para los tipos de zona y de estación, y cobertura máxima resuelta de forma exacta
     (programación lineal entera con HiGHS) y con un algoritmo goloso;
   - **pronóstico:** Random Forest y XGBoost sobre el calendario para la demanda diaria del subte, validados hacia
     adelante en meses que no vieron (notebook 05).
4. **Evaluación:** métricas en comunas no vistas (D² de Poisson, MAE, AUC), MAPE del pronóstico en meses no vistos y
   validación temporal con los cajeros que aparecieron en OpenStreetMap después de 2017.
5. **Despliegue:** app de Streamlit con mapas de deck.gl, que funciona sin conexión.

## Estructura del repositorio

```text
├── app/app.py                      # la app (streamlit run app/app.py)
├── data/
│   ├── base/                       # fuentes livianas, versionadas (ver LEEME.md)
│   ├── processed/                  # resultados del pipeline (los usa la app)
│   └── raw/                        # descargas completas (no se suben)
├── docs/
│   ├── informe.md                  # informe técnico (CRISP-DM)
│   ├── defensa.md                  # guía para la presentación
│   └── figuras/                    # gráficos y diagrama de arquitectura
├── notebooks/
│   ├── 00_descarga_datos.ipynb     # baja todas las fuentes (Colab)
│   ├── 00b_red_peatonal.ipynb      # baja la red de calles (Colab)
│   ├── 00c_serie_diaria_subte.ipynb # baja la serie diaria del subte (Colab)
│   ├── 01_calidad_y_eda.ipynb      # calidad de datos y análisis exploratorio
│   ├── 02_machine_learning.ipynb   # regresión y clasificación: árboles, Random Forest, XGBoost, SHAP
│   ├── 03_optimizacion.ipynb       # tipos de zona (K-Means) y dónde sumar cajeros
│   ├── 04_demanda_por_hora.ipynb   # perfiles horarios del subte
│   └── 05_pronostico_subte.ipynb   # pronóstico diario de pasajeros
├── src/
│   ├── descarga.py  features.py  modelos.py  ml.py  pronostico.py  red.py  horario.py  pipeline.py
└── requirements.txt
```

## Reproducir todo desde cero

```bash
pip install -r requirements.txt
python -m src.pipeline                      # variables, modelos y resultados (≈6 minutos)
python -m src.pronostico                    # pronóstico del subte (segundos)
for f in notebooks/_construir/nb0*.py; do python "$f"; done   # regenera los notebooks (textos con los números nuevos)
python docs/_construir/informe.py && python docs/_construir/readme.py
```

Para volver a bajar las fuentes: [00 · Descarga]({COLAB}/00_descarga_datos.ipynb), [00b · Red peatonal]({COLAB}/00b_red_peatonal.ipynb)
y [00c · Serie diaria del subte]({COLAB}/00c_serie_diaria_subte.ipynb) en Colab (*Entorno de ejecución → Ejecutar todo*).
La serie diaria del subte ya está en `data/base/`, así que el pronóstico ([05]({COLAB}/05_pronostico_subte.ipynb)) corre
directo, también en Colab.

## Fuentes y licencias

- Cajeros automáticos, barrios, usos del suelo y molinetes: [BA Data](https://data.buenosaires.gob.ar), Gobierno de la Ciudad de Buenos Aires.
- Censo Nacional 2022 por radio censal: INDEC, vía [censoargentino](https://github.com/pedroorden/censoargentino).
- Comercios, estaciones y red peatonal: © colaboradores de [OpenStreetMap](https://www.openstreetmap.org/copyright) (ODbL).
- Extracciones por terminal: BCRA, *Informe Mensual de Pagos Minoristas*, agosto 2025.

**Aviso:** el listado público de cajeros corresponde a un relevamiento de alrededor de 2017. Solo se usan datos públicos;
ningún dato interno de Red Link. Un integrante del grupo trabaja en Red Link: por eso elegimos este dominio, pero el trabajo
no usa información interna de la empresa y sus conclusiones no la representan.

**Uso de IA:** usamos asistentes de inteligencia artificial como apoyo para programar, revisar el código y redactar. El grupo
definió el problema, eligió las fuentes y los métodos, y revisó el código y los resultados.
"""
(RAIZ / "README.md").write_text(texto, encoding="utf-8")
print("ok README.md")
