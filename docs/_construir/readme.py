"""Genera README.md a partir de los resultados (python docs/_construir/readme.py)."""
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from src.estilo import num_es as n  # noqa: E402

R = json.loads((RAIZ / "data/processed/resumen.json").read_text(encoding="utf-8"))
cam = R["metodo_distancia"] == "caminando"
COLAB = "https://colab.research.google.com/github/ValentinoDiaz0509/CienciaDeDatos/blob/main/notebooks"

texto = f"""# ¿Dónde faltan cajeros Link en CABA?

**TPO de Ciencia de Datos · UADE · 2º cuatrimestre 2026**

Cruzamos dónde vive y circula la gente con dónde están los cajeros automáticos para responder una pregunta de negocio:
**¿en qué zonas de la Ciudad le conviene a Red Link sumar terminales?**

## Resultados en una mirada

| | |
|---|---|
| Vecinos con un **Link** a menos de 500 m ({"caminando" if cam else "en línea recta"}) | **{n(100 * R["cobertura_link"])} %** |
| Vecinos con un **Banelco** a menos de 500 m | **{n(100 * R["cobertura_banelco"])} %** |
| Vecinos sin un Link a distancia caminable | **{n(R["residentes_sin_link"])}** |
| Con **20 cajeros** en las ubicaciones óptimas | Link pasa al **{n(100 * R["cobertura_link_con_20"], 1)} %** (+{n(R["nuevos_20_exacto"])} vecinos) |
| Modelo elegido (validación espacial) | {R["modelo_elegido"]}, D² = {n(R["d2_espacial"], 2)} |

Los cajeros **siguen al comercio, no a los vecinos**. Hay dos oportunidades: **competir** en los corredores comerciales del
norte y el centro, donde la demanda justifica más terminales Link de las que hay, y **cubrir** los barrios residenciales del
oeste y el sur, donde no hay ningún Link cerca. El detalle está en el [informe técnico](docs/informe.md).

## Cómo ver la app

### Opción 1 · En tu computadora (la que hay que usar en la defensa: funciona sin internet)

Necesitás [Python 3.10 o superior](https://www.python.org/downloads/) y [Git](https://git-scm.com/downloads).

```bash
git clone https://github.com/ValentinoDiaz0509/CienciaDeDatos.git
cd CienciaDeDatos
python -m venv .venv
# Windows:      .venv\\Scripts\\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Se abre sola en el navegador, en `http://localhost:8501`. Para cerrarla, `Ctrl + C` en la terminal.

### Opción 2 · Publicada en internet (para compartirla con el grupo o el profe)

1. Entrá a [share.streamlit.io](https://share.streamlit.io) con tu cuenta de GitHub.
2. *Create app* → elegí el repositorio `ValentinoDiaz0509/CienciaDeDatos`, la rama `main` y el archivo `app/app.py`.
3. *Deploy*. En unos minutos queda una dirección pública para abrir desde cualquier navegador.

La app tiene seis secciones: panorama (mapa 3D de la brecha), simulador de cajeros nuevos, demanda por hora, tipos de zona,
"usá tus datos" (cargar el listado actual de cajeros) y cómo funciona.

## Entregables de la consigna

| # | Entregable | Dónde está |
|---|---|---|
| 1 | Dominio y problema | [informe §1](docs/informe.md#1-comprensión-del-negocio) |
| 2 | Arquitectura de la solución | [diagrama](docs/figuras/arquitectura.png) e [informe §3](docs/informe.md#3-preparación-de-los-datos) |
| 3 | Análisis exploratorio | [`notebooks/01_calidad_y_eda.ipynb`](notebooks/01_calidad_y_eda.ipynb) |
| 4 | Técnica de minería y por qué | [`notebooks/02_modelos.ipynb`](notebooks/02_modelos.ipynb) y [`03_demanda_por_hora.ipynb`](notebooks/03_demanda_por_hora.ipynb) |
| 5 | Conclusión con visualización | [informe](docs/informe.md#resumen-ejecutivo), notebook 02 §8 y la presentación |
| 6 | Aplicación funcional | [`app/app.py`](app/app.py) |

La [guía de defensa](docs/defensa.md) repasa la consigna punto por punto e incluye roles, guion de 15 minutos, checklist y
preguntas probables.

## Metodología (CRISP-DM)

![Arquitectura](docs/figuras/arquitectura.png)

1. **Datos:** seis fuentes públicas: BA Data (GCBA), Censo 2022 (INDEC), molinetes del subte (SBASE) y OpenStreetMap.
   Calidad medida con las seis dimensiones de DAMA.
2. **Preparación:** grilla de {n(R["hexagonos"])} hexágonos H3 (≈3 × 3 manzanas), interpolación areal del censo y áreas de
   influencia de ≈400 m. {"Distancias caminando por la red peatonal de OpenStreetMap (Dijkstra)." if cam else "Distancias en línea recta."}
3. **Modelos:**
   - regresión de conteos con validación cruzada espacial (seis modelos y un ensamble), para medir la brecha;
   - K-Means, para los tipos de zona y de estación de subte;
   - cobertura máxima resuelta de forma exacta (programación lineal entera con HiGHS) y con un algoritmo goloso.
4. **Evaluación:** validación temporal con los cajeros que aparecieron en OpenStreetMap después de 2017.
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
│   ├── 01_calidad_y_eda.ipynb      # calidad de datos y análisis exploratorio
│   ├── 02_modelos.ipynb            # regresión, clustering, optimización, validación
│   └── 03_demanda_por_hora.ipynb   # perfiles horarios del subte
├── src/
│   ├── descarga.py  features.py  modelos.py  red.py  horario.py  pipeline.py  estilo.py
└── requirements.txt
```

## Reproducir todo desde cero

```bash
pip install -r requirements.txt
python -m src.pipeline                      # variables, modelos y resultados (≈1 minuto)
for f in notebooks/_construir/nb0*.py; do python "$f"; done   # regenera los notebooks (textos con los números nuevos)
python docs/_construir/informe.py && python docs/_construir/readme.py
```

Para volver a bajar las fuentes: [00 · Descarga]({COLAB}/00_descarga_datos.ipynb) y [00b · Red peatonal]({COLAB}/00b_red_peatonal.ipynb)
en Colab (*Entorno de ejecución → Ejecutar todo*).

## Fuentes y licencias

- Cajeros automáticos, barrios, usos del suelo y molinetes: [BA Data](https://data.buenosaires.gob.ar), Gobierno de la Ciudad de Buenos Aires.
- Censo Nacional 2022 por radio censal: INDEC, vía [censoargentino](https://github.com/pedroorden/censoargentino).
- Comercios, estaciones y red peatonal: © colaboradores de [OpenStreetMap](https://www.openstreetmap.org/copyright) (ODbL).
- Extracciones por terminal: BCRA, *Informe Mensual de Pagos Minoristas*, agosto 2025.

**Aviso:** el listado público de cajeros corresponde a un relevamiento de alrededor de 2017. Solo se usan datos públicos;
ningún dato interno de Red Link.
"""
(RAIZ / "README.md").write_text(texto, encoding="utf-8")
print("ok README.md")
