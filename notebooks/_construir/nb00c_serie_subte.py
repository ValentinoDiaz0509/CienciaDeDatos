"""Genera notebooks/00c_serie_diaria_subte.ipynb (correr: python notebooks/_construir/nb00c_serie_subte.py)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SALIDA = Path(__file__).resolve().parents[1] / "00c_serie_diaria_subte.ipynb"

cells = [
    md("""# 00c · Serie diaria del subte (para el pronóstico)

Baja de BA Data los **molinetes del subte de 2022 a 2026** (pasajeros cada 15 minutos por molinete) y los resume a
**pasajeros por día y estación** y **por día y hora en toda la red**. Con esa serie se entrena el pronóstico del
notebook 05.

**En Colab:** *Entorno de ejecución → Ejecutar todo*. Cada año pesa varios cientos de MB, así que tarda entre
**20 y 40 minutos**. Al final descarga `serie_subte.zip` (unos pocos MB): pasáselo al grupo para guardarlo en
`data/base/`.

Si una sesión anterior quedó abierta, primero *Entorno de ejecución → Desconectar y borrar entorno de ejecución*."""),
    code("""import subprocess, sys
from pathlib import Path

EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    RAIZ = Path("/content/CienciaDeDatos")
    if not RAIZ.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        "https://github.com/ValentinoDiaz0509/CienciaDeDatos.git", str(RAIZ)], check=True)
    else:
        subprocess.run(["git", "-C", str(RAIZ), "pull", "-q"], check=False)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

sys.path.insert(0, str(RAIZ))
import importlib
from src import descarga as d
d = importlib.reload(d)
d.DIR_RAW.mkdir(parents=True, exist_ok=True)
print("Los datos van a:", d.DIR_RAW)"""),
    md("Qué archivos hay en el catálogo de BA Data para cada año:"),
    code("""ANIOS = range(2022, 2027)
for anio, urls in d.recursos_molinetes(ANIOS).items():
    print(anio, len(urls), "archivo(s)")
print(d.ESTADO.get("catalogo_molinetes"))"""),
    md("Descarga y resumen (es el paso largo):"),
    code("""destino = d.descargar_serie_subte(ANIOS)
d.resumen();"""),
    code("""if EN_COLAB:
    from google.colab import files
    files.download(str(destino))"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, SALIDA)
print("ok", SALIDA)
