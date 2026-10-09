"""Genera notebooks/00b_red_peatonal.ipynb (correr: python notebooks/_construir/nb00b_red.py)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SALIDA = Path(__file__).resolve().parents[1] / "00b_red_peatonal.ipynb"

cells = [
    md("""# 00b · Red peatonal de CABA (OpenStreetMap)

Baja las calles y senderos caminables de la Ciudad para medir distancias **caminando** en lugar de en línea recta.
Usa [OSMnx](https://osmnx.readthedocs.io) (Boeing, 2017), que consulta OpenStreetMap.

**En Colab:** *Entorno de ejecución → Ejecutar todo*. Tarda entre 3 y 6 minutos y al final descarga `red_peatonal.zip`
(unos pocos MB). Si una sesión anterior quedó abierta, primero *Entorno de ejecución → Desconectar y borrar*."""),
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "osmnx>=2.0"], check=True)
else:
    RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

sys.path.insert(0, str(RAIZ))
import importlib
from src import descarga as d
d = importlib.reload(d)
d.DIR_RAW.mkdir(parents=True, exist_ok=True)
print("Los datos van a:", d.DIR_RAW)"""),
    code("d.descargar_red_peatonal()"),
    code("""import zipfile
destino = RAIZ / "data" / "red_peatonal.zip"
with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for nombre in ["red-peatonal-nodos.parquet", "red-peatonal-aristas.parquet"]:
        z.write(d.DIR_RAW / nombre, arcname=nombre)
print(f"Listo: {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
if EN_COLAB:
    from google.colab import files
    files.download(str(destino))"""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["colab"] = {"provenance": []}
nbf.write(nb, SALIDA)
print("ok", SALIDA)
