"""
figura_caminando.py
-------------------
Figura de ejemplo: para una zona concreta, el camino a pie hasta el Link más cercano
comparado con la línea recta. Muestra por qué medir caminando cambia la cobertura.

    python -m src.figura_caminando 89c2e310383ffff
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from . import estilo as e
from . import features as f
from . import red as rp

FIG = f.RAIZ / "docs" / "figuras"


def dibujar(h3_id: str, salida=FIG / "ejemplo_caminando.png") -> dict:
    res = pd.read_parquet(f.DIR_PROC / "resultados_hex.parquet").set_index("h3")
    fila = res.loc[h3_id]
    red = rp.RedPeatonal.desde_archivos()
    caj = f.cargar_cajeros()
    link = caj[caj["red"] == "LINK"].reset_index(drop=True)
    camino, largo, i = red.camino(fila["lon"], fila["lat"], link["lon"], link["lat"])
    origen = rp._metros([fila["lon"]], [fila["lat"]])[0]
    destino = rp._metros([link.loc[i, "lon"]], [link.loc[i, "lat"]])[0]
    recta = float(np.linalg.norm(destino - origen))
    # el Link más cercano en línea recta puede ser otro: lo buscamos para la comparación honesta
    todos = rp._metros(link["lon"], link["lat"])
    j = int(np.argmin(np.linalg.norm(todos - origen, axis=1)))
    destino_recta, recta_min = todos[j], float(np.linalg.norm(todos[j] - origen))

    centro = (origen + destino) / 2
    m = max(np.ptp(np.r_[camino[:, 0], origen[0], destino_recta[0]]), np.ptp(np.r_[camino[:, 1], origen[1], destino_recta[1]])) / 2 + 220
    x0, x1, y0, y1 = centro[0] - m, centro[0] + m, centro[1] - m, centro[1] + m
    a, b = red.xy[red.u], red.xy[red.v]
    vis = ((np.minimum(a[:, 0], b[:, 0]) < x1) & (np.maximum(a[:, 0], b[:, 0]) > x0) &
           (np.minimum(a[:, 1], b[:, 1]) < y1) & (np.maximum(a[:, 1], b[:, 1]) > y0))

    e.mpl_estilo()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(LineCollection(np.stack([a[vis], b[vis]], axis=1), colors="#c3c2b7", linewidths=0.8))
    otros = (todos[:, 0] > x0) & (todos[:, 0] < x1) & (todos[:, 1] > y0) & (todos[:, 1] < y1)
    ax.scatter(todos[otros, 0], todos[otros, 1], s=28, color=e.LINK, alpha=0.35, zorder=2, label="Otros cajeros Link")
    ax.plot(camino[:, 0], camino[:, 1], color=e.LINK, linewidth=3.2, zorder=3,
            label=f"Caminando por las calles: {e.num_es(largo)} m")
    ax.plot([origen[0], destino_recta[0]], [origen[1], destino_recta[1]], color=e.BANELCO, linewidth=2, linestyle=(0, (5, 4)),
            zorder=3, label=f"En línea recta: {e.num_es(recta_min)} m")
    ax.scatter(*origen, s=130, color="#0b0b0b", zorder=4)
    ax.annotate("Centro de la zona", origen, xytext=(10, 8), textcoords="offset points", fontsize=10)
    if i != j:  # el más cercano en línea recta es otro cajero: cuánto queda a pie
        a_pie = float(red.distancia_a_mas_cercano([fila["lon"]], [fila["lat"]], [link.loc[j, "lon"]], [link.loc[j, "lat"]])[0])
        ax.scatter(*destino_recta, s=90, color=e.BANELCO, edgecolor="white", linewidth=1.5, zorder=4)
        ax.annotate(f"Link más cercano en línea recta\n(a pie: {e.num_es(a_pie)} m)", destino_recta, xytext=(10, -22),
                    textcoords="offset points", fontsize=10, color=e.BANELCO)
    ax.scatter(*destino, s=110, color=e.LINK, edgecolor="white", linewidth=1.5, zorder=4)
    ax.annotate("Link más cercano a pie", destino, xytext=(10, -14), textcoords="offset points", fontsize=10, color=e.LINK)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    titulo = (f"{fila['barrio']}: el mismo cajero queda {e.num_es(largo / recta_min, 1)} veces más lejos a pie" if i == j else
              f"{fila['barrio']}: a pie, el Link más cercano queda {e.num_es(largo / recta_min, 1)} veces más lejos")
    ax.set_title(titulo, loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, fontsize=9.5)
    ax.text(x1, y0 + 30, "Calles: © colaboradores de OpenStreetMap", ha="right", fontsize=8, color="#898781")
    fig.tight_layout()
    fig.savefig(salida, dpi=200)
    plt.close(fig)
    return {"barrio": fila["barrio"], "caminando_m": largo, "recta_m": recta_min, "mismo_cajero": i == j, "recta_al_de_a_pie_m": recta}


if __name__ == "__main__":
    print(dibujar(sys.argv[1] if len(sys.argv) > 1 else "89c2e310383ffff"))
