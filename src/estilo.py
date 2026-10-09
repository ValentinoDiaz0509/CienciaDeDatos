"""
estilo.py
---------
Colores y formato compartidos por los notebooks, la app y la presentación, para que
todo se lea como un mismo sistema.

- Identidad (categórico, orden fijo): Link = azul, Banelco = naranja, sugerencias = verde agua.
  Paleta validada para daltonismo (ΔE CVD ≥ 9 entre pares adyacentes).
- Magnitud (secuencial): un solo tono, de claro a oscuro.
- Brecha (divergente): azul = sobran terminales, gris = equilibrio, rojo = faltan.
"""

from __future__ import annotations

import numpy as np

LINK = "#2a78d6"
BANELCO = "#eb6834"
SUGERIDO = "#1baf7a"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_SUAVE = "#898781"
GRILLA = "#e1e0d9"
EJE = "#c3c2b7"
FONDO = "#fcfcfb"

SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SECUENCIAL_NARANJA = ["#fde3d6", "#f8bfa4", "#f39a72", "#eb6834", "#c2501f", "#963c15", "#6b2a0d"]
DIVERGENTE = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a9a8", "#e34948", "#a32524"]
# Tipos de zona (hasta 5): orden fijo de la paleta categórica validada
CATEGORICO = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def hex_a_rgb(h: str, alfa: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [alfa]


def _interpolar(valores: np.ndarray, paradas: list[str], vmin: float, vmax: float, alfa: int) -> list[list[int]]:
    rgb = np.array([hex_a_rgb(p)[:3] for p in paradas], dtype=float)
    pos = np.linspace(0, 1, len(paradas))
    t = np.clip((np.asarray(valores, float) - vmin) / ((vmax - vmin) or 1), 0, 1)
    t = np.nan_to_num(t, nan=0.5)
    canales = [np.interp(t, pos, rgb[:, c]) for c in range(3)]
    return [[int(r), int(g), int(b), alfa] for r, g, b in zip(*canales)]


def color_secuencial(valores, vmax: float | None = None, alfa: int = 200, paradas=SECUENCIAL):
    v = np.asarray(valores, float)
    vmax = vmax if vmax is not None else np.nanquantile(v, 0.98)
    return _interpolar(v, paradas, 0, vmax, alfa)


def color_divergente(valores, limite: float | None = None, alfa: int = 210):
    """Centrado en 0: el mismo límite a cada lado para que el gris signifique 'equilibrio'."""
    v = np.asarray(valores, float)
    limite = limite if limite is not None else np.nanquantile(np.abs(v), 0.95)
    return _interpolar(v, DIVERGENTE, -limite, limite, alfa)


def mpl_estilo() -> None:
    """Estilo sobrio para matplotlib: grilla fina, sin bordes superiores, tipografía sans."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "axes.titlecolor": TINTA,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.family": "sans-serif",
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })


def cmap_divergente():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("brecha", DIVERGENTE)


def cmap_secuencial(paradas=SECUENCIAL):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("magnitud", paradas)
