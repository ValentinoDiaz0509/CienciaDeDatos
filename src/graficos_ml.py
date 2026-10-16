"""
graficos_ml.py
--------------
Gráficos del notebook de machine learning que llevan más código del que conviene mostrar en
una celda: el árbol de decisión dibujado con nombres de negocio, la explicación de una zona,
las curvas ROC, la matriz de confusión y la comparación "tiene / le corresponden" por barrio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import estilo as e


# --------------------------------------------------------------------------
# Árbol de decisión
# --------------------------------------------------------------------------
def _umbral(variable: str, valor: float) -> str:
    if variable.startswith("pct_") or variable == "tasa_empleo_k1":
        return f"{e.num_es(100 * valor, 0)} %"
    if variable == "dist_centro_km":
        return f"{e.num_es(valor, 1)} km"
    if variable == "dist_subte_m":
        return f"{e.num_es(valor, 0)} m"
    return e.num_es(np.floor(valor), 0)  # conteos: x <= 12,5 es lo mismo que x <= 12


def arbol(modelo, nombres: dict, ax, clase_positiva: str = "con cajero") -> None:
    """Dibuja un árbol de clasificación chico: cada caja dice la pregunta, cuántas zonas llegan y qué parte es positiva."""
    t = modelo.tree_
    cols = list(modelo.feature_names_in_)
    pos, siguiente = {}, [0]

    def recorrer(n, d):
        if t.children_left[n] == -1:
            pos[n] = (siguiente[0], -d)
            siguiente[0] += 1
        else:
            recorrer(t.children_left[n], d + 1)
            recorrer(t.children_right[n], d + 1)
            pos[n] = ((pos[t.children_left[n]][0] + pos[t.children_right[n]][0]) / 2, -d)

    recorrer(0, 0)
    cmap = e.cmap_secuencial()
    for n, (x, y) in pos.items():
        for hijo, etiqueta in ((t.children_left[n], "sí"), (t.children_right[n], "no")):
            if hijo != -1:
                hx, hy = pos[hijo]
                ax.plot([x, hx], [y, hy], color=e.EJE, linewidth=1.2, zorder=1)
                ax.text((x + hx) / 2, (y + hy) / 2, etiqueta, fontsize=8, color=e.TINTA_2, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=e.FONDO, edgecolor="none"))
        valores = np.asarray(t.value[n][0], float)
        prop = valores[1] / valores.sum()
        total = int(t.n_node_samples[n])
        if t.children_left[n] == -1:
            texto = f"{e.num_es(total)} zonas\n{e.num_es(100 * prop)} % {clase_positiva}"
        else:
            v = cols[t.feature[n]]
            texto = (f"{nombres.get(v, v)}\n≤ {_umbral(v, t.threshold[n])}\n"
                     f"{e.num_es(total)} zonas · {e.num_es(100 * prop)} % {clase_positiva}")
        color = cmap(0.15 + 0.75 * prop)
        oscuro = prop > 0.55
        ax.text(x, y, texto, ha="center", va="center", fontsize=8, zorder=3, color="#ffffff" if oscuro else e.TINTA,
                linespacing=1.3, bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.6", facecolor=color,
                                           edgecolor=e.FONDO, linewidth=2))
    ax.set_xlim(-0.6, siguiente[0] - 0.4)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.45, 0.45)
    ax.set_axis_off()


# --------------------------------------------------------------------------
# Explicación de una zona (aportes multiplicativos)
# --------------------------------------------------------------------------
def explicacion(tabla_aportes: pd.DataFrame, ax, titulo: str) -> None:
    """
    Barras horizontales en escala logarítmica: cuánto multiplica cada variable lo que le corresponde
    a la zona respecto de una zona típica de la ciudad.
    """
    d = tabla_aportes.iloc[::-1]
    colores = [e.DIVERGENTE[-2] if a > 0 else e.DIVERGENTE[1] for a in d["aporte_log"]]
    if "resto" in set(d["variable"]):
        colores = [e.EJE if v == "resto" else c for v, c in zip(d["variable"], colores)]
    y = np.arange(len(d))
    ax.barh(y, d["aporte_log"], color=colores, height=0.62)
    ax.axvline(0, color=e.TINTA_SUAVE, linewidth=1)
    etiquetas = []
    for _, r in d.iterrows():
        if r["variable"] == "resto":
            etiquetas.append(r["nombre"])
        else:
            etiquetas.append(f"{r['nombre']}: {valor_legible(r['variable'], r['valor'])}")
    ax.set_yticks(y, etiquetas)
    lim = max(np.abs(d["aporte_log"]).max() * 1.35, np.log(1.6))
    ax.set_xlim(-lim, lim)
    for yi, a in zip(y, d["aporte_log"]):
        ax.text(a + (0.03 if a >= 0 else -0.03) * lim * 3, yi, f"×{e.num_es(np.exp(a), 2)}", va="center",
                ha="left" if a >= 0 else "right", fontsize=9, color=e.TINTA)
    marcas = [m for m in (0.25, 0.5, 0.75, 1, 1.5, 2, 4) if abs(np.log(m)) <= lim]
    ax.set_xticks([np.log(m) for m in marcas],
                  [f"×{e.num_es(m, 2).rstrip('0') if m != int(m) else e.num_es(m)}" for m in marcas])
    ax.set_title(titulo, fontsize=11)
    ax.grid(axis="y", visible=False)


def valor_legible(variable: str, valor: float) -> str:
    """El valor de una variable como se lee en negocio: porcentajes, km, metros o conteos."""
    if pd.isna(valor):
        return ""
    if variable.startswith("pct_") or variable == "tasa_empleo_k1":
        return f"{e.num_es(100 * valor, 0)} %"
    if variable == "dist_centro_km":
        return f"{e.num_es(valor, 1)} km"
    if variable == "dist_subte_m":
        return f"{e.num_es(valor, 0)} m"
    return e.num_es(valor, 0)


# --------------------------------------------------------------------------
# Clasificación
# --------------------------------------------------------------------------
def roc(oof: pd.DataFrame, modelos: list[str], elegido: str, ax) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    for nombre in modelos:
        fpr, tpr, _ = roc_curve(oof["real"], oof[nombre])
        auc = roc_auc_score(oof["real"], oof[nombre])
        principal = nombre == elegido
        ax.plot(fpr, tpr, color=e.LINK if principal else e.TINTA_SUAVE, linewidth=2.4 if principal else 1.2,
                alpha=1 if principal else 0.8, label=f"{nombre} (AUC {e.num_es(auc, 3)})", zorder=3 if principal else 2)
    ax.plot([0, 1], [0, 1], color=e.EJE, linewidth=1, linestyle="--", label="Azar (AUC 0,5)")
    ax.set(xlabel="falsos positivos: zonas sin cajero que el modelo marca", ylabel="verdaderos positivos: zonas con cajero detectadas",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    e.ejes_es(ax, x=1, y=1)


def matriz(cm: pd.DataFrame, ax) -> None:
    """Matriz de confusión con el significado de negocio de cada casillero."""
    valores = cm.to_numpy()
    total = valores.sum()
    textos = [["Bien: sin cajero\ny sin demanda", "Oportunidad: tiene demanda\ny no tiene cajero"],
              ["Tiene cajero con menos\ndemanda de la esperada", "Bien: con cajero\ny con demanda"]]
    cmap = e.cmap_secuencial()
    ax.imshow(valores / valores.max(), cmap=cmap, vmin=-0.15, vmax=1.15)
    for i in range(2):
        for j in range(2):
            v = valores[i, j]
            oscuro = v / valores.max() > 0.55
            ax.text(j, i - 0.12, e.num_es(v), ha="center", va="center", fontsize=16, fontweight="bold",
                    color="#ffffff" if oscuro else e.TINTA)
            ax.text(j, i + 0.22, f"{e.num_es(100 * v / total)} % · " + textos[i][j], ha="center", va="center",
                    fontsize=8, color="#ffffff" if oscuro else e.TINTA_2, linespacing=1.2)
    ax.set_xticks([0, 1], ["Predicho: sin cajero", "Predicho: con cajero"])
    ax.set_yticks([0, 1], ["Real: sin cajero", "Real: con cajero"])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# --------------------------------------------------------------------------
# Barrios: tiene vs. le corresponden
# --------------------------------------------------------------------------
def tiene_vs_corresponden(b: pd.DataFrame, ax, columna_tiene="tiene_link", columna_corr="corresponden_link") -> None:
    d = b.iloc[::-1]
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r[columna_tiene], r[columna_corr]], [yi, yi], color=e.EJE, linewidth=2.5, zorder=1)
    ax.scatter(d[columna_tiene], y, s=70, color=e.TINTA_SUAVE, zorder=2, label="Terminales Link que tiene")
    ax.scatter(d[columna_corr], y, s=70, color=e.LINK, zorder=3, label="Le corresponden según su demanda")
    for yi, (_, r) in zip(y, d.iterrows()):
        dif = r[columna_corr] - r[columna_tiene]
        ax.text(max(r[columna_tiene], r[columna_corr]) + 1.5, yi, f"{'+' if dif >= 0 else '−'}{e.num_es(abs(dif))}",
                va="center", fontsize=9, color=e.TINTA)
    ax.set_yticks(y, d.index)
    ax.set_xlabel("terminales Link")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    e.ejes_es(ax, x=0)
