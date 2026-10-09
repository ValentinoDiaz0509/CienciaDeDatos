"""
red.py
------
Distancias CAMINANDO por la red de calles de OpenStreetMap (bajada con OSMnx en
notebooks/00b_red_peatonal.ipynb), en lugar de en línea recta.

Cada punto (centro de hexágono, cajero, sitio candidato) se "engancha" a la esquina más
cercana de la red; la distancia total es: tramo hasta la red + camino más corto por la red
(algoritmo de Dijkstra) + tramo desde la red.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .features import CRS_METRICO, DIR_BASE

ARCHIVO_NODOS = "red-peatonal-nodos.parquet"
ARCHIVO_ARISTAS = "red-peatonal-aristas.parquet"


def hay_red(base: Path = DIR_BASE) -> bool:
    return (base / ARCHIVO_NODOS).exists() and (base / ARCHIVO_ARISTAS).exists()


def _metros(lon, lat) -> np.ndarray:
    import geopandas as gpd

    p = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(CRS_METRICO)
    return np.column_stack([p.x, p.y])


class RedPeatonal:
    def __init__(self, nodos: pd.DataFrame, aristas: pd.DataFrame):
        self.n = len(nodos)
        indice = pd.Series(np.arange(self.n), index=nodos["osmid"].to_numpy())
        u = indice.reindex(aristas["u"].to_numpy()).to_numpy()
        v = indice.reindex(aristas["v"].to_numpy()).to_numpy()
        ok = ~(np.isnan(u) | np.isnan(v))
        u, v, largo = u[ok].astype(int), v[ok].astype(int), aristas["largo_m"].to_numpy(float)[ok]
        # grafo no dirigido: se camina en ambos sentidos
        self.grafo = coo_matrix((np.r_[largo, largo], (np.r_[u, v], np.r_[v, u])), shape=(self.n, self.n)).tocsr()
        self.xy = _metros(nodos["lon"], nodos["lat"])
        self.arbol = cKDTree(self.xy)
        self.largo_total_km = float(largo.sum() / 1000)

    @classmethod
    def desde_archivos(cls, base: Path = DIR_BASE) -> "RedPeatonal":
        return cls(pd.read_parquet(base / ARCHIVO_NODOS), pd.read_parquet(base / ARCHIVO_ARISTAS))

    def enganchar(self, lon, lat) -> tuple[np.ndarray, np.ndarray]:
        """Esquina más cercana y distancia en línea recta hasta ella."""
        d, i = self.arbol.query(_metros(lon, lat), k=1)
        return i, d

    def distancia_a_mas_cercano(self, destinos_lon, destinos_lat, fuentes_lon, fuentes_lat) -> np.ndarray:
        """
        Para cada destino (p. ej. centro de hexágono), distancia caminando al punto fuente más
        cercano (p. ej. cajero Link). Usa un nodo virtual unido a todas las fuentes con un peso
        igual a su tramo de enganche, así un único Dijkstra resuelve todas las fuentes.
        """
        if len(fuentes_lon) == 0:
            return np.full(len(destinos_lon), np.inf)
        nf, df = self.enganchar(fuentes_lon, fuentes_lat)
        nd, dd = self.enganchar(destinos_lon, destinos_lat)
        virtual = self.n
        extra = coo_matrix((df + 1e-6, (np.full(len(nf), virtual), nf)), shape=(self.n + 1, self.n + 1))
        g = self._ampliado() + extra.tocsr()
        dist = dijkstra(g, directed=True, indices=virtual)
        return dist[nd] + dd

    def _ampliado(self) -> csr_matrix:
        g = self.grafo.tocoo()
        return coo_matrix((g.data, (g.row, g.col)), shape=(self.n + 1, self.n + 1)).tocsr()

    def vecinos_en_radio(self, origenes_lon, origenes_lat, destinos_lon, destinos_lat, radio_m: float,
                         lote: int = 150) -> list[np.ndarray]:
        """
        Para cada origen (sitio candidato), índices de los destinos (hexágonos) a menos de
        `radio_m` caminando. Dijkstra acotado al radio, en lotes para no usar demasiada memoria.
        """
        no, do = self.enganchar(origenes_lon, origenes_lat)
        nd, dd = self.enganchar(destinos_lon, destinos_lat)
        salida = []
        for a in range(0, len(no), lote):
            b = min(a + lote, len(no))
            dist = dijkstra(self.grafo, directed=False, indices=no[a:b], limit=radio_m)
            total = dist[:, nd] + dd[None, :] + do[a:b, None]
            salida.extend(np.flatnonzero(fila <= radio_m) for fila in total)
        return salida


def agregar_distancias_red(tabla: pd.DataFrame, cajeros: pd.DataFrame, red: RedPeatonal) -> pd.DataFrame:
    """Suma a la tabla las distancias caminando al cajero Link, Banelco y de cualquier red más cercano."""
    t = tabla.copy()
    for nombre, sub in [("link", cajeros[cajeros["red"] == "LINK"]), ("banelco", cajeros[cajeros["red"] == "BANELCO"])]:
        t[f"dist_{nombre}_m_red"] = red.distancia_a_mas_cercano(t["lon"], t["lat"], sub["lon"], sub["lat"])
    t["dist_cajero_m_red"] = np.minimum(t["dist_link_m_red"], t["dist_banelco_m_red"])
    return t


def conjuntos_red(tabla: pd.DataFrame, red: RedPeatonal, radio_m: float = 500) -> dict:
    """Misma estructura que modelos.conjuntos_recta, pero con distancias caminando."""
    from .modelos import es_candidato

    cand = np.flatnonzero(es_candidato(tabla))
    vecinos = red.vecinos_en_radio(tabla["lon"].to_numpy()[cand], tabla["lat"].to_numpy()[cand],
                                   tabla["lon"], tabla["lat"], radio_m)
    return {"candidatos": cand, "vecinos": vecinos, "cubierto": tabla["dist_link_m_red"].to_numpy() <= radio_m,
            "metodo": "caminando", "radio_m": radio_m}
