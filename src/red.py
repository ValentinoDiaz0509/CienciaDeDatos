"""
red.py
------
Distancias CAMINANDO por la red de calles de OpenStreetMap (bajada con OSMnx en
notebooks/00b_red_peatonal.ipynb), en lugar de en línea recta.

Cada punto (centro de hexágono, cajero, sitio candidato) se proyecta sobre la cuadra más
cercana de la red, y esa cuadra se parte en el punto proyectado. La distancia total es:
tramo perpendicular hasta la cuadra + camino más corto por la red (algoritmo de Dijkstra)
+ tramo perpendicular desde la cuadra.

Proyectar sobre la cuadra, y no sobre la esquina más cercana, evita sumar hasta media
cuadra de más en cada punta: un cajero en la mitad de una cuadra está a 50 m de cada
esquina, pero alguien que camina por esa misma vereda llega directo. Es el mismo criterio
que usa OSMnx con `nearest_edges` (Boeing, 2017).

Aproximación: el tramo de cada cuadra se toma como el segmento recto entre sus dos
esquinas para ubicar la proyección; el largo real (con curvas) viene de OSM en `largo_m`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from .features import CRS_METRICO, DIR_BASE

ARCHIVO_NODOS = "red-peatonal-nodos.parquet"
ARCHIVO_ARISTAS = "red-peatonal-aristas.parquet"
_EPS = 1e-6  # peso mínimo: Dijkstra de SciPy trata los ceros como "sin arista"


def hay_red(base: Path = DIR_BASE) -> bool:
    return (base / ARCHIVO_NODOS).exists() and (base / ARCHIVO_ARISTAS).exists()


def _metros(lon, lat) -> np.ndarray:
    import geopandas as gpd

    p = gpd.GeoSeries(gpd.points_from_xy(np.asarray(lon, float), np.asarray(lat, float)), crs="EPSG:4326").to_crs(CRS_METRICO)
    return np.column_stack([p.x, p.y])


class RedPeatonal:
    def __init__(self, nodos: pd.DataFrame, aristas: pd.DataFrame):
        self.n = len(nodos)
        indice = pd.Series(np.arange(self.n), index=nodos["osmid"].to_numpy())
        u = indice.reindex(aristas["u"].to_numpy()).to_numpy()
        v = indice.reindex(aristas["v"].to_numpy()).to_numpy()
        largo = aristas["largo_m"].to_numpy(float)
        ok = ~(np.isnan(u) | np.isnan(v)) & (u != v)  # sin autolazos
        self.u, self.v, self.largo = u[ok].astype(int), v[ok].astype(int), np.maximum(largo[ok], _EPS)
        self.xy = _metros(nodos["lon"], nodos["lat"])
        self.largo_total_km = float(self.largo.sum() / 1000)
        self._arbol = None

    @classmethod
    def desde_archivos(cls, base: Path = DIR_BASE) -> "RedPeatonal":
        return cls(pd.read_parquet(base / ARCHIVO_NODOS), pd.read_parquet(base / ARCHIVO_ARISTAS))

    # ------------------------------------------------------------------ proyección
    def _arbol_cuadras(self):
        if self._arbol is None:
            import shapely

            a, b = self.xy[self.u], self.xy[self.v]
            self._arbol = shapely.STRtree(shapely.linestrings(np.stack([a, b], axis=1)))
        return self._arbol

    def proyectar(self, lon, lat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Para cada punto: cuadra más cercana, posición sobre ella (0 = esquina u, 1 = esquina v) y distancia a ella."""
        import shapely

        p = _metros(lon, lat)
        _, cuadra = self._arbol_cuadras().query_nearest(shapely.points(p), all_matches=False)
        a, b = self.xy[self.u[cuadra]], self.xy[self.v[cuadra]]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), _EPS), 0, 1)
        h = np.linalg.norm(p - (a + t[:, None] * ab), axis=1)
        return cuadra, t, h

    def con_puntos(self, *grupos) -> tuple[csr_matrix, list[tuple[np.ndarray, np.ndarray]]]:
        """
        Grafo con los puntos insertados: cada cuadra que tiene puntos se parte en tramos
        u → p1 → p2 → … → v. Recibe pares (lon, lat) y devuelve, por grupo, el nodo de cada
        punto en el grafo nuevo y su tramo perpendicular hasta la cuadra.
        """
        tamanos = [len(np.asarray(lon)) for lon, _ in grupos]
        lon = np.concatenate([np.asarray(g[0], float) for g in grupos])
        lat = np.concatenate([np.asarray(g[1], float) for g in grupos])
        cuadra, t, h = self.proyectar(lon, lat)
        k = len(lon)
        ids = self.n + np.arange(k)

        orden = np.lexsort((t, cuadra))
        c_o, t_o, id_o = cuadra[orden], t[orden], ids[orden]
        primero = np.r_[True, c_o[1:] != c_o[:-1]]
        ultimo = np.r_[c_o[1:] != c_o[:-1], True]
        nodo_previo = np.where(primero, self.u[c_o], np.r_[-1, id_o[:-1]])
        t_previo = np.where(primero, 0.0, np.r_[0.0, t_o[:-1]])
        filas = [nodo_previo, id_o[ultimo]]
        cols = [id_o, self.v[c_o[ultimo]]]
        pesos = [(t_o - t_previo) * self.largo[c_o], (1 - t_o[ultimo]) * self.largo[c_o[ultimo]]]

        intactas = np.ones(len(self.u), bool)
        intactas[cuadra] = False
        filas.append(self.u[intactas])
        cols.append(self.v[intactas])
        pesos.append(self.largo[intactas])

        f, c, w = np.concatenate(filas), np.concatenate(cols), np.maximum(np.concatenate(pesos), _EPS)
        total = self.n + k
        grafo = coo_matrix((np.r_[w, w], (np.r_[f, c], np.r_[c, f])), shape=(total, total)).tocsr()

        salida, inicio = [], 0
        for tam in tamanos:
            salida.append((ids[inicio:inicio + tam], h[inicio:inicio + tam]))
            inicio += tam
        return grafo, salida

    # ------------------------------------------------------------------ consultas
    def distancia_a_mas_cercano(self, destinos_lon, destinos_lat, fuentes_lon, fuentes_lat) -> np.ndarray:
        """
        Para cada destino (p. ej. centro de hexágono), distancia caminando al punto fuente más
        cercano (p. ej. cajero Link). Un nodo virtual unido a todas las fuentes, con peso igual a
        su tramo perpendicular, permite resolver todas las fuentes con un único Dijkstra.
        """
        if len(np.asarray(fuentes_lon)) == 0:
            return np.full(len(np.asarray(destinos_lon)), np.inf)
        g, [(nd, hd), (nf, hf)] = self.con_puntos((destinos_lon, destinos_lat), (fuentes_lon, fuentes_lat))
        total = g.shape[0]
        virtual = total
        g = g.tocoo()
        g = coo_matrix((np.r_[g.data, hf + _EPS], (np.r_[g.row, np.full(len(nf), virtual)], np.r_[g.col, nf])),
                       shape=(total + 1, total + 1)).tocsr()
        dist = dijkstra(g, directed=True, indices=virtual)
        return dist[nd] + hd

    def vecinos_en_radio(self, origenes_lon, origenes_lat, destinos_lon, destinos_lat, radio_m: float,
                         lote: int = 150) -> list[np.ndarray]:
        """
        Para cada origen (sitio candidato), índices de los destinos (hexágonos) a menos de
        `radio_m` caminando. Dijkstra acotado al radio, en lotes para no usar demasiada memoria.
        """
        g, [(no, ho), (nd, hd)] = self.con_puntos((origenes_lon, origenes_lat), (destinos_lon, destinos_lat))
        salida = []
        for a in range(0, len(no), lote):
            b = min(a + lote, len(no))
            dist = dijkstra(g, directed=False, indices=no[a:b], limit=radio_m)
            total = dist[:, nd] + hd[None, :] + ho[a:b, None]
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
