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

Dos opciones:
- `tolerancia_m`: además de la cuadra más cercana, conecta cada punto con las cuadras que
  están a no más de `tolerancia_m` metros extra (hasta 4). Representa, por ejemplo, una
  esquina, desde donde se sale a cualquiera de las dos calles. Se usa como análisis de
  sensibilidad; por defecto es 0 (solo la cuadra más cercana).
- `sobre_la_calle`: para los sitios candidatos, el cajero nuevo se ubica sobre la vereda
  (en el punto proyectado), no en el centro de la manzana.

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
_MAX_CUADRAS = 4


def hay_red(base: Path = DIR_BASE) -> bool:
    return (base / ARCHIVO_NODOS).exists() and (base / ARCHIVO_ARISTAS).exists()


def _metros(lon, lat) -> np.ndarray:
    import geopandas as gpd

    p = gpd.GeoSeries(gpd.points_from_xy(np.asarray(lon, float), np.asarray(lat, float)), crs="EPSG:4326").to_crs(CRS_METRICO)
    return np.column_stack([p.x, p.y])


class RedPeatonal:
    def __init__(self, nodos: pd.DataFrame, aristas: pd.DataFrame, tolerancia_m: float = 0.0):
        self.n = len(nodos)
        indice = pd.Series(np.arange(self.n), index=nodos["osmid"].to_numpy())
        u = indice.reindex(aristas["u"].to_numpy()).to_numpy()
        v = indice.reindex(aristas["v"].to_numpy()).to_numpy()
        largo = aristas["largo_m"].to_numpy(float)
        ok = ~(np.isnan(u) | np.isnan(v)) & (u != v)  # sin autolazos
        self.u, self.v, self.largo = u[ok].astype(int), v[ok].astype(int), np.maximum(largo[ok], _EPS)
        self.xy = _metros(nodos["lon"], nodos["lat"])
        self.largo_total_km = float(self.largo.sum() / 1000)
        self.tolerancia_m = float(tolerancia_m)
        self._arbol = None

    @classmethod
    def desde_archivos(cls, base: Path = DIR_BASE, tolerancia_m: float = 0.0) -> "RedPeatonal":
        return cls(pd.read_parquet(base / ARCHIVO_NODOS), pd.read_parquet(base / ARCHIVO_ARISTAS), tolerancia_m)

    # ------------------------------------------------------------------ proyección
    def _arbol_cuadras(self):
        if self._arbol is None:
            import shapely

            a, b = self.xy[self.u], self.xy[self.v]
            self._arbol = shapely.STRtree(shapely.linestrings(np.stack([a, b], axis=1)))
        return self._arbol

    def _sobre(self, p: np.ndarray, cuadra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posición sobre la cuadra (0 = esquina u, 1 = esquina v) y distancia perpendicular."""
        a, b = self.xy[self.u[cuadra]], self.xy[self.v[cuadra]]
        ab = b - a
        t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), _EPS), 0, 1)
        return t, np.linalg.norm(p - (a + t[:, None] * ab), axis=1)

    def proyectar(self, lon, lat, tolerancia_m: float | None = None):
        """
        Cuadras a las que se conecta cada punto. Devuelve (punto, cuadra, t, h): una fila por
        conexión, con la posición t sobre la cuadra y la distancia perpendicular h.
        """
        import shapely

        tol = self.tolerancia_m if tolerancia_m is None else tolerancia_m
        p = _metros(lon, lat)
        geoms = shapely.points(p)
        _, c0 = self._arbol_cuadras().query_nearest(geoms, all_matches=False)
        t0, h0 = self._sobre(p, c0)
        if tol <= 0:
            return np.arange(len(p)), c0, t0, h0
        i, c = self._arbol_cuadras().query(geoms, predicate="dwithin", distance=h0 + tol + 1e-3)
        t, h = self._sobre(p[i], c)
        tabla = (pd.DataFrame({"i": i, "c": c, "t": t, "h": h})
                 .loc[lambda d: d["h"] <= h0[d["i"]] + tol + 1e-3]
                 .sort_values(["i", "h"]).groupby("i").head(_MAX_CUADRAS))
        return tabla["i"].to_numpy(), tabla["c"].to_numpy(), tabla["t"].to_numpy(), tabla["h"].to_numpy()

    def con_puntos(self, *grupos, sobre_la_calle: tuple[bool, ...] | None = None) -> tuple[csr_matrix, list[np.ndarray]]:
        """
        Grafo con los puntos insertados. Cada cuadra que tiene proyecciones se parte en
        tramos u → p1 → p2 → … → v, y cada punto queda como un nodo propio unido a sus
        proyecciones por su tramo perpendicular. Recibe pares (lon, lat) y devuelve el grafo
        y, por grupo, el nodo de cada punto. Con `sobre_la_calle[k]` verdadero, los puntos del
        grupo k se ubican en la vereda (tramo perpendicular nulo, solo la cuadra más cercana).
        """
        sobre_la_calle = sobre_la_calle or (False,) * len(grupos)
        filas_p, filas_c, filas_t, filas_h, tamanos, inicio = [], [], [], [], [], 0
        for (lon, lat), en_calle in zip(grupos, sobre_la_calle):
            i, c, t, h = self.proyectar(lon, lat, tolerancia_m=0 if en_calle else None)
            filas_p.append(i + inicio), filas_c.append(c), filas_t.append(t)
            filas_h.append(np.zeros_like(h) if en_calle else h)
            tamanos.append(len(np.asarray(lon)))
            inicio += tamanos[-1]
        pun, cua, t, h = (np.concatenate(x) for x in (filas_p, filas_c, filas_t, filas_h))
        id_punto = self.n + np.arange(inicio)
        id_proy = self.n + inicio + np.arange(len(pun))

        orden = np.lexsort((t, cua))
        c_o, t_o, id_o = cua[orden], t[orden], id_proy[orden]
        primero = np.r_[True, c_o[1:] != c_o[:-1]]
        ultimo = np.r_[c_o[1:] != c_o[:-1], True]
        intactas = np.ones(len(self.u), bool)
        intactas[cua] = False
        f = [np.where(primero, self.u[c_o], np.r_[-1, id_o[:-1]]), id_o[ultimo], self.u[intactas], id_punto[pun]]
        c = [id_o, self.v[c_o[ultimo]], self.v[intactas], id_proy]
        w = [(t_o - np.where(primero, 0.0, np.r_[0.0, t_o[:-1]])) * self.largo[c_o],
             (1 - t_o[ultimo]) * self.largo[c_o[ultimo]], self.largo[intactas], h]
        f, c, w = np.concatenate(f), np.concatenate(c), np.maximum(np.concatenate(w), _EPS)
        total = self.n + inicio + len(pun)
        grafo = coo_matrix((np.r_[w, w], (np.r_[f, c], np.r_[c, f])), shape=(total, total)).tocsr()
        salida = np.split(id_punto, np.cumsum(tamanos)[:-1])
        return grafo, salida

    # ------------------------------------------------------------------ consultas
    def distancia_a_mas_cercano(self, destinos_lon, destinos_lat, fuentes_lon, fuentes_lat) -> np.ndarray:
        """
        Para cada destino (p. ej. centro de hexágono), distancia caminando al punto fuente más
        cercano (p. ej. cajero Link). Un nodo virtual unido a todas las fuentes permite
        resolver todas las fuentes con un único Dijkstra.
        """
        if len(np.asarray(fuentes_lon)) == 0:
            return np.full(len(np.asarray(destinos_lon)), np.inf)
        g, (nd, nf) = self.con_puntos((destinos_lon, destinos_lat), (fuentes_lon, fuentes_lat))
        total = g.shape[0]
        g = g.tocoo()
        g = coo_matrix((np.r_[g.data, np.full(len(nf), _EPS)], (np.r_[g.row, np.full(len(nf), total)], np.r_[g.col, nf])),
                       shape=(total + 1, total + 1)).tocsr()
        return dijkstra(g, directed=True, indices=total)[nd]

    def vecinos_en_radio(self, origenes_lon, origenes_lat, destinos_lon, destinos_lat, radio_m: float,
                         lote: int = 150) -> list[np.ndarray]:
        """
        Para cada origen (sitio candidato, con el cajero sobre la vereda), índices de los
        destinos (hexágonos) a menos de `radio_m` caminando. Dijkstra acotado al radio, en
        lotes para no usar demasiada memoria.
        """
        g, (no, nd) = self.con_puntos((origenes_lon, origenes_lat), (destinos_lon, destinos_lat),
                                      sobre_la_calle=(True, False))
        salida = []
        for a in range(0, len(no), lote):
            dist = dijkstra(g, directed=False, indices=no[a:a + lote], limit=radio_m)
            salida.extend(np.flatnonzero(fila <= radio_m) for fila in dist[:, nd])
        return salida

    def camino(self, origen_lon: float, origen_lat: float, fuentes_lon, fuentes_lat) -> tuple[np.ndarray, float, int]:
        """Camino a pie (coordenadas en metros) desde un punto hasta la fuente más cercana: (camino, largo, fuente)."""
        g, (nd, nf) = self.con_puntos(([origen_lon], [origen_lat]), (fuentes_lon, fuentes_lat))
        total = g.shape[0]
        g = g.tocoo()
        g = coo_matrix((np.r_[g.data, np.full(len(nf), _EPS)], (np.r_[g.row, np.full(len(nf), total)], np.r_[g.col, nf])),
                       shape=(total + 1, total + 1)).tocsr()
        dist, pred = dijkstra(g, directed=True, indices=total, return_predecessors=True)
        # coordenadas de los nodos nuevos: puntos y sus proyecciones, en el mismo orden que con_puntos
        lon = np.r_[[origen_lon], np.asarray(fuentes_lon, float)]
        lat = np.r_[[origen_lat], np.asarray(fuentes_lat, float)]
        p = _metros(lon, lat)
        pun, cua, t, _ = self.proyectar(lon[:1], lat[:1])
        pun2, cua2, t2, _ = self.proyectar(lon[1:], lat[1:])
        cua, t = np.r_[cua, cua2], np.r_[t, t2]
        a, b = self.xy[self.u[cua]], self.xy[self.v[cua]]
        xy = np.vstack([self.xy, p, a + t[:, None] * (b - a)])
        nodos, nodo = [], int(nd[0])
        while nodo != total and nodo >= 0:
            nodos.append(nodo)
            nodo = int(pred[nodo])
        fuente = int(np.flatnonzero(nf == nodos[-1])[0])
        return xy[nodos[::-1]], float(dist[nd[0]]), fuente


def agregar_distancias_red(tabla: pd.DataFrame, cajeros: pd.DataFrame, red: RedPeatonal, sufijo: str = "_red") -> pd.DataFrame:
    """Suma a la tabla las distancias caminando al cajero Link, Banelco y de cualquier red más cercano."""
    t = tabla.copy()
    for nombre, sub in [("link", cajeros[cajeros["red"] == "LINK"]), ("banelco", cajeros[cajeros["red"] == "BANELCO"])]:
        t[f"dist_{nombre}_m{sufijo}"] = red.distancia_a_mas_cercano(t["lon"], t["lat"], sub["lon"], sub["lat"])
    t[f"dist_cajero_m{sufijo}"] = np.minimum(t[f"dist_link_m{sufijo}"], t[f"dist_banelco_m{sufijo}"])
    return t


def conjuntos_red(tabla: pd.DataFrame, red: RedPeatonal, radio_m: float = 500) -> dict:
    """Misma estructura que modelos.conjuntos_recta, pero con distancias caminando."""
    from .modelos import es_candidato

    cand = np.flatnonzero(es_candidato(tabla))
    vecinos = red.vecinos_en_radio(tabla["lon"].to_numpy()[cand], tabla["lat"].to_numpy()[cand],
                                   tabla["lon"], tabla["lat"], radio_m)
    return {"candidatos": cand, "vecinos": vecinos, "cubierto": tabla["dist_link_m_red"].to_numpy() <= radio_m,
            "metodo": "caminando", "radio_m": radio_m}
