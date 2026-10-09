"""
descarga.py
-----------
Baja todas las fuentes de datos del TPO y las deja en data/raw/.

Fuentes:
  * BA Data (GCBA): cajeros automáticos, barrios, usos del suelo 2022-2024,
    molinetes del subte 2025, bocas de subte.
  * Censo 2022 (INDEC) por radio censal, vía el paquete `censoargentino`
    (los datos están publicados en Hugging Face).
  * OpenStreetMap (Overpass API): estaciones, sucursales bancarias, cajeros
    de todas las redes y comercios.

BA Data a veces responde con errores de servidor, así que cada archivo se
intenta con varias URLs y varios reintentos. Si una fuente falla, el resto
sigue igual y al final se muestra un resumen.

Uso desde la terminal (parado en la raíz del repo):
    python -m src.descarga                 # todo
    python -m src.descarga --sin-pesados   # sin molinetes ni usos del suelo
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
DIR_RAW = RAIZ / "data" / "raw"
ZIP_SALIDA = RAIZ / "data" / "datos_tpo.zip"

BA = "https://data.buenosaires.gob.ar/dataset"
CDN = "https://cdn.buenosaires.gob.ar/datosabiertos/datasets"

# Resultado de cada paso: clave -> (ok, detalle)
ESTADO: dict[str, tuple[bool, str]] = {}


# --------------------------------------------------------------------------
# Fuentes de BA Data
# --------------------------------------------------------------------------
@dataclass
class Fuente:
    clave: str
    descripcion: str
    archivo: str
    urls: list[str]
    pesada: bool = False


FUENTES_BA: list[Fuente] = [
    Fuente(
        "cajeros",
        "Cajeros automáticos (ubicación, banco, red, terminales)",
        "cajeros-automaticos.csv",
        [
            f"{CDN}/secretaria-de-desarrollo-urbano/cajeros-automaticos/cajeros-automaticos.csv",
            f"{BA}/cajeros-automaticos/resource/juqdkmgo-281-resource/download",
        ],
    ),
    Fuente(
        "barrios",
        "Barrios de CABA (polígonos)",
        "barrios.geojson",
        [
            f"{CDN}/innovacion-transformacion-digital/barrios/barrios.geojson",
            f"{BA}/barrios/resource/1c3d185b-fdc9-474b-b41b-9bd960a3806e/download",
        ],
    ),
    Fuente(
        "bocas_subte",
        "Bocas de subte",
        "bocas-subte.geojson",
        [f"{CDN}/transporte-y-obras-publicas/bocas-subte/bocas-subte.geojson"],
    ),
    Fuente(
        "usos_suelo_doc",
        "Documentación del relevamiento de usos del suelo 2022-2024",
        "usos-suelo-2022-2024-documentacion.pdf",
        [
            f"{CDN}/secretaria-de-desarrollo-urbano/relevamiento-usos-suelo/"
            "documentacion-relevamiento-usos-suelo-2022-2024.pdf",
            f"{BA}/relevamiento-usos-suelo/resource/801aecc9-d361-4f83-8f61-08c9b76f987d/download",
        ],
    ),
    Fuente(
        "usos_suelo",
        "Relevamiento de usos del suelo 2022-2024 (una fila por parcela)",
        "usos-suelo-2022-2024.csv",
        [f"{BA}/relevamiento-usos-suelo/resource/3c7e5f10-577a-44ea-b614-82cc05f842aa/download"],
        pesada=True,
    ),
    Fuente(
        "usos_suelo_shp",
        "Relevamiento de usos del suelo 2022-2024 (shapefile, para tener coordenadas)",
        "usos-suelo-2022-2024-shp.zip",
        [f"{BA}/relevamiento-usos-suelo/resource/613ef164-131b-422a-b9e2-257cee4b46b4/download"],
        pesada=True,
    ),
    Fuente(
        "molinetes_2025",
        "Subte: pasajeros por molinete cada 15 minutos (2025)",
        "molinetes-2025.zip",
        [f"{BA}/subte-viajes-molinetes/resource/0d689701-9efd-4644-85d0-f6a1504509f1/download"],
        pesada=True,
    ),
    Fuente(
        "molinetes_2026",
        "Subte: pasajeros por molinete cada 15 minutos (2026, primer semestre)",
        "molinetes-2026.zip",
        [f"{BA}/subte-viajes-molinetes/resource/f4689712-8698-43ab-9e02-559d03245cb0/download"],
        pesada=True,
    ),
]

_SESION = requests.Session()
_SESION.headers["User-Agent"] = "Mozilla/5.0 (TPO Ciencia de Datos UADE)"


def _validar(ruta: Path, extension: str) -> None:
    """Rechaza descargas vacías o páginas de error HTML disfrazadas de archivo."""
    tam = ruta.stat().st_size
    if tam == 0:
        raise ValueError("el archivo llegó vacío")
    with open(ruta, "rb") as f:
        inicio = f.read(512).lstrip().lower()
    if inicio.startswith((b"<!doctype html", b"<html")):
        raise ValueError("el servidor devolvió una página HTML en vez del archivo")
    if extension == ".zip" and not zipfile.is_zipfile(ruta):
        raise ValueError("el .zip está corrupto o incompleto")
    if extension == ".pdf" and not inicio.startswith(b"%pdf"):
        raise ValueError("no es un PDF válido")
    if extension in (".geojson", ".json") and not inicio.startswith(b"{"):
        raise ValueError("no es un JSON válido")
    if extension == ".parquet" and not inicio.startswith(b"par1"):
        raise ValueError("no es un Parquet válido")


def descargar(urls: list[str], destino: Path, intentos: int = 4, timeout: int = 300) -> tuple[bool, str]:
    """Prueba cada URL con reintentos y backoff. Devuelve (ok, url usada o error)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        return True, "ya estaba descargado"

    ultimo_error = "sin URLs"
    for url in urls:
        for n in range(1, intentos + 1):
            tmp = destino.with_name(destino.name + ".part")
            try:
                with _SESION.get(url, stream=True, timeout=timeout, allow_redirects=True) as r:
                    if r.status_code in (403, 404, 410):
                        raise FileNotFoundError(f"HTTP {r.status_code}")
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for bloque in r.iter_content(chunk_size=1 << 20):
                            f.write(bloque)
                _validar(tmp, destino.suffix.lower())
                tmp.replace(destino)
                return True, url
            except FileNotFoundError as e:
                ultimo_error = f"{url} -> {e}"
                print(f"   {e} en {url}; pruebo la siguiente URL")
                break  # no tiene sentido reintentar un 404
            except Exception as e:  # errores de servidor, timeouts, HTML de error
                ultimo_error = f"{type(e).__name__}: {e}"
                if n < intentos:
                    espera = min(60, 5 * 2 ** (n - 1))
                    print(f"   intento {n}/{intentos} falló ({ultimo_error[:100]}); reintento en {espera}s")
                    time.sleep(espera)
            finally:
                if tmp.exists():
                    tmp.unlink()
    return False, ultimo_error


def descargar_fuentes_ba(incluir_pesadas: bool = True) -> None:
    for fuente in FUENTES_BA:
        if fuente.pesada and not incluir_pesadas:
            continue
        print(f"→ {fuente.descripcion}")
        ok, detalle = descargar(fuente.urls, DIR_RAW / fuente.archivo)
        ESTADO[fuente.clave] = (ok, detalle)
        print(f"   {'OK' if ok else 'FALLÓ'}: {detalle}")


# --------------------------------------------------------------------------
# Censo 2022 por radio censal
# --------------------------------------------------------------------------
VARIABLES_CENSO = [
    "PERSONA_P02",      # sexo -> sumando categorías da la población total
    "PERSONA_EDADGRU",  # grandes grupos de edad
    "PERSONA_CONDACT",  # condición de actividad (ocupado, desocupado, inactivo)
    "PERSONA_MNI",      # máximo nivel de instrucción
    "HOGAR_NBI_TOT",    # necesidades básicas insatisfechas
    "HOGAR_IPMH",       # índice de privación material del hogar
    "HOGAR_H24A",       # acceso a internet / celular / computadora
    "HOGAR_H24B",
    "HOGAR_H24C",
]


# Polígonos de todos los radios del país (~58 MB). Lo bajamos entero y filtramos CABA
# nosotros, porque el `geometry=True` de censoargentino falla con la versión actual del archivo.
RADIOS_URL = "https://huggingface.co/datasets/pedroorden/censoargentino/resolve/main/radios-2022.parquet"
# Copia del mismo archivo publicada por el proyecto original (ciut-redatam), por si falla Hugging Face
RADIOS_URL_S3 = "https://arg-fulbright-data.s3.us-east-2.amazonaws.com/censo-argentino-2022/radios-2022.parquet"


def descargar_censo() -> None:
    """Variables del Censo 2022 para CABA (provincia 02) + polígonos de radios con población."""
    out_largo = DIR_RAW / "censo2022-caba-largo.parquet"
    out_radios = DIR_RAW / "censo2022-caba-radios.geojson"

    if not out_largo.exists():
        print("→ Censo 2022: variables por radio censal (CABA)")
        try:
            import censoargentino as censo

            largo = censo.query(variables=VARIABLES_CENSO, provincia="02")
            largo.to_parquet(out_largo, index=False)
            faltan = sorted(set(VARIABLES_CENSO) - set(largo["codigo_variable"].unique()))
            if faltan:
                print(f"   aviso: no vinieron datos para {faltan}")
        except Exception as e:
            ESTADO["censo"] = (False, f"{type(e).__name__}: {e}")
    if out_largo.exists():
        n = pd.read_parquet(out_largo, columns=["id_geo"])["id_geo"].nunique()
        ESTADO["censo"] = (True, f"{n:,} radios censales")
    print(f"   {'OK' if ESTADO['censo'][0] else 'FALLÓ'}: {ESTADO['censo'][1]}")

    if not out_radios.exists():
        print("→ Censo 2022: polígonos de radios censales (CABA)")
        try:
            ok, det = descargar([RADIOS_URL, RADIOS_URL_S3], DIR_RAW / "radios-2022-pais.parquet")
            if not ok:
                raise RuntimeError(f"no se pudo bajar radios-2022.parquet: {det}")
            radios = radios_caba(DIR_RAW / "radios-2022-pais.parquet")
            if out_largo.exists():
                radios = radios.merge(poblacion_por_radio(out_largo), on="id_geo", how="left")
            radios.to_file(out_radios, driver="GeoJSON")
        except Exception as e:
            ESTADO["censo_radios"] = (False, f"{type(e).__name__}: {e}")
    if out_radios.exists():
        import geopandas as gpd

        r = gpd.read_file(out_radios)
        con_pob = int(r["poblacion"].notna().sum()) if "poblacion" in r else 0
        ESTADO["censo_radios"] = (True, f"{len(r):,} polígonos, {con_pob:,} con población")
    print(f"   {'OK' if ESTADO['censo_radios'][0] else 'FALLÓ'}: {ESTADO['censo_radios'][1]}")


_ID_RADIO_CANDIDATOS = ["cod_2022", "COD_2022", "link", "LINK", "id_geo", "cod_radio", "COD_RADIO",
                        "geocodigo", "GEOCODIGO", "codigo", "CODIGO"]


def _columna_id_radio(df) -> str:
    for c in _ID_RADIO_CANDIDATOS:
        if c in df.columns:
            return c
    for c in df.columns:  # si cambió el nombre: la columna cuyos valores son códigos de 8-9 dígitos
        if c == "geometry":
            continue
        muestra = df[c].dropna().astype(str).str.replace(r"\.0$", "", regex=True).head(200)
        if len(muestra) and muestra.str.fullmatch(r"\d{8,9}").all():
            return c
    raise ValueError(f"no encuentro la columna con el código de radio; columnas: {list(df.columns)}")


def radios_caba(parquet_path: Path):
    """Polígonos de los radios de CABA (código que empieza con 02), en EPSG:4326."""
    import geopandas as gpd

    try:
        g = gpd.read_parquet(parquet_path)  # GeoParquet
    except Exception:
        df = pd.read_parquet(parquet_path)
        col_geo = next((c for c in ["geometry", "geom", "wkb_geometry", "GEOMETRY"] if c in df.columns), None)
        if col_geo is None:
            raise ValueError(f"no encuentro la geometría; columnas: {list(df.columns)}")
        geom = gpd.GeoSeries.from_wkb(df[col_geo].map(lambda b: bytes(b) if b is not None else None))
        g = gpd.GeoDataFrame(df.drop(columns=[col_geo]), geometry=geom, crs=None)

    col_id = _columna_id_radio(g)
    g["id_geo"] = g[col_id].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(9)
    g = g[g["id_geo"].str.startswith("02")].copy()
    if g.empty:
        raise ValueError(f"no hay radios de CABA usando la columna '{col_id}'; columnas: {list(g.columns)}")
    if g.crs is None:
        minx = g.total_bounds[0]
        if -75 < minx < -50:
            g = g.set_crs("EPSG:4326")
        else:
            raise ValueError(f"no sé en qué sistema de coordenadas están los radios (bounds {g.total_bounds})")
    return g.to_crs("EPSG:4326")[["id_geo", "geometry"]].reset_index(drop=True)


def poblacion_por_radio(largo_path: Path) -> pd.DataFrame:
    """Población total por radio = suma de las categorías de sexo (PERSONA_P02)."""
    l = pd.read_parquet(largo_path, columns=["id_geo", "codigo_variable", "conteo"])
    l = l[l["codigo_variable"] == "PERSONA_P02"]
    return l.groupby("id_geo", as_index=False)["conteo"].sum().rename(columns={"conteo": "poblacion"})


# --------------------------------------------------------------------------
# OpenStreetMap
# --------------------------------------------------------------------------
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

_FILTROS_OSM = """
  nwr["amenity"~"^(atm|bank)$"]{zona};
  nwr["railway"~"^(station|halt|subway_entrance)$"]{zona};
  nwr["station"="subway"]{zona};
  nwr["shop"]{zona};
  nwr["amenity"~"^(restaurant|cafe|fast_food|bar|pharmacy|fuel|marketplace|hospital|clinic|university|college|school)$"]{zona};
"""

# Área de CABA por su código ISO; si no la encuentra, se usa un rectángulo
# que la contiene (después se recorta con los polígonos de barrios).
CONSULTA_OSM_AREA = (
    '[out:json][timeout:300];\narea["ISO3166-2"="AR-C"]->.caba;\n(\n'
    + _FILTROS_OSM.format(zona="(area.caba)")
    + ");\nout center tags;"
)
BBOX_CABA = "(-34.71,-58.54,-34.52,-58.33)"
CONSULTA_OSM_BBOX = (
    "[out:json][timeout:300];\n(\n" + _FILTROS_OSM.format(zona=BBOX_CABA) + ");\nout center tags;"
)

_GASTRONOMIA = {"restaurant", "cafe", "fast_food", "bar"}


def categoria_osm(t: dict) -> str:
    amenity, railway, station = t.get("amenity"), t.get("railway"), t.get("station")
    if amenity == "atm":
        return "cajero"
    if amenity == "bank":
        return "banco"
    if railway == "subway_entrance":
        return "boca_subte"
    if station == "subway":
        return "estacion_subte"
    if station == "light_rail":
        return "estacion_premetro"
    if railway in ("station", "halt"):
        return "estacion_tren"
    if "shop" in t or amenity == "marketplace":
        return "comercio"
    if amenity in _GASTRONOMIA:
        return "gastronomia"
    if amenity == "pharmacy":
        return "farmacia"
    if amenity == "fuel":
        return "estacion_servicio"
    if amenity in ("hospital", "clinic"):
        return "salud"
    if amenity in ("university", "college", "school"):
        return "educacion"
    return "otro"


def elementos_osm_a_tabla(elementos: list[dict]) -> pd.DataFrame:
    filas = []
    for el in elementos:
        if el.get("type") == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            centro = el.get("center") or {}
            lat, lon = centro.get("lat"), centro.get("lon")
        t = el.get("tags") or {}
        filas.append({
            "osm_tipo": el.get("type"),
            "osm_id": el.get("id"),
            "lat": lat,
            "lon": lon,
            "categoria": categoria_osm(t),
            "nombre": t.get("name"),
            "operador": t.get("operator"),
            "red": t.get("network"),
            "marca": t.get("brand"),
            "amenity": t.get("amenity"),
            "shop": t.get("shop"),
            "railway": t.get("railway"),
            "station": t.get("station"),
            "linea": t.get("line") or t.get("ref"),
        })
    df = pd.DataFrame(filas)
    return df.dropna(subset=["lat", "lon"]).drop_duplicates(["osm_tipo", "osm_id"])


def descargar_osm() -> None:
    destino = DIR_RAW / "osm-caba-puntos.csv"
    if destino.exists():
        ESTADO["osm"] = (True, "ya estaba descargado")
        print("   OK: ya estaba descargado")
        return
    print("→ OpenStreetMap: estaciones, bancos, cajeros y comercios de CABA")
    for consulta, nombre in ((CONSULTA_OSM_AREA, "área CABA"), (CONSULTA_OSM_BBOX, "rectángulo CABA")):
        for url in OVERPASS_URLS:
            try:
                r = _SESION.post(url, data={"data": consulta}, timeout=400)
                r.raise_for_status()
                elementos = r.json().get("elements", [])
                if not elementos:
                    print(f"   {url}: 0 resultados con {nombre}")
                    break  # probar la otra consulta
                df = elementos_osm_a_tabla(elementos)
                df["consulta"] = nombre
                df.to_csv(destino, index=False)
                resumen = df["categoria"].value_counts().to_dict()
                ESTADO["osm"] = (True, f"{len(df):,} puntos ({nombre}) {resumen}")
                print(f"   OK: {ESTADO['osm'][1]}")
                return
            except Exception as e:
                print(f"   {url} falló ({type(e).__name__}: {str(e)[:100]}); pruebo otro servidor")
                time.sleep(10)
    ESTADO["osm"] = (False, "ningún servidor de Overpass respondió")
    print("   FALLÓ: " + ESTADO["osm"][1])


# --------------------------------------------------------------------------
# Procesamiento de los archivos pesados (para que el zip final sea liviano)
# --------------------------------------------------------------------------
def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", texto) if not unicodedata.combining(c))


def _normalizar_columnas(cols) -> list[str]:
    return [_sin_tildes(str(c)).strip().lower().replace(" ", "_") for c in cols]


def _detectar_formato(muestra: bytes) -> tuple[str, str, bool]:
    """
    Devuelve (encoding, separador, filas_entre_comillas) a partir de los primeros bytes de un CSV.
    `filas_entre_comillas` cubre archivos como los molinetes, donde cada fila entera viene
    entre comillas: "1/1/2025;07:45:00;...;2"
    """
    if muestra.startswith(b"\xef\xbb\xbf"):
        encoding, texto = "utf-8-sig", muestra[3:].decode("utf-8", errors="replace")
    else:
        texto, encoding = None, "latin-1"
        # la muestra puede cortar un carácter multibyte (hasta 3 bytes) justo al final
        candidatas = [muestra[: len(muestra) - k] for k in range(4)] if len(muestra) > 4 else [muestra]
        for cand in candidatas:
            try:
                texto, encoding = cand.decode("utf-8"), "utf-8"
                break
            except UnicodeDecodeError:
                continue
        if texto is None:
            texto = muestra.decode("latin-1")
    primera = texto.splitlines()[0].strip() if texto else ""
    sep = max([";", ",", "\t", "|"], key=primera.count)
    entre_comillas = primera.startswith('"') and primera.endswith('"') and primera.count('"') == 2
    return encoding, sep, entre_comillas


def _leer_csv(fuente, encoding: str, sep: str, entre_comillas: bool, chunksize: int | None = None):
    """read_csv que además limpia las comillas de las filas entrecomilladas."""
    lector = pd.read_csv(fuente, sep=sep, encoding=encoding, dtype=str, on_bad_lines="skip",
                         quoting=csv.QUOTE_NONE if entre_comillas else csv.QUOTE_MINIMAL,
                         chunksize=chunksize)

    def limpiar(df: pd.DataFrame) -> pd.DataFrame:
        if entre_comillas:
            df.columns = [str(c).strip('"') for c in df.columns]
            primera, ultima = df.columns[0], df.columns[-1]
            df[primera] = df[primera].str.lstrip('"')
            df[ultima] = df[ultima].str.rstrip('"')
        return df

    if chunksize is None:
        return limpiar(lector)
    return (limpiar(c) for c in lector)


def _buscar(cols: list[str], *claves: str) -> str | None:
    for clave in claves:
        for c in cols:
            if c == clave:
                return c
    for clave in claves:
        for c in cols:
            if clave in c:
                return c
    return None


def _csvs_en_zip(z: zipfile.ZipFile):
    """Devuelve (nombre, zip que lo contiene) de cada CSV, incluso dentro de zips anidados."""
    encontrados = []
    for nombre in z.namelist():
        if nombre.lower().endswith((".csv", ".txt")):
            encontrados.append((nombre, z))
        elif nombre.lower().endswith(".zip"):
            interno = zipfile.ZipFile(io.BytesIO(z.read(nombre)))
            encontrados.extend(_csvs_en_zip(interno))
    return encontrados


def muestra_de_zip(zip_path: Path, destino: Path, filas: int = 2000) -> None:
    """Guarda las primeras filas del primer CSV del zip, para revisar el formato si algo falla."""
    with zipfile.ZipFile(zip_path) as z:
        csvs = _csvs_en_zip(z)
        if not csvs:
            destino.write_text("contenido del zip:\n" + "\n".join(z.namelist()[:200]), encoding="utf-8")
            return
        nombre, contenedor = csvs[0]
        with contenedor.open(nombre) as f:
            lineas = f.read(5_000_000).splitlines()[:filas]
    destino.write_bytes(b"\n".join(lineas))


def agregar_molinetes(zip_path: Path, destino: Path) -> str:
    """
    Resume los molinetes (millones de filas cada 15 min) a:
    línea x estación x tipo de día (hábil / fin de semana) x hora -> pasajeros promedio por día.
    """
    partes, fechas = [], set()
    with zipfile.ZipFile(zip_path) as z:
        csvs = _csvs_en_zip(z)
        if not csvs:
            raise ValueError(f"el zip no tiene CSV: {z.namelist()[:10]}")
        for nombre, contenedor in csvs:
            with contenedor.open(nombre) as f:
                formato = _detectar_formato(f.read(50_000))
            with contenedor.open(nombre) as f:
                for chunk in _leer_csv(f, *formato, chunksize=500_000):
                    chunk.columns = _normalizar_columnas(chunk.columns)
                    cols = list(chunk.columns)
                    c_fecha = _buscar(cols, "fecha")
                    c_desde = _buscar(cols, "desde", "hora")
                    c_linea = _buscar(cols, "linea")
                    c_est = _buscar(cols, "estacion")
                    c_total = _buscar(cols, "pax_total", "total")
                    if not all([c_fecha, c_desde, c_linea, c_est]):
                        raise ValueError(f"no reconozco las columnas de {nombre}: {cols}")
                    if c_total:
                        total = pd.to_numeric(chunk[c_total], errors="coerce")
                    else:
                        pax = [c for c in cols if c.startswith("pax")]
                        if not pax:
                            raise ValueError(f"no encuentro columnas de pasajeros en {nombre}: {cols}")
                        total = chunk[pax].apply(pd.to_numeric, errors="coerce").sum(axis=1)
                    # "05:00:00" o "2025-01-01 05:00:00" -> 5
                    hora = pd.to_numeric(chunk[c_desde].str.extract(r"(\d{1,2}):\d{2}")[0], errors="coerce")
                    df = pd.DataFrame({
                        "fecha": chunk[c_fecha].str.strip(),
                        "linea": chunk[c_linea].str.strip().str.upper(),
                        "estacion": chunk[c_est].str.strip(),
                        "hora": hora,
                        "pasajeros": total,
                    }).dropna(subset=["hora", "pasajeros"])
                    fechas.update(df["fecha"].unique())
                    partes.append(df.groupby(["fecha", "linea", "estacion", "hora"], as_index=False)["pasajeros"].sum())

    todo = pd.concat(partes).groupby(["fecha", "linea", "estacion", "hora"], as_index=False)["pasajeros"].sum()
    fechas_dt = {f: pd.to_datetime(f, dayfirst=True, errors="coerce") for f in fechas}
    todo["dia"] = todo["fecha"].map(fechas_dt)
    todo = todo.dropna(subset=["dia"])
    todo["tipo_dia"] = todo["dia"].dt.dayofweek.map(lambda d: "fin_de_semana" if d >= 5 else "habil")
    dias = todo.drop_duplicates("dia").groupby("tipo_dia")["dia"].count().rename("dias")
    res = todo.groupby(["linea", "estacion", "tipo_dia", "hora"], as_index=False)["pasajeros"].sum()
    res = res.merge(dias, on="tipo_dia")
    res["pasajeros_promedio_dia"] = (res["pasajeros"] / res["dias"]).round(1)
    res["hora"] = res["hora"].astype(int)
    res.to_csv(destino, index=False)
    return (f"{res['estacion'].nunique()} estaciones, {int(dias.sum())} días "
            f"({todo['dia'].min():%d/%m/%Y} a {todo['dia'].max():%d/%m/%Y})")


def usos_suelo_a_parquet(csv_path: Path, destino: Path) -> str:
    """Mismo contenido que el CSV, en Parquet (pesa varias veces menos)."""
    with open(csv_path, "rb") as f:
        formato = _detectar_formato(f.read(50_000))
    df = _leer_csv(csv_path, *formato)
    df.to_parquet(destino, index=False, compression="zstd")
    return f"{len(df):,} parcelas, columnas: {list(df.columns)}"


# Gauss-Krüger Buenos Aires: el sistema en metros que usa el GCBA en muchos shapefiles
GKBA = ("+proj=tmerc +lat_0=-34.6297166 +lon_0=-58.4627 +k=1 +x_0=100000 +y_0=100000 "
        "+ellps=intl +units=m +no_defs")


def usos_suelo_puntos(zip_path: Path, destino: Path) -> str:
    """
    Del shapefile de usos del suelo (polígonos de parcelas) a una tabla con un punto
    (lon, lat) por fila: el CSV oficial no trae coordenadas.
    """
    import tempfile

    import geopandas as gpd

    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
        shps = sorted(Path(tmp).rglob("*.shp"))
        if not shps:
            raise ValueError(f"el zip no tiene .shp: {z.namelist()[:20]}")
        g = gpd.read_file(max(shps, key=lambda p: p.stat().st_size))

    if g.crs is None:
        minx = g.total_bounds[0]
        if 80_000 < minx < 120_000:
            g = g.set_crs(GKBA)
        elif -59 < minx < -58:
            g = g.set_crs("EPSG:4326")
        else:
            raise ValueError(f"no sé en qué sistema de coordenadas está (bounds {g.total_bounds})")
    puntos = g.to_crs("EPSG:5347").geometry.representative_point().to_crs("EPSG:4326")
    tabla = pd.DataFrame(g.drop(columns=g.geometry.name))
    tabla["lon"], tabla["lat"] = puntos.x.round(6), puntos.y.round(6)
    for c in tabla.columns:  # parquet no acepta columnas mixtas
        if tabla[c].dtype == object:
            tabla[c] = tabla[c].astype("string")
    tabla.to_parquet(destino, index=False, compression="zstd")
    return f"{len(tabla):,} filas con coordenadas (CRS original: {g.crs.name if g.crs else '?'})"


def procesar_pesados() -> None:
    for mol_zip in sorted(DIR_RAW.glob("molinetes-20??.zip")):
        anio = mol_zip.stem.split("-")[-1]
        mol_out = DIR_RAW / f"molinetes-{anio}-estacion-hora.csv"
        if mol_out.exists():
            continue
        clave = f"molinetes_{anio}_resumen"
        print(f"→ Resumiendo molinetes {anio} por estación y hora (tarda unos minutos)")
        try:
            ESTADO[clave] = (True, agregar_molinetes(mol_zip, mol_out))
        except Exception as e:
            ESTADO[clave] = (False, f"{type(e).__name__}: {e}")
            try:  # una muestra para poder adaptar el código al formato real
                muestra_de_zip(mol_zip, DIR_RAW / f"molinetes-{anio}-muestra.csv")
            except Exception:
                pass
        print(f"   {'OK' if ESTADO[clave][0] else 'FALLÓ'}: {ESTADO[clave][1]}")

    shp_zip = DIR_RAW / "usos-suelo-2022-2024-shp.zip"
    shp_out = DIR_RAW / "usos-suelo-2022-2024-puntos.parquet"
    if shp_zip.exists() and not shp_out.exists():
        print("→ Usos del suelo: un punto (lon, lat) por parcela")
        try:
            ESTADO["usos_suelo_puntos"] = (True, usos_suelo_puntos(shp_zip, shp_out))
        except Exception as e:
            ESTADO["usos_suelo_puntos"] = (False, f"{type(e).__name__}: {e}")
        print(f"   {'OK' if ESTADO['usos_suelo_puntos'][0] else 'FALLÓ'}: {ESTADO['usos_suelo_puntos'][1]}")

    uso_csv = DIR_RAW / "usos-suelo-2022-2024.csv"
    uso_out = DIR_RAW / "usos-suelo-2022-2024.parquet"
    if uso_csv.exists() and not uso_out.exists():
        print("→ Pasando usos del suelo a Parquet")
        try:
            ESTADO["usos_suelo_parquet"] = (True, usos_suelo_a_parquet(uso_csv, uso_out))
        except Exception as e:
            ESTADO["usos_suelo_parquet"] = (False, f"{type(e).__name__}: {e}")
            with open(uso_csv, "rb") as f:  # muestra para revisar el formato
                (DIR_RAW / "usos-suelo-2022-2024-muestra.csv").write_bytes(b"\n".join(f.read(2_000_000).splitlines()[:2000]))
        print(f"   {'OK' if ESTADO['usos_suelo_parquet'][0] else 'FALLÓ'}: {ESTADO['usos_suelo_parquet'][1]}")


# --------------------------------------------------------------------------
# Resumen y empaquetado
# --------------------------------------------------------------------------
# Los crudos pesados no van al zip: van sus versiones resumidas.
EXCLUIR_DEL_ZIP = {
    "molinetes-2024.zip", "molinetes-2025.zip", "molinetes-2026.zip",
    "usos-suelo-2022-2024.csv", "usos-suelo-2022-2024-shp.zip",
    "radios-2022-pais.parquet",
}


def resumen() -> pd.DataFrame:
    filas = [{"paso": k, "ok": "sí" if ok else "NO", "detalle": det[:160]} for k, (ok, det) in ESTADO.items()]
    archivos = [
        {"archivo": p.name, "MB": round(p.stat().st_size / 1e6, 2), "va_al_zip": p.name not in EXCLUIR_DEL_ZIP}
        for p in sorted(DIR_RAW.glob("*")) if p.is_file() and not p.name.endswith(".part")
    ]
    print(pd.DataFrame(filas).to_string(index=False) if filas else "(no se ejecutó ningún paso)")
    print()
    print(pd.DataFrame(archivos).to_string(index=False) if archivos else "(todavía no hay archivos en data/raw)")
    return pd.DataFrame(filas)


def empaquetar(destino: Path = ZIP_SALIDA) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destino, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DIR_RAW.glob("*")):
            if p.is_file() and p.name not in EXCLUIR_DEL_ZIP and not p.name.endswith(".part"):
                z.write(p, arcname=p.name)
        z.writestr("estado_descarga.json", json.dumps(
            {k: {"ok": ok, "detalle": det} for k, (ok, det) in ESTADO.items()}, ensure_ascii=False, indent=2))
    mb = destino.stat().st_size / 1e6
    print(f"Zip listo: {destino} ({mb:.1f} MB)")
    return destino


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Descarga las fuentes del TPO a data/raw/")
    parser.add_argument("--sin-pesados", action="store_true", help="no bajar molinetes ni usos del suelo")
    args = parser.parse_args(argv)

    DIR_RAW.mkdir(parents=True, exist_ok=True)
    descargar_fuentes_ba(incluir_pesadas=not args.sin_pesados)
    descargar_censo()
    descargar_osm()
    procesar_pesados()
    print()
    resumen()
    empaquetar()


if __name__ == "__main__":
    main(sys.argv[1:])
