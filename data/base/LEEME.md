# data/base

Versiones livianas de las fuentes, listas para el análisis. Las generan `notebooks/00_descarga_datos.ipynb`,
`notebooks/00b_red_peatonal.ipynb` y `notebooks/00c_serie_diaria_subte.ipynb`, y se versionan en el repo para poder trabajar
y presentar sin depender de internet. `estado_descarga.json` y `estado_serie_subte.json` registran qué se bajó y si hubo
errores.

| Archivo | Qué es | Registros | Fuente |
|---|---|---|---|
| `cajeros-automaticos.csv` | Cajeros con ubicación, banco, red (LINK / BANELCO) y cantidad de terminales | 1.279 ubicaciones, 2.673 terminales | BA Data |
| `barrios.geojson` | Polígonos de los 48 barrios | 48 | BA Data |
| `censo2022-caba-largo.parquet` | Censo 2022 por radio censal, formato largo (radio × variable × categoría × conteo) | 3.820 radios | INDEC, vía `censoargentino` |
| `censo2022-caba-radios.geojson` | Polígonos de los radios censales con su población | 3.820 | INDEC / CONICET |
| `molinetes-2025-estacion-hora.csv` | Entradas promedio por día, por estación, hora y tipo de día (2025) | 101 estaciones, 335 días | BA Data / SBASE |
| `molinetes-2026-estacion-hora.csv` | Ídem, enero a junio de 2026 | 100 estaciones, 181 días | BA Data / SBASE |
| `subte-diario-estacion.parquet` | Entradas por día, línea y estación: la serie diaria para el pronóstico (notebook 05) | 578 días del 01/01/2025 al 31/08/2026 (falta marzo de 2025), 52.503 filas | BA Data / SBASE |
| `subte-hora-red.parquet` | Entradas por día y hora en toda la red | 11.922 filas | BA Data / SBASE |
| `osm-caba-puntos.csv` | Estaciones, bocas de subte, bancos, cajeros, comercios, gastronomía, farmacias, salud, educación | ≈29 mil puntos | © colaboradores de OpenStreetMap (ODbL) |
| `usos-suelo-2022-2024.parquet` | Uso de cada parcela (CSV oficial, sin coordenadas) | 417.764 | BA Data |
| `usos-suelo-2022-2024-puntos.parquet` | Ídem, con un punto (lon, lat) por parcela tomado del shapefile | 417.737 | BA Data |
| `usos-suelo-2022-2024-documentacion.pdf` | Glosario de los campos de usos del suelo | | BA Data |
| `red-peatonal-nodos.parquet`, `red-peatonal-aristas.parquet` | Red de calles y sendas caminables (esquinas y cuadras con su largo en metros), bajada con OSMnx | 52.090 esquinas, 79.657 cuadras (4.817 km) | © colaboradores de OpenStreetMap (ODbL) |

## Cuidado con los cajeros

El listado de cajeros de BA Data dice "actualizado en 2026", pero los bancos que aparecen indican que **el relevamiento es
de alrededor de 2017**: figuran CitiBank (dejó la banca minorista en 2017), "BBVA Banco Francés" y "Banco Santander Río"
(cambiaron de nombre en 2019) y HSBC (hoy Galicia Más). En total, el 31 % de las ubicaciones. Se trata como una foto de
la oferta de ≈2017; la app permite cargar el listado actual.

## Cuidado con los molinetes de 2025

Al archivo de molinetes de 2025 le falta **marzo**, y una parte de los registros del **1 al 12 de agosto** trae la fecha
como mes/día (`08/04/2025` es el 4 de agosto). Leída como día/mes, que es el formato del resto del archivo, cae en el día 8
de otro mes: por eso el 8 de cada mes de 2025 tiene un tercio más de entradas que un día típico y del 1 al 12 de agosto falta
lo mismo (en 2026 no pasa). La suma por día ya mezcla los registros, así que no se pueden separar: el pronóstico deja afuera
esos 22 días (los detecta `fechas_cruzadas`, en `src/pronostico.py`). En los promedios por hora del notebook 04 el efecto es menor al 1 %.
