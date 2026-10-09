# data/base

Versiones livianas de las fuentes, listas para el análisis. Las genera `notebooks/00_descarga_datos.ipynb` y se suben al repo para poder trabajar (y presentar) sin depender de internet.

| Archivo | Qué es | Filas | Fuente |
|---|---|---|---|
| `cajeros-automaticos.csv` | Cajeros con ubicación, banco, red (LINK / BANELCO) y cantidad de terminales | 1.279 ubicaciones, 2.673 terminales | BA Data |
| `barrios.geojson` | Polígonos de los 48 barrios | 48 | BA Data |
| `censo2022-caba-largo.parquet` | Variables del Censo 2022 por radio censal, formato largo (radio × categoría × conteo) | 3.820 radios | INDEC, vía `censoargentino` |
| `molinetes-2026-estacion-hora.csv` | Pasajeros promedio por día, por estación, hora y tipo de día (hábil / fin de semana), enero a junio 2026 | 100 estaciones | BA Data / SBASE |
| `osm-caba-puntos.csv` | Estaciones, bocas de subte, bancos, cajeros, comercios, gastronomía, farmacias, salud, educación | 29.292 puntos | © colaboradores de OpenStreetMap (ODbL) |
| `usos-suelo-2022-2024.parquet` | Uso de cada parcela (residencial, comercial, oficinas…). **Todavía sin coordenadas**: se agregan desde el shapefile | 417.764 | BA Data |
| `usos-suelo-2022-2024-documentacion.pdf` | Glosario de los campos de usos del suelo | | BA Data |

## Cuidado con los cajeros

El listado de cajeros de BA Data dice "actualizado en 2026", pero los bancos que aparecen indican que **el relevamiento es de alrededor de 2017**: figuran CitiBank (dejó la banca minorista en 2017), "BBVA Banco Francés" y "Banco Santander Río" (cambiaron de nombre en 2019). Hay que decirlo en la defensa y tratarlo como una foto vieja de la oferta.

## Pendiente

- Polígonos de los radios censales con población (`censo2022-caba-radios.geojson`).
- Coordenadas de usos del suelo (`usos-suelo-2022-2024-puntos.parquet`).

Los dos salen de volver a correr el notebook de descarga.
