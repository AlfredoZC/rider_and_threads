# Aviso de datos y licencia

Los datos geográficos de este directorio y los JSON de ciudad derivados de ellos están
bajo la licencia **Open Database License (ODbL) 1.0**.

**© OpenStreetMap contributors** — https://www.openstreetmap.org/copyright
Texto de la licencia: https://opendatacommons.org/licenses/odbl/1-0/

| Archivo | Qué es | Origen |
|---------|--------|--------|
| `data/equipetrol.osm` | Extracto de OpenStreetMap (bbox de Equipetrol) | API de openstreetmap.org |
| `data/ciudad.osm` | Extracto de OpenStreetMap (bbox de la ciudad), reducido a las etiquetas usadas | Overpass API + `tools/osm_strip_tags.py`, ver `docs/ciudad.md` |
| `config/equipetrol.json`, `config/ciudad.json` | Bases de datos derivadas: grafo simplificado + restaurantes | `tools/osm_extract.py` sobre los `.osm` |
| `data/ciudad.png` | Obra producida (*Produced Work*) dibujada con `tools/render_background.py` solo a partir de los vectores del JSON; no usa teselas de OSM | `tools/render_background.py` |
| `data/equipetrol.png` | Imagen del mapa exportada de openstreetmap.org: obra producida a partir de datos ODbL, © OpenStreetMap contributors | openstreetmap.org |

Condiciones que se siguen aquí:

- **Atribución**: cada config declara `map.attribution = "© OpenStreetMap contributors"` y la aplicación
  debe mostrarla junto al mapa.
- **Compartir igual**: si se redistribuyen estos datos o una base derivada, debe hacerse bajo ODbL.
- **Sin teselas**: no se descargan teselas de `tile.openstreetmap.org` (su *Tile Usage Policy* no permite
  su uso masivo); el fondo de la ciudad se genera localmente.
- **Sin metadatos de editores ni datos de contacto**: `data/ciudad.osm` se pidió a Overpass con `out;` y
  después se redujo con `tools/osm_strip_tags.py` a las etiquetas que usa el extractor (`highway`, `oneway`,
  `name`, `amenity`); no contiene `user`, `uid`, `changeset`, `timestamp`, `email` ni `phone`.
- **Código y datos tienen licencias distintas**: el código del repositorio sigue bajo MIT (`LICENSE`); la
  MIT **no** cubre estos datos, que siguen ODbL.
