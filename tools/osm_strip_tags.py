#!/usr/bin/env python3
"""Reduce un export de OpenStreetMap a lo que usa tools/osm_extract.py.

Conserva coordenadas de nodos y las referencias de las vias, y SOLO estas etiquetas:
  - way:  highway, oneway, name
  - node: amenity (restaurant / fast_food) y name
Elimina todo lo demas (email, phone, website, addr:*, opening_hours...) y los metadatos de editores
(user, uid, changeset, timestamp, version). Sin red, solo biblioteca estandar, en streaming.

  python3 tools/osm_strip_tags.py crudo.osm data/ciudad.osm

Datos (c) OpenStreetMap contributors, licencia ODbL (ver data/NOTICE.md).
"""
import os
import sys
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape, quoteattr

WAY_TAGS = ("highway", "oneway", "name")
NODE_TAGS = ("amenity", "name")
AMENITIES = ("restaurant", "fast_food")


def attr(name, value):
    return f"{name}={quoteattr(value)}"


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if os.path.exists(dst) and os.path.samefile(src, dst):
        sys.exit("error: la salida es el mismo archivo que la entrada; usa otra ruta")
    tmp = dst + ".tmp"
    try:
        convert(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def convert(src, dst):
    with open(dst, "w", encoding="utf-8", newline="\n") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        out.write('<osm version="0.6" generator="tools/osm_strip_tags.py">\n')
        depth = 0
        for ev, el in ET.iterparse(src, events=("start", "end")):
            if ev == "start":
                depth += 1
                continue
            depth -= 1
            if depth != 1:  # solo hijos directos de <osm>: un elemento anidado (p. ej. dentro de <note>) se ignora
                continue
            if el.tag == "note":
                out.write(f"<note>{escape(el.text or '')}</note>\n")
            elif el.tag == "meta" and el.get("osm_base"):
                out.write(f"<meta {attr('osm_base', el.get('osm_base'))}/>\n")
            elif el.tag == "node":
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                keep = []
                if tags.get("amenity") in AMENITIES:
                    keep = [(k, tags[k]) for k in NODE_TAGS if k in tags]
                head = (f'  <node {attr("id", el.get("id"))} {attr("lat", el.get("lat"))} '
                        f'{attr("lon", el.get("lon"))}')
                if keep:
                    out.write(head + ">\n")
                    for k, v in keep:
                        out.write(f"    <tag {attr('k', k)} {attr('v', v)}/>\n")
                    out.write("  </node>\n")
                else:
                    out.write(head + "/>\n")
                el.clear()
            elif el.tag == "way":
                tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
                out.write(f'  <way {attr("id", el.get("id"))}>\n')
                for nd in el.findall("nd"):
                    out.write(f'    <nd {attr("ref", nd.get("ref"))}/>\n')
                for k in WAY_TAGS:
                    if k in tags:
                        out.write(f"    <tag {attr('k', k)} {attr('v', tags[k])}/>\n")
                out.write("  </way>\n")
                el.clear()
        out.write("</osm>\n")


if __name__ == "__main__":
    main()
