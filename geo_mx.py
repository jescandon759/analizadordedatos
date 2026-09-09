"""Ubicar estados de México en un mapa, sin depender de internet.

Plotly sabe dibujar países por nombre, pero no estados mexicanos: para pintarlos
rellenos haría falta un archivo de fronteras que hay que descargar. Aquí
guardamos el centro de cada estado y los dibujamos como burbujas sobre el mapa.
Se ve dónde está cada cosa y cuánto pesa, que es para lo que sirve el mapa.

Los nombres vienen escritos de muchas formas —«CDMX», «Distrito Federal»,
«coahuila de zaragoza», «SLP»— así que todo pasa por `normaliza()` antes de
buscarse.
"""
from __future__ import annotations

import re
import unicodedata

# Centro aproximado de cada estado (lat, lon). Es el punto donde se planta la
# burbuja; no pretende ser el centroide exacto de la frontera.
CENTROS: dict[str, tuple[float, float]] = {
    "aguascalientes": (21.88, -102.29),
    "baja california": (30.47, -115.37),
    "baja california sur": (25.75, -111.75),
    "campeche": (18.85, -90.35),
    "chiapas": (16.50, -92.50),
    "chihuahua": (28.63, -106.07),
    "ciudad de mexico": (19.43, -99.13),
    "coahuila": (27.30, -102.05),
    "colima": (19.12, -103.87),
    "durango": (24.80, -104.75),
    "estado de mexico": (19.35, -99.63),
    "guanajuato": (20.92, -101.10),
    "guerrero": (17.55, -99.50),
    "hidalgo": (20.50, -98.75),
    "jalisco": (20.55, -103.55),
    "michoacan": (19.30, -101.70),
    "morelos": (18.75, -99.07),
    "nayarit": (21.75, -104.85),
    "nuevo leon": (25.60, -99.90),
    "oaxaca": (17.07, -96.72),
    "puebla": (19.05, -98.20),
    "queretaro": (20.72, -100.00),
    "quintana roo": (19.60, -88.05),
    "san luis potosi": (22.15, -100.98),
    "sinaloa": (25.00, -107.50),
    "sonora": (29.65, -110.65),
    "tabasco": (18.00, -92.90),
    "tamaulipas": (24.30, -98.65),
    "tlaxcala": (19.42, -98.20),
    "veracruz": (19.20, -96.35),
    "yucatan": (20.70, -89.10),
    "zacatecas": (23.15, -102.60),
}

# Formas alternas con las que la gente escribe lo mismo. La clave ya viene sin
# acentos y en minúsculas, tal como sale de `_plano()`.
ALIAS: dict[str, str] = {
    "cdmx": "ciudad de mexico",
    "df": "ciudad de mexico",
    "d f": "ciudad de mexico",
    "distrito federal": "ciudad de mexico",
    "mexico df": "ciudad de mexico",
    "ciudad de mexico cdmx": "ciudad de mexico",
    "mexico": "estado de mexico",
    "edomex": "estado de mexico",
    "edo de mexico": "estado de mexico",
    "edo mex": "estado de mexico",
    "estado mexico": "estado de mexico",
    "coahuila de zaragoza": "coahuila",
    "michoacan de ocampo": "michoacan",
    "veracruz de ignacio de la llave": "veracruz",
    "veracruz llave": "veracruz",
    "slp": "san luis potosi",
    "s l p": "san luis potosi",
    "nl": "nuevo leon",
    "n l": "nuevo leon",
    "bc": "baja california",
    "b c": "baja california",
    "bcn": "baja california",
    "bcs": "baja california sur",
    "b c s": "baja california sur",
    "qroo": "quintana roo",
    "q roo": "quintana roo",
    "qro": "queretaro",
    "queretaro de arteaga": "queretaro",
    "ags": "aguascalientes",
    "gto": "guanajuato",
    "jal": "jalisco",
    "chih": "chihuahua",
    "tamps": "tamaulipas",
    "yuc": "yucatan",
    "zac": "zacatecas",
    "son": "sonora",
    "sin": "sinaloa",
    "pue": "puebla",
    "ver": "veracruz",
    "oax": "oaxaca",
    "chis": "chiapas",
    "tab": "tabasco",
    "tlax": "tlaxcala",
    "mor": "morelos",
    "hgo": "hidalgo",
    "gro": "guerrero",
    "dgo": "durango",
    "col": "colima",
    "nay": "nayarit",
    "camp": "campeche",
    "mich": "michoacan",
}

# Cómo se escribe bonito al mostrarlo, ya normalizado.
BONITO: dict[str, str] = {
    "ciudad de mexico": "Ciudad de México",
    "estado de mexico": "Estado de México",
    "nuevo leon": "Nuevo León",
    "san luis potosi": "San Luis Potosí",
    "queretaro": "Querétaro",
    "michoacan": "Michoacán",
    "yucatan": "Yucatán",
    "baja california sur": "Baja California Sur",
    "baja california": "Baja California",
    "quintana roo": "Quintana Roo",
}


def _plano(texto: str) -> str:
    """Minúsculas, sin acentos y sin puntuación: «S.L.P.» y «slp» se juntan."""
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9ñ ]+", " ", s)).strip()


def normaliza(nombre) -> str | None:
    """Devuelve la clave del estado, o None si eso no es un estado de México."""
    if nombre is None:
        return None
    p = _plano(nombre)
    if not p:
        return None
    if p in CENTROS:
        return p
    if p in ALIAS:
        return ALIAS[p]
    # «estado de jalisco», «edo. de sonora»
    sin_prefijo = re.sub(r"^(estado de|edo de|edo)\s+", "", p).strip()
    if sin_prefijo in CENTROS:
        return sin_prefijo
    return None


def bonito(clave: str) -> str:
    return BONITO.get(clave, clave.title())


def es_columna_de_estados(valores, minimo: float = 0.6) -> bool:
    """¿Esta columna trae estados de México?

    Pide que la mayoría de los valores distintos se reconozcan. Con el umbral en
    0.6 una columna de ciudades no pasa (aunque «Querétaro» y «Aguascalientes»
    sean ciudad y estado a la vez, son la excepción entre cientos de municipios),
    y una de estados sí pasa aunque traiga un par de valores sucios.
    """
    distintos = {str(v) for v in valores if str(v).strip() and str(v).lower() != "nan"}
    if len(distintos) < 3:
        return False
    reconocidos = sum(1 for v in distintos if normaliza(v))
    return reconocidos / len(distintos) >= minimo


def agrupa_por_estado(serie) -> dict[str, float]:
    """Suma una serie indexada por nombre de estado, juntando las variantes.

    Aquí es donde «méxico» y «estado de méxico» dejan de ser dos renglones.
    """
    out: dict[str, float] = {}
    for nombre, valor in serie.items():
        clave = normaliza(nombre)
        if clave is None:
            continue
        try:
            v = float(valor)
        except (TypeError, ValueError):
            continue
        out[clave] = out.get(clave, 0.0) + v
    return out


def con_coordenadas(agrupado: dict[str, float]):
    """(etiquetas, lats, lons, valores) listos para dibujar."""
    claves = [k for k in agrupado if k in CENTROS]
    claves.sort(key=lambda k: agrupado[k], reverse=True)
    return ([bonito(k) for k in claves],
            [CENTROS[k][0] for k in claves],
            [CENTROS[k][1] for k in claves],
            [agrupado[k] for k in claves])
