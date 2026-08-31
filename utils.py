"""Utilidades compartidas: helpers de tipos, formato y normalización de texto."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

# --------------------------------------------------------------------------
# Helpers de dtype (compatibles con pandas 2.x y 3.x, donde el dtype string
# dejó de reportarse como object)
# --------------------------------------------------------------------------


def is_textual(s: pd.Series) -> bool:
    return ptypes.is_object_dtype(s) or ptypes.is_string_dtype(s)


def is_numeric(s: pd.Series) -> bool:
    return ptypes.is_numeric_dtype(s) and not ptypes.is_bool_dtype(s)


def is_datetime(s: pd.Series) -> bool:
    return ptypes.is_datetime64_any_dtype(s)


def is_bool(s: pd.Series) -> bool:
    return ptypes.is_bool_dtype(s)


def as_str(s: pd.Series) -> pd.Series:
    """Serie como texto, preservando NaN."""
    return s.astype("object").where(s.notna(), np.nan).map(
        lambda v: v if (isinstance(v, float) and pd.isna(v)) else str(v)
    )


# --------------------------------------------------------------------------
# Normalización de texto
# --------------------------------------------------------------------------

_MOJIBAKE_MARKERS = ("Ã¡", "Ã©", "Ã­", "Ã³", "Ãº", "Ã±", "Ã‘", "Â¿", "Â°", "â€™", "â€œ", "ï¿½")


def has_mojibake(text: str) -> bool:
    return any(m in text for m in _MOJIBAKE_MARKERS)


_MOJIBAKE_MAP = {
    "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú", "Ã¼": "ü",
    "Ã±": "ñ", "Ã‘": "Ñ", "Ã": "Á", "Ã‰": "É", "Ã": "Í", "Ã“": "Ó", "Ãš": "Ú",
    "Â¿": "¿", "Â¡": "¡", "Â°": "°", "Âº": "º", "Âª": "ª", "Â": "",
    "â€™": "'", "â€œ": "“", "â€": "”", "â€“": "–", "â€”": "—", "â€¦": "…",
    "ï¿½": "",
}


def fix_mojibake(text: str) -> str:
    """Repara texto UTF-8 leído como latin-1 (el clásico 'Ã±' por 'ñ').

    Primero intenta el round-trip completo; si el texto está mezclado (parte
    corrupta y parte correcta, que es lo habitual en archivos reales) cae a una
    sustitución por secuencias conocidas.
    """
    if not isinstance(text, str) or not has_mojibake(text):
        return text
    try:
        fixed = text.encode("latin-1", errors="strict").decode("utf-8", errors="strict")
        if not has_mojibake(fixed):
            return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    out = text
    for bad, good in _MOJIBAKE_MAP.items():
        out = out.replace(bad, good)
    return out


def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return text
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def slugify(text: str) -> str:
    """Nombre de columna seguro: minúsculas, sin acentos, snake_case."""
    text = strip_accents(str(text)).lower().strip()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "columna"


def norm_key(text: str) -> str:
    """Clave de comparación para detectar categorías casi-duplicadas."""
    return re.sub(r"\s+", " ", strip_accents(str(text)).lower().strip())


# --------------------------------------------------------------------------
# Coerción numérica tolerante (moneda, miles, decimales con coma, paréntesis)
# --------------------------------------------------------------------------

_CURRENCY = re.compile(r"[\$€£¥₡₱]|\b(mxn|usd|eur|pesos?|dlls?|dolares?)\b", re.IGNORECASE)


def clean_number_token(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"na", "n/a", "nan", "null", "none", "-", "--", "s/d", "nd"}:
        return np.nan
    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = _CURRENCY.sub("", s)
    s = s.replace("%", "").replace(" ", "").replace(" ", "")
    if "," in s and "." in s:
        # el separador decimal es el último que aparece
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        parts = s.split(",")
        # 1,234 -> miles ; 1,23 -> decimal
        s = s.replace(",", "") if len(parts[-1]) == 3 and len(parts) > 1 and parts[0] != "" else s.replace(",", ".")
    try:
        out = float(s)
    except ValueError:
        return np.nan
    return -out if negative else out


def to_numeric_series(s: pd.Series) -> pd.Series:
    if is_numeric(s):
        return s.astype(float)
    return pd.to_numeric(s.map(clean_number_token), errors="coerce")


def to_datetime_series(s: pd.Series, dayfirst: bool = True) -> pd.Series:
    if is_datetime(s):
        return s
    try:
        out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
    except (ValueError, TypeError):
        try:
            out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        except (ValueError, TypeError):
            return pd.Series(pd.NaT, index=s.index)
    return out


def numeric_convertible_ratio(s: pd.Series) -> float:
    """Qué proporción de los valores no nulos se puede leer como número."""
    non_null = s.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.sample(min(len(non_null), 2000), random_state=0)
    return float(pd.to_numeric(sample.map(clean_number_token), errors="coerce").notna().mean())


def datetime_convertible_ratio(s: pd.Series) -> float:
    non_null = s.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.sample(min(len(non_null), 1000), random_state=0)
    # evita que enteros tipo 2024 o folios se lean como fechas
    if sample.map(lambda v: isinstance(v, (int, float, np.integer, np.floating))).all():
        return 0.0
    return float(to_datetime_series(sample).notna().mean())


# --------------------------------------------------------------------------
# Formato de salida
# --------------------------------------------------------------------------


def fmt_num(value, decimals: int | None = None) -> str:
    """Formato legible para humanos: 1.2M, 34.5K, 1,234.56."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if decimals is not None:
        return f"{sign}{a:,.{decimals}f}"
    if a >= 1_000_000_000:
        return f"{sign}{a/1_000_000_000:,.2f}B"
    if a >= 1_000_000:
        return f"{sign}{a/1_000_000:,.2f}M"
    if a >= 10_000:
        return f"{sign}{a/1_000:,.1f}K"
    if a >= 1:
        return f"{sign}{a:,.2f}"
    if a == 0:
        return "0"
    return f"{sign}{a:,.4f}"


def etiqueta(v) -> str:
    """Nombre de categoría legible: 4088.0 -> '4088', NaN -> '(sin dato)'.

    Los identificadores numéricos llegan de pandas como float y se imprimen con
    un '.0' que no significa nada para quien lee.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "(sin dato)"
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    s = str(v)
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def fmt_pct(value, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{float(value) * 100:,.{decimals}f}%"


def safe_div(a, b):
    try:
        b = float(b)
        return float(a) / b if b != 0 else np.nan
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan
