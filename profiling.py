"""Fase 2 — Data Understanding: perfilado de columnas y detección de tipos semánticos."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from utils import (
    datetime_convertible_ratio,
    is_bool,
    is_datetime,
    is_numeric,
    is_textual,
    norm_key,
    numeric_convertible_ratio,
    strip_accents,
    to_datetime_series,
    to_numeric_series,
)

# ---------------------------------------------------------------- pistas de nombre

NAME_HINTS: dict[str, tuple[str, ...]] = {
    "monetario": (
        "precio", "monto", "importe", "total", "subtotal", "venta", "ventas", "ingreso",
        "ingresos", "costo", "coste", "gasto", "pago", "saldo", "cargo", "abono",
        "utilidad", "margen", "comision", "salario", "sueldo", "factura", "ticket",
        "amount", "price", "revenue", "sales", "cost", "profit", "payment", "balance",
        "value", "valor", "iva", "descuento",
    ),
    "cantidad": (
        "cantidad", "cant", "unidades", "piezas", "pzas", "volumen", "stock",
        "existencia", "inventario", "qty", "quantity", "units", "count", "conteo",
        "num_", "numero_de",
    ),
    "fecha": (
        "fecha", "date", "dia", "mes", "anio", "año", "ano", "periodo", "timestamp",
        "created", "alta", "baja", "vencimiento", "emision", "registro", "hora",
    ),
    "identificador": (
        "id", "folio", "clave", "cve", "codigo", "sku", "matricula", "rfc", "curp",
        "serie", "numero", "no", "num", "referencia", "uuid",
    ),
    "porcentaje": ("porcentaje", "pct", "percent", "tasa", "ratio", "%", "avance", "cumplimiento"),
    "geografia": (
        "estado", "ciudad", "municipio", "pais", "region", "zona", "sucursal",
        "codigo_postal", "cp", "colonia", "direccion", "state", "city", "country",
    ),
    "persona": (
        "cliente", "vendedor", "empleado", "usuario", "nombre", "contacto",
        "proveedor", "responsable", "agente", "customer", "user", "employee",
    ),
    "contacto": ("email", "correo", "mail", "telefono", "celular", "phone", "whatsapp"),
}

BOOL_TOKENS = {
    "si", "no", "s", "n", "true", "false", "verdadero", "falso", "yes",
    "1", "0", "activo", "inactivo", "vigente", "cancelado",
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")
PHONE_RE = re.compile(r"^[\d\s\-\+\(\)\.]{7,20}$")


def _name_role(colname: str) -> str | None:
    key = strip_accents(str(colname)).lower()
    tokens = set(re.split(r"[^a-z0-9]+", key)) - {""}
    for role, hints in NAME_HINTS.items():
        for h in hints:
            hs = strip_accents(h).lower()
            if hs in tokens or (len(hs) > 3 and hs in key):
                return role
    return None


# ---------------------------------------------------------------- perfil


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic: str            # numerico | categorico | fecha | booleano | texto | identificador | constante | vacia
    role: str | None         # monetario | cantidad | fecha | identificador | ... (pista de negocio)
    n: int = 0
    n_missing: int = 0
    pct_missing: float = 0.0
    n_unique: int = 0
    unique_ratio: float = 0.0
    stats: dict = field(default_factory=dict)
    top_values: list = field(default_factory=list)
    sample: list = field(default_factory=list)
    convertible_numeric: float = 0.0
    convertible_datetime: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def profile_column(s: pd.Series, name: str) -> ColumnProfile:
    n = int(len(s))
    n_missing = int(s.isna().sum())
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    unique_ratio = n_unique / max(len(non_null), 1)
    role = _name_role(name)
    notes: list[str] = []
    stats: dict = {}
    top_values: list = []
    conv_num = conv_dt = 0.0

    if len(non_null) == 0:
        semantic = "vacia"
    elif n_unique == 1:
        semantic = "constante"
    elif is_bool(s):
        semantic = "booleano"
    elif is_datetime(s):
        semantic = "fecha"
    elif is_numeric(s):
        semantic = "numerico"
    else:
        conv_num = numeric_convertible_ratio(s)
        conv_dt = datetime_convertible_ratio(s)
        low = {norm_key(v) for v in non_null.unique()[:20]}
        if n_unique <= 2 and low <= BOOL_TOKENS:
            semantic = "booleano"
        elif conv_dt >= 0.85 and conv_dt > conv_num:
            semantic = "fecha"
            notes.append("Fecha almacenada como texto.")
        elif conv_num >= 0.85:
            semantic = "numerico"
            notes.append("Número almacenado como texto.")
        elif unique_ratio > 0.95 and len(non_null) > 20:
            semantic = "identificador"
        else:
            avg_len = float(non_null.astype(str).str.len().mean())
            semantic = "texto" if (avg_len > 40 and unique_ratio > 0.5) else "categorico"

    if semantic == "numerico":
        vals = to_numeric_series(s).dropna()
        if len(vals):
            q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
            stats = {
                "min": float(vals.min()), "max": float(vals.max()),
                "mean": float(vals.mean()), "median": float(vals.median()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "sum": float(vals.sum()),
                "q1": q1, "q3": q3, "iqr": q3 - q1,
                "skew": float(vals.skew()) if len(vals) > 2 else 0.0,
                "zeros": int((vals == 0).sum()),
                "negatives": int((vals < 0).sum()),
            }
            if unique_ratio > 0.95 and float(vals.apply(float.is_integer).mean() if vals.dtype.kind == "f" else 1) == 1:
                if role == "identificador":
                    semantic = "identificador"
    elif semantic == "fecha":
        vals = to_datetime_series(s).dropna()
        if len(vals):
            stats = {
                "min": vals.min(), "max": vals.max(),
                "rango_dias": int((vals.max() - vals.min()).days),
            }
    elif semantic in ("categorico", "booleano", "constante", "texto", "identificador"):
        vc = non_null.astype(str).value_counts()
        top_values = [(str(k), int(v)) for k, v in vc.head(15).items()]
        stats = {"moda": str(vc.index[0]) if len(vc) else None,
                 "freq_moda": int(vc.iloc[0]) if len(vc) else 0}

    if role is None and semantic == "identificador":
        role = "identificador"

    return ColumnProfile(
        name=name, dtype=str(s.dtype), semantic=semantic, role=role,
        n=n, n_missing=n_missing, pct_missing=n_missing / max(n, 1),
        n_unique=n_unique, unique_ratio=unique_ratio,
        stats=stats, top_values=top_values,
        sample=[str(v) for v in non_null.head(5).tolist()],
        convertible_numeric=conv_num, convertible_datetime=conv_dt,
        notes=notes,
    )


def profile_dataframe(df: pd.DataFrame) -> dict[str, ColumnProfile]:
    return {c: profile_column(df[c], c) for c in df.columns}


# ---------------------------------------------------------------- vistas derivadas


def profiles_table(profiles: dict[str, ColumnProfile]) -> pd.DataFrame:
    rows = []
    for p in profiles.values():
        rows.append({
            "Columna": p.name,
            "Tipo detectado": p.semantic,
            "Rol de negocio": p.role or "—",
            "dtype": p.dtype,
            "Nulos": p.n_missing,
            "% Nulos": round(p.pct_missing * 100, 1),
            "Únicos": p.n_unique,
            "Ejemplo": p.sample[0] if p.sample else "—",
        })
    return pd.DataFrame(rows)


def columns_by(profiles: dict[str, ColumnProfile], *semantics: str) -> list[str]:
    return [p.name for p in profiles.values() if p.semantic in semantics]


def columns_by_role(profiles: dict[str, ColumnProfile], *roles: str) -> list[str]:
    return [p.name for p in profiles.values() if p.role in roles]


def suggest_metric_columns(profiles: dict[str, ColumnProfile]) -> list[str]:
    """Columnas numéricas que tienen sentido sumar (excluye ids y folios).

    Ojo: una columna de importes con decimales tiene casi todos sus valores
    distintos, igual que un folio. Lo que los separa es que el folio es entero
    y no tiene rol de negocio.
    """
    out = []
    for p in profiles.values():
        if p.semantic != "numerico" or p.role == "identificador":
            continue
        rango = float(p.stats.get("max", 0)) - float(p.stats.get("min", 0)) + 1
        consecutiva = 0 < rango <= p.n_unique * 1.2          # firma de un folio
        if (p.unique_ratio > 0.99 and p.n > 50 and consecutiva
                and p.role not in ("monetario", "cantidad", "porcentaje")):
            continue
        out.append(p.name)
    # las monetarias y de cantidad primero
    out.sort(key=lambda c: 0 if profiles[c].role in ("monetario", "cantidad") else 1)
    return out


def suggest_date_columns(profiles: dict[str, ColumnProfile]) -> list[str]:
    return [p.name for p in profiles.values() if p.semantic == "fecha"]


def suggest_dimension_columns(profiles: dict[str, ColumnProfile], max_card: int = 60) -> list[str]:
    out = [
        p.name for p in profiles.values()
        if p.semantic in ("categorico", "booleano") and 1 < p.n_unique <= max_card
    ]
    out.sort(key=lambda c: profiles[c].n_unique)
    return out


def dataset_overview(df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict:
    counts: dict[str, int] = {}
    for p in profiles.values():
        counts[p.semantic] = counts.get(p.semantic, 0) + 1
    total_cells = df.shape[0] * max(df.shape[1], 1)
    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "celdas_vacias": int(df.isna().sum().sum()),
        "pct_vacias": float(df.isna().sum().sum() / max(total_cells, 1)),
        "filas_duplicadas": int(df.duplicated().sum()),
        "memoria_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "tipos": counts,
    }
