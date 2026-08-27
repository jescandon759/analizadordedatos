"""Fase 3 — Data Preparation: transformaciones reversibles con bitácora.

Cada acción se registra para que el resultado sea auditable y reproducible,
que es el requisito real de CRISP-DM en esta fase.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import (
    fix_mojibake,
    is_numeric,
    norm_key,
    slugify,
    to_datetime_series,
    to_numeric_series,
)


@dataclass
class PrepPlan:
    drop_duplicates: bool = False
    drop_empty_rows: bool = False
    drop_columns: list[str] = field(default_factory=list)
    trim: list[str] = field(default_factory=list)
    fix_mojibake: list[str] = field(default_factory=list)
    normalize_categories: list[str] = field(default_factory=list)
    to_numeric: list[str] = field(default_factory=list)
    to_datetime: list[str] = field(default_factory=list)
    null_dates: list[str] = field(default_factory=list)
    impute: dict[str, str] = field(default_factory=dict)      # col -> estrategia
    clip_outliers: dict[str, str] = field(default_factory=dict)  # col -> "iqr" | "p1p99"
    snake_case: bool = False

    def is_empty(self) -> bool:
        return not any([
            self.drop_duplicates, self.drop_empty_rows, self.drop_columns, self.trim,
            self.fix_mojibake, self.normalize_categories, self.to_numeric,
            self.to_datetime, self.null_dates, self.impute, self.clip_outliers,
            self.snake_case,
        ])


IMPUTE_STRATEGIES = {
    "nada": "Dejar como está",
    "mediana": "Rellenar con la mediana",
    "media": "Rellenar con la media",
    "moda": "Rellenar con el valor más frecuente",
    "cero": "Rellenar con 0",
    "desconocido": "Rellenar con 'DESCONOCIDO'",
    "ffill": "Arrastrar el valor anterior",
    "eliminar_filas": "Eliminar las filas con este campo vacío",
}


def _mode_or_none(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else None


def apply_plan(df: pd.DataFrame, plan: PrepPlan) -> tuple[pd.DataFrame, list[str]]:
    """Aplica el plan y devuelve (df_limpio, bitácora)."""
    out = df.copy()
    log: list[str] = []
    n0, c0 = out.shape

    for col in plan.drop_columns:
        if col in out.columns:
            out = out.drop(columns=[col])
            log.append(f"Columna eliminada: '{col}'.")

    if plan.drop_empty_rows:
        before = len(out)
        out = out.dropna(how="all")
        if before - len(out):
            log.append(f"Eliminadas {before - len(out):,} filas totalmente vacías.")

    for col in plan.fix_mojibake:
        if col in out.columns:
            out[col] = out[col].map(lambda v: fix_mojibake(v) if isinstance(v, str) else v)
            log.append(f"Codificación reparada en '{col}'.")

    for col in plan.trim:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: " ".join(v.split()) if isinstance(v, str) else v
            )
            log.append(f"Espacios sobrantes eliminados en '{col}'.")

    for col in plan.normalize_categories:
        if col not in out.columns:
            continue
        s = out[col]
        canon: dict[str, str] = {}
        for key, grp in s.dropna().astype(str).groupby(s.dropna().astype(str).map(norm_key)):
            # canónico = la variante más frecuente
            canon[key] = grp.value_counts().index[0]
        n_changed = 0

        def _map(v):
            nonlocal n_changed
            if not isinstance(v, str):
                return v
            target = canon.get(norm_key(v), v)
            if target != v:
                n_changed += 1
            return target

        out[col] = s.map(_map)
        log.append(f"Categorías unificadas en '{col}': {n_changed:,} valores homologados.")

    for col in plan.to_numeric:
        if col in out.columns:
            before_na = int(out[col].isna().sum())
            out[col] = to_numeric_series(out[col])
            new_na = int(out[col].isna().sum()) - before_na
            log.append(
                f"'{col}' convertida a número"
                + (f"; {new_na:,} valores no convertibles quedaron como nulos." if new_na > 0 else ".")
            )

    for col in plan.to_datetime:
        if col in out.columns:
            before_na = int(out[col].isna().sum())
            out[col] = to_datetime_series(out[col])
            new_na = int(out[col].isna().sum()) - before_na
            log.append(
                f"'{col}' convertida a fecha"
                + (f"; {new_na:,} valores no convertibles quedaron como nulos." if new_na > 0 else ".")
            )

    for col in plan.null_dates:
        if col not in out.columns:
            continue
        s = to_datetime_series(out[col])
        malas = int(((s < pd.Timestamp("1950-01-01")) & s.notna()).sum())
        out[col] = s.where(s >= pd.Timestamp("1950-01-01"))
        log.append(f"'{col}': {malas:,} fechas anteriores a 1950 se vaciaron (quedan como nulas).")

    for col, strat in plan.impute.items():
        if col not in out.columns or strat in ("nada", None):
            continue
        missing = int(out[col].isna().sum())
        if missing == 0:
            continue
        if strat == "eliminar_filas":
            out = out[out[col].notna()]
            log.append(f"Eliminadas {missing:,} filas con '{col}' vacío.")
            continue
        if strat == "mediana" and is_numeric(out[col]):
            val = out[col].median()
        elif strat == "media" and is_numeric(out[col]):
            val = out[col].mean()
        elif strat == "cero":
            val = 0
        elif strat == "moda":
            val = _mode_or_none(out[col])
        elif strat == "desconocido":
            val = "DESCONOCIDO"
        elif strat == "ffill":
            out[col] = out[col].ffill().bfill()
            log.append(f"'{col}': {missing:,} nulos rellenados arrastrando el valor anterior.")
            continue
        else:
            continue
        if val is None:
            continue
        out[col] = out[col].fillna(val)
        log.append(f"'{col}': {missing:,} nulos rellenados con {strat} ({val!r}).")

    for col, method in plan.clip_outliers.items():
        if col not in out.columns or not is_numeric(out[col]):
            continue
        s = out[col].astype(float)
        if method == "p1p99":
            lo, hi = s.quantile(0.01), s.quantile(0.99)
        else:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        n_clip = int(((s < lo) | (s > hi)).sum())
        out[col] = s.clip(lo, hi)
        log.append(
            f"'{col}': {n_clip:,} valores extremos acotados al rango "
            f"[{lo:,.4g}, {hi:,.4g}] (método {method})."
        )

    if plan.drop_duplicates:
        before = len(out)
        out = out.drop_duplicates()
        if before - len(out):
            log.append(f"Eliminadas {before - len(out):,} filas duplicadas.")

    if plan.snake_case:
        mapping = {c: slugify(c) for c in out.columns}
        # evita colisiones
        seen: dict[str, int] = {}
        for k, v in list(mapping.items()):
            if v in seen:
                seen[v] += 1
                mapping[k] = f"{v}_{seen[v]}"
            else:
                seen[v] = 0
        out = out.rename(columns=mapping)
        log.append("Nombres de columna normalizados a snake_case sin acentos.")

    out = out.reset_index(drop=True)
    n1, c1 = out.shape
    log.append(
        f"Resultado: {n0:,}×{c0} → {n1:,}×{c1} "
        f"({n0 - n1:,} filas y {c0 - c1} columnas menos)."
    )
    return out, log


def plan_from_issues(issues, aggressive: bool = False) -> PrepPlan:
    """Construye un plan sugerido a partir de los problemas detectados.

    Conservador por defecto: solo repara lo que no cambia la semántica del dato.
    """
    plan = PrepPlan()
    for i in issues:
        if i.fix is None:
            continue
        if i.fix == "drop_duplicates":
            plan.drop_duplicates = True
        elif i.fix == "drop_empty_rows":
            plan.drop_empty_rows = True
        elif i.fix == "drop_column" and i.column:
            if i.code == "columna_vacia" or aggressive:
                plan.drop_columns.append(i.column)
        elif i.fix == "trim" and i.column:
            plan.trim.append(i.column)
        elif i.fix == "fix_mojibake" and i.column:
            plan.fix_mojibake.append(i.column)
        elif i.fix == "normalize_categories" and i.column:
            plan.normalize_categories.append(i.column)
        elif i.fix == "to_numeric" and i.column:
            plan.to_numeric.append(i.column)
        elif i.fix == "to_datetime" and i.column:
            plan.to_datetime.append(i.column)
        elif i.fix == "null_dates" and i.column:
            plan.null_dates.append(i.column)
        elif i.fix == "impute" and i.column and aggressive:
            plan.impute[i.column] = "mediana"
        elif i.fix == "clip_outliers" and i.column and aggressive:
            plan.clip_outliers[i.column] = "iqr"
    return plan
