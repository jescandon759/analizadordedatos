"""Fase 2 — Detección de errores y problemas de calidad de datos.

Cada hallazgo es un `Issue` con severidad, evidencia numérica y, cuando aplica,
una acción de reparación que la fase de Preparación puede ejecutar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from profiling import ColumnProfile, EMAIL_RE, PHONE_RE
from utils import (
    fmt_num,
    has_mojibake,
    is_numeric,
    norm_key,
    to_datetime_series,
    to_numeric_series,
)

CRITICO, ADVERTENCIA, INFO = "crítico", "advertencia", "info"
SEVERITY_ORDER = {CRITICO: 0, ADVERTENCIA: 1, INFO: 2}


@dataclass
class Issue:
    code: str
    severity: str
    title: str
    detail: str
    column: str | None = None
    n_affected: int = 0
    pct_affected: float = 0.0
    fix: str | None = None            # clave de acción para prep.py
    fix_label: str | None = None
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------- reglas


def _rows_and_duplicates(df: pd.DataFrame, out: list[Issue]) -> None:
    n = len(df)
    if n == 0:
        out.append(Issue("sin_filas", CRITICO, "El archivo no tiene filas de datos",
                         "Después de descartar filas vacías no queda ningún registro."))
        return

    dups = int(df.duplicated().sum())
    if dups:
        out.append(Issue(
            "filas_duplicadas", CRITICO if dups / n > 0.05 else ADVERTENCIA,
            "Filas completamente duplicadas",
            f"{dups:,} de {n:,} filas ({dups/n:.1%}) están repetidas en todas sus columnas. "
            "Inflan cualquier suma, conteo o promedio.",
            n_affected=dups, pct_affected=dups / n,
            fix="drop_duplicates", fix_label="Eliminar filas duplicadas",
        ))

    empty_rows = int(df.isna().all(axis=1).sum())
    if empty_rows:
        out.append(Issue(
            "filas_vacias", ADVERTENCIA, "Filas totalmente vacías",
            f"{empty_rows:,} filas no tienen ningún valor.",
            n_affected=empty_rows, pct_affected=empty_rows / n,
            fix="drop_empty_rows", fix_label="Eliminar filas vacías",
        ))

    if n < 30:
        out.append(Issue(
            "pocos_registros", ADVERTENCIA, "Muy pocos registros",
            f"Solo hay {n} filas. Las conclusiones estadísticas y cualquier modelo "
            "serán poco confiables.",
        ))


def _candidate_key_duplicates(df: pd.DataFrame, profiles, out: list[Issue]) -> None:
    for p in profiles.values():
        if p.semantic != "identificador" and p.role != "identificador":
            continue
        # un SKU o una clave de categoría se repite por diseño; solo es un
        # problema cuando la columna casi es única y unos pocos valores se salen
        if p.unique_ratio < 0.9:
            continue
        s = df[p.name].dropna()
        if len(s) < 10:
            continue
        dups = int(s.duplicated().sum())
        if dups:
            out.append(Issue(
                "id_duplicado", CRITICO,
                f"Identificador repetido en '{p.name}'",
                f"'{p.name}' parece un identificador único, pero {dups:,} valores se repiten. "
                "Es señal de doble captura o de un join mal hecho aguas arriba.",
                column=p.name, n_affected=dups, pct_affected=dups / len(s),
            ))


def _missing_values(df: pd.DataFrame, profiles, out: list[Issue]) -> None:
    n = max(len(df), 1)
    for p in profiles.values():
        if p.pct_missing == 0:
            continue
        if p.pct_missing >= 0.6:
            sev, extra = CRITICO, " Prácticamente no aporta información; considera eliminarla."
        elif p.pct_missing >= 0.2:
            sev, extra = ADVERTENCIA, " Imputar o filtrar antes de sacar conclusiones."
        else:
            sev, extra = INFO, ""
        out.append(Issue(
            "nulos", sev, f"Valores faltantes en '{p.name}'",
            f"{p.n_missing:,} de {n:,} valores ({p.pct_missing:.1%}) están vacíos.{extra}",
            column=p.name, n_affected=p.n_missing, pct_affected=p.pct_missing,
            fix="impute", fix_label=f"Tratar nulos en '{p.name}'",
        ))


def _constant_and_empty(profiles, out: list[Issue]) -> None:
    for p in profiles.values():
        if p.semantic == "vacia":
            out.append(Issue(
                "columna_vacia", CRITICO, f"Columna '{p.name}' completamente vacía",
                "No contiene ningún valor.", column=p.name,
                n_affected=p.n, pct_affected=1.0,
                fix="drop_column", fix_label=f"Eliminar columna '{p.name}'",
            ))
        elif p.semantic == "constante":
            out.append(Issue(
                "columna_constante", ADVERTENCIA, f"Columna '{p.name}' con un solo valor",
                f"Todos los registros valen '{p.stats.get('moda')}'. No sirve para segmentar "
                "ni para modelar.", column=p.name,
                fix="drop_column", fix_label=f"Eliminar columna '{p.name}'",
            ))


def _type_mismatch(profiles, out: list[Issue]) -> None:
    for p in profiles.values():
        if "Número almacenado como texto." in p.notes:
            bad = 1 - p.convertible_numeric
            out.append(Issue(
                "numero_como_texto", CRITICO,
                f"'{p.name}' guarda números como texto",
                f"El {p.convertible_numeric:.0%} de los valores son numéricos pero la columna es texto "
                f"(símbolos de moneda, separadores de miles o espacios). No se puede sumar ni promediar así."
                + (f" Además, {bad:.0%} de los valores no se pueden convertir." if bad > 0.01 else ""),
                column=p.name, pct_affected=1.0,
                fix="to_numeric", fix_label=f"Convertir '{p.name}' a número",
            ))
        if "Fecha almacenada como texto." in p.notes:
            out.append(Issue(
                "fecha_como_texto", ADVERTENCIA,
                f"'{p.name}' guarda fechas como texto",
                f"El {p.convertible_datetime:.0%} de los valores se pueden leer como fecha. "
                "Sin convertir, no hay análisis temporal posible.",
                column=p.name, pct_affected=1.0,
                fix="to_datetime", fix_label=f"Convertir '{p.name}' a fecha",
            ))
        # texto con algunos números sueltos (tipo mixto real)
        if p.semantic == "categorico" and 0.2 < p.convertible_numeric < 0.85:
            out.append(Issue(
                "tipo_mixto", ADVERTENCIA, f"Tipos mezclados en '{p.name}'",
                f"El {p.convertible_numeric:.0%} de los valores son números y el resto texto "
                "(por ejemplo 'N/D', 'pendiente', notas). Decide una convención antes de analizar.",
                column=p.name, pct_affected=p.convertible_numeric,
            ))


def _text_hygiene(df: pd.DataFrame, profiles, out: list[Issue]) -> None:
    for p in profiles.values():
        if p.semantic not in ("categorico", "texto", "booleano", "identificador"):
            continue
        s = df[p.name].dropna().astype(str)
        if s.empty:
            continue

        ws = int((s != s.str.strip()).sum())
        if ws:
            out.append(Issue(
                "espacios", ADVERTENCIA, f"Espacios sobrantes en '{p.name}'",
                f"{ws:,} valores tienen espacios al inicio o al final. Provoca categorías "
                "duplicadas invisibles ('Norte' ≠ 'Norte ').",
                column=p.name, n_affected=ws, pct_affected=ws / len(s),
                fix="trim", fix_label=f"Quitar espacios en '{p.name}'",
            ))

        moji = int(s.map(has_mojibake).sum())
        if moji:
            out.append(Issue(
                "mojibake", ADVERTENCIA, f"Caracteres corruptos en '{p.name}'",
                f"{moji:,} valores muestran texto tipo 'Ã±' o 'Ã©': el archivo se guardó con "
                "una codificación distinta a UTF-8.",
                column=p.name, n_affected=moji, pct_affected=moji / len(s),
                fix="fix_mojibake", fix_label=f"Reparar codificación en '{p.name}'",
            ))

        if p.semantic in ("categorico", "booleano") and 1 < p.n_unique <= 500:
            groups: dict[str, set[str]] = {}
            for v in s.unique():
                groups.setdefault(norm_key(v), set()).add(v)
            collisions = {k: v for k, v in groups.items() if len(v) > 1}
            if collisions:
                ejemplos = "; ".join(
                    " / ".join(sorted(v)[:3]) for v in list(collisions.values())[:3]
                )
                afectados = int(sum(
                    s.isin(v).sum() for v in collisions.values()
                ))
                out.append(Issue(
                    "categorias_inconsistentes", CRITICO,
                    f"Categorías escritas de varias formas en '{p.name}'",
                    f"{len(collisions)} grupos de valores solo difieren en mayúsculas, acentos o "
                    f"espacios: {ejemplos}. Se cuentan como distintos y parten tus totales.",
                    column=p.name, n_affected=afectados, pct_affected=afectados / len(s),
                    fix="normalize_categories",
                    fix_label=f"Unificar categorías en '{p.name}'",
                    payload={"grupos": {k: sorted(v) for k, v in list(collisions.items())[:50]}},
                ))

        if p.role == "contacto":
            if "mail" in p.name.lower() or "correo" in p.name.lower():
                bad = int((~s.str.match(EMAIL_RE)).sum())
                if bad:
                    out.append(Issue(
                        "email_invalido", ADVERTENCIA, f"Correos con formato inválido en '{p.name}'",
                        f"{bad:,} valores no tienen forma de correo electrónico.",
                        column=p.name, n_affected=bad, pct_affected=bad / len(s),
                    ))
            else:
                bad = int((~s.str.match(PHONE_RE)).sum())
                if bad:
                    out.append(Issue(
                        "telefono_invalido", INFO, f"Teléfonos con formato irregular en '{p.name}'",
                        f"{bad:,} valores no parecen números telefónicos.",
                        column=p.name, n_affected=bad, pct_affected=bad / len(s),
                    ))


def _numeric_sanity(df: pd.DataFrame, profiles, out: list[Issue]) -> None:
    for p in profiles.values():
        if p.semantic != "numerico" or not p.stats:
            continue
        # un SKU "extremo" o un folio "sesgado" no significan nada: son etiquetas
        # que resultan ser números, no cantidades
        if p.role == "identificador":
            continue
        vals = to_numeric_series(df[p.name]).dropna()
        if vals.empty:
            continue
        st = p.stats

        if p.role in ("monetario", "cantidad") and st.get("negatives", 0):
            neg = st["negatives"]
            out.append(Issue(
                "negativos_sospechosos", ADVERTENCIA,
                f"Valores negativos en '{p.name}'",
                f"{neg:,} registros son negativos en una columna que parece "
                f"{'monetaria' if p.role=='monetario' else 'de cantidad'}. Pueden ser devoluciones "
                "legítimas o errores de signo — hay que distinguirlos antes de sumar.",
                column=p.name, n_affected=neg, pct_affected=neg / len(vals),
            ))

        q1, q3, iqr = st["q1"], st["q3"], st["iqr"]
        if iqr > 0:
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            extremos = int(((vals < lo) | (vals > hi)).sum())
            if extremos:
                out.append(Issue(
                    "outliers", ADVERTENCIA if extremos / len(vals) < 0.05 else CRITICO,
                    f"Valores extremos en '{p.name}'",
                    f"{extremos:,} registros ({extremos/len(vals):.1%}) caen fuera de "
                    f"[{fmt_num(lo)}, {fmt_num(hi)}] (3×RIC). Máximo observado: {fmt_num(st['max'])}, "
                    f"mediana: {fmt_num(st['median'])}.",
                    column=p.name, n_affected=extremos, pct_affected=extremos / len(vals),
                    fix="clip_outliers", fix_label=f"Acotar extremos en '{p.name}'",
                    payload={"low": lo, "high": hi},
                ))

        if abs(st.get("skew", 0)) > 3:
            out.append(Issue(
                "asimetria", INFO, f"Distribución muy sesgada en '{p.name}'",
                f"Asimetría = {st['skew']:.1f}. El promedio ({fmt_num(st['mean'])}) no representa "
                f"al registro típico; usa la mediana ({fmt_num(st['median'])}).",
                column=p.name,
            ))

        if p.role == "porcentaje":
            fuera = int(((vals < 0) | (vals > 100)).sum())
            if fuera:
                out.append(Issue(
                    "porcentaje_fuera_rango", ADVERTENCIA,
                    f"Porcentajes fuera de rango en '{p.name}'",
                    f"{fuera:,} valores están fuera de 0–100.",
                    column=p.name, n_affected=fuera, pct_affected=fuera / len(vals),
                ))

        zeros = st.get("zeros", 0)
        if p.role == "monetario" and zeros / len(vals) > 0.3:
            out.append(Issue(
                "muchos_ceros", ADVERTENCIA, f"Exceso de ceros en '{p.name}'",
                f"{zeros:,} registros ({zeros/len(vals):.1%}) valen 0. Suele indicar que el "
                "campo se dejó vacío y se guardó como cero, lo que hunde los promedios.",
                column=p.name, n_affected=zeros, pct_affected=zeros / len(vals),
            ))


def _date_sanity(df: pd.DataFrame, profiles, out: list[Issue]) -> None:
    hoy = pd.Timestamp.today().normalize()
    for p in profiles.values():
        if p.semantic != "fecha":
            continue
        vals = to_datetime_series(df[p.name]).dropna()
        if vals.empty:
            continue
        futuras = int((vals > hoy).sum())
        if futuras:
            out.append(Issue(
                "fechas_futuras", ADVERTENCIA, f"Fechas en el futuro en '{p.name}'",
                f"{futuras:,} registros tienen fecha posterior a hoy (máx: {vals.max():%Y-%m-%d}). "
                "Puede ser dato programado o error de captura.",
                column=p.name, n_affected=futuras, pct_affected=futuras / len(vals),
            ))
        antiguas = int((vals < pd.Timestamp("1950-01-01")).sum())
        if antiguas:
            out.append(Issue(
                "fechas_imposibles", CRITICO, f"Fechas imposibles en '{p.name}'",
                f"{antiguas:,} registros son anteriores a 1950 (mín: {vals.min():%Y-%m-%d}). "
                "Típico de fechas vacías convertidas a 1900-01-01. Distorsionan cualquier "
                "análisis temporal.",
                column=p.name, n_affected=antiguas, pct_affected=antiguas / len(vals),
                fix="null_dates", fix_label=f"Vaciar fechas imposibles de '{p.name}'",
            ))
        # huecos en la serie
        if len(vals) > 30:
            dias = pd.Series(sorted(vals.dt.normalize().unique()))
            if len(dias) > 5:
                gaps = dias.diff().dt.days.dropna()
                gmax = float(gaps.max())
                gmed = float(gaps.median()) or 1.0
                if gmax > max(gmed * 10, 30):
                    idx = int(gaps.idxmax())
                    out.append(Issue(
                        "hueco_temporal", ADVERTENCIA, f"Hueco en la serie de '{p.name}'",
                        f"Hay {int(gmax)} días sin registros entre {dias.iloc[idx-1]:%Y-%m-%d} y "
                        f"{dias.iloc[idx]:%Y-%m-%d}, contra un intervalo típico de {gmed:.0f} día(s). "
                        "Puede faltar un periodo completo de datos.",
                        column=p.name,
                    ))


def _redundancy(df: pd.DataFrame, profiles, out: list[Issue]) -> None:
    num_cols = [p.name for p in profiles.values()
                if p.semantic == "numerico" and p.role != "identificador"]
    if len(num_cols) < 2:
        return
    sub = df[num_cols].apply(to_numeric_series)
    if len(sub.dropna(how="all")) < 10:
        return
    corr = sub.corr(numeric_only=True).abs()
    seen = set()
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1:]:
            r = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
            if pd.notna(r) and r > 0.98 and (a, b) not in seen:
                seen.add((a, b))
                out.append(Issue(
                    "columnas_redundantes", INFO,
                    f"'{a}' y '{b}' son prácticamente la misma columna",
                    f"Correlación de {r:.3f}. Una probablemente se calcula de la otra; "
                    "si vas a modelar, conserva solo una.",
                    payload={"a": a, "b": b, "r": float(r)},
                ))


def _high_cardinality(profiles, out: list[Issue]) -> None:
    for p in profiles.values():
        if p.semantic == "categorico" and p.n_unique > 50 and p.unique_ratio > 0.5:
            out.append(Issue(
                "alta_cardinalidad", INFO, f"'{p.name}' tiene demasiadas categorías",
                f"{p.n_unique:,} valores distintos sobre {p.n:,} registros "
                f"({p.unique_ratio:.0%} únicos). Como dimensión de gráfica es ilegible; "
                "conviene agruparla o usar solo el Top N.",
                column=p.name,
            ))


# ---------------------------------------------------------------- API


def detect_issues(df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> list[Issue]:
    out: list[Issue] = []
    _rows_and_duplicates(df, out)
    if len(df) == 0:
        return out
    _candidate_key_duplicates(df, profiles, out)
    _constant_and_empty(profiles, out)
    _missing_values(df, profiles, out)
    _type_mismatch(profiles, out)
    _text_hygiene(df, profiles, out)
    _numeric_sanity(df, profiles, out)
    _date_sanity(df, profiles, out)
    _redundancy(df, profiles, out)
    _high_cardinality(profiles, out)
    out.sort(key=lambda i: (SEVERITY_ORDER[i.severity], -i.pct_affected))
    return out


def quality_score(df: pd.DataFrame, issues: list[Issue]) -> tuple[int, dict]:
    """Puntaje 0-100. Penaliza por severidad y por alcance del problema."""
    if len(df) == 0:
        return 0, {"crítico": 1, "advertencia": 0, "info": 0}
    weights = {CRITICO: 12.0, ADVERTENCIA: 5.0, INFO: 1.5}
    # Se penaliza por TIPO de problema, no por cada columna afectada: veinte
    # columnas con nulos son un problema de nulos, no veinte problemas.
    por_codigo: dict[str, list[Issue]] = {}
    for i in issues:
        por_codigo.setdefault(i.code, []).append(i)
    penalty = 0.0
    for grupo in por_codigo.values():
        w = weights[max(grupo, key=lambda i: -SEVERITY_ORDER[i.severity]).severity]
        alcance = 0.35 + 0.65 * min(max(max(i.pct_affected for i in grupo), 0.0), 1.0)
        repeticion = min(0.3 * (len(grupo) - 1), 1.5)
        penalty += w * (alcance + repeticion)
    score = int(round(max(0.0, 100.0 - penalty)))
    counts = {s: sum(1 for i in issues if i.severity == s) for s in (CRITICO, ADVERTENCIA, INFO)}
    return score, counts


def issues_table(issues: list[Issue]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Severidad": i.severity,
        "Columna": i.column or "(tabla completa)",
        "Problema": i.title,
        "Registros": i.n_affected or "—",
        "% Afectado": round(i.pct_affected * 100, 1) if i.pct_affected else "—",
        "Detalle": i.detail,
    } for i in issues])
