"""Carga robusta de archivos CSV / TSV / Excel subidos por el usuario."""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

from .utils import has_mojibake

EXTENSIONS = [".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"]
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
SEPARATORS = [",", ";", "\t", "|"]
MAX_PREVIEW_ROWS = 30


@dataclass
class LoadResult:
    df: pd.DataFrame
    source_name: str
    kind: str                      # "excel" | "delimitado"
    sheet: str | None = None
    sheets: list[str] = field(default_factory=list)
    encoding: str | None = None
    separator: str | None = None
    header_row: int = 0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- utilidades


def _decode(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1 (con reemplazos)"


def _sniff_separator(text: str) -> str:
    head = "\n".join(text.splitlines()[:25])
    if not head:
        return ","
    scores = {}
    for sep in SEPARATORS:
        counts = [line.count(sep) for line in head.splitlines() if line.strip()]
        if not counts or max(counts) == 0:
            continue
        # premia el separador con conteo alto y consistente entre líneas
        consistency = sum(1 for c in counts if c == counts[0]) / len(counts)
        scores[sep] = max(counts) * consistency
    return max(scores, key=scores.get) if scores else ","


def _guess_header_row(df_raw: pd.DataFrame, max_scan: int = 8) -> int:
    """Detecta encabezados desplazados (títulos y filas vacías arriba, típico de Excel)."""
    best_row, best_score = 0, -1.0
    limit = min(max_scan, len(df_raw))
    for i in range(limit):
        row = df_raw.iloc[i]
        filled = row.notna().sum()
        if filled < 2:
            continue
        texty = sum(1 for v in row.dropna() if isinstance(v, str) and v.strip() != "")
        unique = row.dropna().astype(str).nunique()
        score = filled + texty * 0.5 + unique * 0.5
        # el encabezado real suele tener más columnas llenas que las filas basura previas
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def _postprocess(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        f"columna_{i+1}" if (str(c).strip() == "" or str(c).lower().startswith("unnamed"))
        else str(c).strip()
        for i, c in enumerate(df.columns)
    ]
    # nombres duplicados
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
            warnings.append(f"Columna duplicada '{c}' renombrada a '{c}_{seen[c]}'.")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols
    # Se descartan filas totalmente vacías, pero NO columnas vacías: son un
    # hallazgo de calidad que la fase 2 debe reportar, no algo a ocultar.
    df = df.dropna(axis=0, how="all").reset_index(drop=True)
    if any(has_mojibake(str(c)) for c in df.columns):
        warnings.append(
            "Los encabezados muestran caracteres corruptos (mojibake). "
            "Puedes repararlo en la fase de Preparación."
        )
    return df


# ---------------------------------------------------------------- API pública


def list_excel_sheets(raw: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(raw)).sheet_names


def load_bytes(
    raw: bytes,
    filename: str,
    sheet: str | None = None,
    header_row: int | None = None,
    separator: str | None = None,
    encoding: str | None = None,
) -> LoadResult:
    name = filename.lower()
    warnings: list[str] = []

    if name.endswith((".xlsx", ".xlsm", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(raw))
        sheets = xls.sheet_names
        sheet = sheet if sheet in sheets else sheets[0]
        if header_row is None:
            probe = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=12)
            header_row = _guess_header_row(probe)
            if header_row > 0:
                warnings.append(
                    f"Se detectó el encabezado en la fila {header_row + 1}; las filas previas se descartaron."
                )
        df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
        return LoadResult(
            df=_postprocess(df, warnings), source_name=filename, kind="excel",
            sheet=sheet, sheets=sheets, header_row=header_row, warnings=warnings,
        )

    text, enc = (raw.decode(encoding), encoding) if encoding else _decode(raw)
    sep = separator or _sniff_separator(text)
    if header_row is None:
        # nombres explícitos: si no, una primera línea de título con un solo campo
        # haría que read_csv descartara como "malas" todas las filas reales
        ancho = max((line.count(sep) + 1 for line in text.splitlines()[:40] if line.strip()),
                    default=1)
        probe = pd.read_csv(io.StringIO(text), sep=sep, header=None, nrows=12,
                            names=list(range(ancho)), engine="python", on_bad_lines="skip")
        header_row = _guess_header_row(probe)
        if header_row > 0:
            warnings.append(
                f"Se detectó el encabezado en la fila {header_row + 1}; las filas previas se descartaron."
            )
    df = pd.read_csv(
        io.StringIO(text), sep=sep, header=header_row,
        engine="python", on_bad_lines="warn", skip_blank_lines=True,
    )
    if enc.startswith("latin-1"):
        warnings.append(
            "El archivo no está en UTF-8; se leyó como latin-1. Revisa acentos y ñ."
        )
    return LoadResult(
        df=_postprocess(df, warnings), source_name=filename, kind="delimitado",
        encoding=enc, separator=sep, header_row=header_row, warnings=warnings,
    )


def load_uploaded(uploaded, **kwargs) -> LoadResult:
    """Acepta un UploadedFile de Streamlit."""
    return load_bytes(uploaded.getvalue(), uploaded.name, **kwargs)
