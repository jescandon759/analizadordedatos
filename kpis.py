"""Fase 1 — Business Understanding: KPIs de catálogo y KPIs definidos por el usuario.

El catálogo funciona por "roles de negocio": el usuario mapea qué columna es el
ingreso, cuál el costo, cuál el cliente, etc., y la app calcula solo los KPIs que
esas columnas permiten. Los KPIs propios se escriben como fórmula y se evalúan
con un intérprete restringido (nada de `eval` abierto).
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .utils import fmt_num, fmt_pct, safe_div, to_numeric_series

# --------------------------------------------------------------------------
# Slots de negocio
# --------------------------------------------------------------------------

SLOTS: dict[str, str] = {
    "fecha": "Fecha del evento (venta, registro, movimiento)",
    "ingreso": "Importe / venta / ingreso",
    "costo": "Costo asociado",
    "cantidad": "Unidades o piezas",
    "cliente": "Identificador de cliente",
    "producto": "Producto, servicio o SKU",
    "transaccion": "Identificador de la transacción / folio",
    "segmento": "Segmento, categoría, canal, sucursal o vendedor",
    "estatus": "Estatus del registro (pagado, cancelado, activo…)",
}

FMT_MONEY, FMT_NUM, FMT_PCT, FMT_INT = "moneda", "numero", "porcentaje", "entero"


@dataclass
class KPIResult:
    name: str
    value: float
    fmt: str = FMT_NUM
    help: str = ""
    delta: float | None = None          # variación vs periodo anterior (fracción)
    delta_label: str | None = None
    target: float | None = None
    source: str = "catálogo"

    def display(self, currency: str = "$") -> str:
        if self.value is None or (isinstance(self.value, float) and not np.isfinite(self.value)):
            return "—"
        if self.fmt == FMT_MONEY:
            return f"{currency}{fmt_num(self.value)}"
        if self.fmt == FMT_PCT:
            return fmt_pct(self.value)
        if self.fmt == FMT_INT:
            return f"{int(round(self.value)):,}"
        return fmt_num(self.value)

    @property
    def status(self) -> str | None:
        if self.target is None or self.value is None:
            return None
        r = safe_div(self.value, self.target)
        if not np.isfinite(r):
            return None
        return "ok" if r >= 1 else ("cerca" if r >= 0.9 else "bajo")


# --------------------------------------------------------------------------
# Catálogo
# --------------------------------------------------------------------------


def _num(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    if not col or col not in df.columns:
        return None
    s = to_numeric_series(df[col]).dropna()
    return s if len(s) else None


def compute_catalog(df: pd.DataFrame, mapping: dict[str, str | None],
                    selected: list[str] | None = None) -> list[KPIResult]:
    """`mapping` va de slot -> nombre de columna (o None)."""
    out: list[KPIResult] = []
    m = {k: (v if v else None) for k, v in mapping.items()}

    ingreso = _num(df, m.get("ingreso"))
    costo = _num(df, m.get("costo"))
    cantidad = _num(df, m.get("cantidad"))
    n_filas = len(df)

    if m.get("transaccion") and m["transaccion"] in df.columns:
        n_tx = int(df[m["transaccion"]].nunique())
        tx_help = f"Valores únicos de '{m['transaccion']}'."
    else:
        n_tx = n_filas
        tx_help = "Una fila = una transacción (no se mapeó folio)."

    add = out.append

    add(KPIResult("Registros", float(n_filas), FMT_INT, "Filas del conjunto de datos analizado."))

    if ingreso is not None:
        total = float(ingreso.sum())
        add(KPIResult("Ingreso total", total, FMT_MONEY, f"Suma de '{m['ingreso']}'."))
        add(KPIResult("Transacciones", float(n_tx), FMT_INT, tx_help))
        add(KPIResult("Ticket promedio", safe_div(total, n_tx), FMT_MONEY,
                      "Ingreso total ÷ transacciones."))
        add(KPIResult("Ticket mediano", float(ingreso.median()), FMT_MONEY,
                      "Más representativo que el promedio cuando hay ventas atípicas."))
        add(KPIResult("Venta máxima", float(ingreso.max()), FMT_MONEY, "Mayor importe registrado."))

    if cantidad is not None:
        add(KPIResult("Unidades", float(cantidad.sum()), FMT_INT, f"Suma de '{m['cantidad']}'."))
        if ingreso is not None and cantidad.sum() != 0:
            add(KPIResult("Precio promedio por unidad",
                          safe_div(ingreso.sum(), cantidad.sum()), FMT_MONEY,
                          "Ingreso total ÷ unidades totales."))

    if costo is not None:
        add(KPIResult("Costo total", float(costo.sum()), FMT_MONEY, f"Suma de '{m['costo']}'."))
        if ingreso is not None:
            util = float(ingreso.sum() - costo.sum())
            add(KPIResult("Utilidad bruta", util, FMT_MONEY, "Ingreso total − costo total."))
            add(KPIResult("Margen bruto", safe_div(util, ingreso.sum()), FMT_PCT,
                          "Utilidad bruta ÷ ingreso total."))

    cli = m.get("cliente")
    if cli and cli in df.columns:
        n_cli = int(df[cli].nunique())
        add(KPIResult("Clientes únicos", float(n_cli), FMT_INT, f"Valores distintos de '{cli}'."))
        if ingreso is not None and n_cli:
            add(KPIResult("Ingreso por cliente", safe_div(ingreso.sum(), n_cli), FMT_MONEY,
                          "Ingreso total ÷ clientes únicos."))
        if n_cli:
            add(KPIResult("Frecuencia de compra", safe_div(n_tx, n_cli), FMT_NUM,
                          "Transacciones ÷ clientes únicos."))
        if ingreso is not None and n_cli >= 5 and m.get("ingreso"):
            por_cli = df.groupby(cli)[m["ingreso"]].apply(lambda x: to_numeric_series(x).sum())
            por_cli = por_cli.sort_values(ascending=False)
            k = max(1, int(round(len(por_cli) * 0.1)))
            add(KPIResult("Concentración top 10% clientes",
                          safe_div(por_cli.head(k).sum(), por_cli.sum()), FMT_PCT,
                          f"Porcentaje del ingreso que aportan los {k} clientes más grandes. "
                          "Arriba de 50% es riesgo de dependencia."))

    prod = m.get("producto")
    if prod and prod in df.columns:
        add(KPIResult("Productos distintos", float(df[prod].nunique()), FMT_INT,
                      f"Valores distintos de '{prod}'."))

    est = m.get("estatus")
    if est and est in df.columns and df[est].nunique() <= 12:
        vc = df[est].astype(str).value_counts(normalize=True)
        neg = [v for v in vc.index if any(
            t in v.lower() for t in ("cancel", "rechaz", "devuel", "baja", "perdid", "no ", "fail")
        )]
        if neg:
            add(KPIResult(f"Tasa de {neg[0]}", float(vc[neg].sum()), FMT_PCT,
                          f"Proporción de registros con estatus negativo en '{est}'."))

    fecha = m.get("fecha")
    if fecha and fecha in df.columns:
        f = pd.to_datetime(df[fecha], errors="coerce").dropna()
        if len(f) > 1:
            dias = max((f.max() - f.min()).days, 1)
            add(KPIResult("Cobertura (días)", float(dias), FMT_INT,
                          f"De {f.min():%Y-%m-%d} a {f.max():%Y-%m-%d}."))
            if ingreso is not None:
                add(KPIResult("Ingreso promedio diario", safe_div(ingreso.sum(), dias), FMT_MONEY,
                              "Ingreso total ÷ días cubiertos."))

    if selected is not None:
        out = [k for k in out if k.name in selected]
    return out


def add_period_deltas(kpis: list[KPIResult], df: pd.DataFrame,
                      mapping: dict[str, str | None], freq: str = "ME") -> list[KPIResult]:
    """Añade variación del último periodo cerrado contra el anterior."""
    fecha, ingreso_col = mapping.get("fecha"), mapping.get("ingreso")
    if not fecha or fecha not in df.columns:
        return kpis
    f = pd.to_datetime(df[fecha], errors="coerce")
    tmp = df.assign(_f=f).dropna(subset=["_f"])
    if len(tmp) < 4:
        return kpis
    grp = tmp.set_index("_f").resample(freq)
    label = {"D": "día", "W": "semana", "ME": "mes", "QE": "trimestre", "YE": "año"}.get(freq, "periodo")

    series: dict[str, pd.Series] = {"Registros": grp.size()}
    if ingreso_col and ingreso_col in df.columns:
        series["Ingreso total"] = grp[ingreso_col].apply(lambda x: to_numeric_series(x).sum())

    # descarta el último periodo si está incompleto: comparar un mes a medias
    # contra uno completo produce caídas ficticias
    fin_datos = tmp["_f"].max()
    for name, s in list(series.items()):
        if len(s) and s.index[-1] > fin_datos:
            series[name] = s.iloc[:-1]

    for k in kpis:
        s = series.get(k.name)
        if s is None or len(s) < 2:
            continue
        prev, last = float(s.iloc[-2]), float(s.iloc[-1])
        k.delta = safe_div(last - prev, abs(prev)) if prev else None
        k.delta_label = f"vs. {label} anterior"
    return kpis


# --------------------------------------------------------------------------
# KPIs personalizados: intérprete restringido
# --------------------------------------------------------------------------

FUNCTIONS: dict[str, str] = {
    "suma": "suma(columna) — total de la columna",
    "promedio": "promedio(columna) — media aritmética",
    "mediana": "mediana(columna)",
    "minimo": "minimo(columna)",
    "maximo": "maximo(columna)",
    "conteo": "conteo() o conteo(columna) — filas / valores no vacíos",
    "unicos": "unicos(columna) — valores distintos",
    "desviacion": "desviacion(columna) — desviación estándar",
    "percentil": "percentil(columna, 90) — percentil indicado",
    "suma_si": "suma_si(columna_a_sumar, columna_filtro, \"valor\") — suma condicionada",
    "conteo_si": "conteo_si(columna_filtro, \"valor\") — cuenta filas que cumplen",
    "raiz": "raiz(x)",
    "abs": "abs(x)",
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Tuple,
)


class FormulaError(ValueError):
    pass


class FormulaEvaluator:
    """Evalúa fórmulas del usuario sobre un DataFrame, sin ejecutar código arbitrario."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._alias = {c.lower(): c for c in df.columns}

    # -- resolución de columnas -------------------------------------------
    def _col(self, name) -> pd.Series:
        if isinstance(name, pd.Series):
            return name
        key = str(name)
        if key in self.df.columns:
            return self.df[key]
        if key.lower() in self._alias:
            return self.df[self._alias[key.lower()]]
        raise FormulaError(f"La columna '{key}' no existe en los datos.")

    def _numcol(self, name) -> pd.Series:
        return to_numeric_series(self._col(name)).dropna()

    # -- funciones ---------------------------------------------------------
    def _fn(self, fname: str, args: list):
        f = fname.lower()
        if f in ("suma", "sum"):
            return float(self._numcol(args[0]).sum())
        if f in ("promedio", "media", "mean", "avg"):
            return float(self._numcol(args[0]).mean())
        if f in ("mediana", "median"):
            return float(self._numcol(args[0]).median())
        if f in ("minimo", "min"):
            return float(self._numcol(args[0]).min())
        if f in ("maximo", "max"):
            return float(self._numcol(args[0]).max())
        if f in ("conteo", "count"):
            return float(len(self.df)) if not args else float(self._col(args[0]).notna().sum())
        if f in ("unicos", "nunique", "distintos"):
            return float(self._col(args[0]).nunique())
        if f in ("desviacion", "std", "desvest"):
            return float(self._numcol(args[0]).std(ddof=1))
        if f in ("percentil", "percentile"):
            q = float(args[1]) / 100.0 if float(args[1]) > 1 else float(args[1])
            return float(self._numcol(args[0]).quantile(q))
        if f in ("suma_si", "sumif"):
            col, filt, val = args[0], self._col(args[1]), args[2]
            mask = filt.astype(str).str.lower() == str(val).lower()
            return float(to_numeric_series(self._col(col))[mask].sum())
        if f in ("conteo_si", "countif"):
            filt, val = self._col(args[0]), args[1]
            return float((filt.astype(str).str.lower() == str(val).lower()).sum())
        if f in ("raiz", "sqrt"):
            return float(math.sqrt(float(args[0])))
        if f == "abs":
            return float(abs(float(args[0])))
        raise FormulaError(f"Función desconocida: '{fname}'. Disponibles: {', '.join(FUNCTIONS)}.")

    # -- recorrido del AST -------------------------------------------------
    def _eval(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            raise FormulaError("La fórmula contiene una expresión no permitida.")
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return node.id                       # nombre de columna sin comillas
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            return -float(v) if isinstance(node.op, ast.USub) else float(v)
        if isinstance(node, ast.BinOp):
            a, b = float(self._eval(node.left)), float(self._eval(node.right))
            op = node.op
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                if b == 0:
                    raise FormulaError("División entre cero.")
                return a / b
            if isinstance(op, ast.Pow):
                return a ** b
            if isinstance(op, ast.Mod):
                return a % b
            raise FormulaError("Operador no permitido.")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise FormulaError("Llamada de función no válida.")
            args = [self._eval(a) for a in node.args]
            return self._fn(node.func.id, args)
        raise FormulaError("Expresión no soportada.")

    def evaluate(self, formula: str) -> float:
        formula = (formula or "").strip()
        if not formula:
            raise FormulaError("La fórmula está vacía.")
        # permite escribir columnas con espacios entre comillas o corchetes
        formula = formula.replace("[", '"').replace("]", '"')
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError as e:
            raise FormulaError(f"Error de sintaxis: {e.msg}") from e
        value = self._eval(tree)
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            raise FormulaError("La fórmula no produjo un número.") from e


@dataclass
class CustomKPI:
    name: str
    formula: str
    fmt: str = FMT_NUM
    target: float | None = None
    help: str = ""


def compute_custom(df: pd.DataFrame, kpis: list[CustomKPI]) -> tuple[list[KPIResult], list[str]]:
    ev = FormulaEvaluator(df)
    results, errors = [], []
    for k in kpis:
        try:
            v = ev.evaluate(k.formula)
            results.append(KPIResult(k.name, v, k.fmt, k.help or f"Fórmula: {k.formula}",
                                     target=k.target, source="personalizado"))
        except FormulaError as e:
            errors.append(f"**{k.name}**: {e}")
        except Exception as e:  # noqa: BLE001 - fórmula del usuario
            errors.append(f"**{k.name}**: no se pudo calcular ({e}).")
    return results, errors


_PRIORIDAD_INGRESO = ("importe", "total", "venta", "ingreso", "monto", "revenue", "amount", "sales")
_PRIORIDAD_COSTO = ("costo", "coste", "cost", "gasto", "compra")


def _rank(nombre: str, claves: tuple[str, ...]) -> int:
    n = nombre.lower()
    for i, k in enumerate(claves):
        if k in n:
            return i
    return len(claves)


def suggest_mapping(profiles) -> dict[str, str | None]:
    """Propone un mapeo inicial de slots leyendo los roles y nombres detectados."""
    by_role: dict[str, list[str]] = {}
    for p in profiles.values():
        if p.role:
            by_role.setdefault(p.role, []).append(p.name)

    def first(role, semantics=None):
        for c in by_role.get(role, []):
            if semantics is None or profiles[c].semantic in semantics:
                return c
        return None

    monetarias = [c for c in by_role.get("monetario", []) if profiles[c].semantic == "numerico"]
    costo = min(monetarias, key=lambda c: _rank(c, _PRIORIDAD_COSTO), default=None)
    if costo is not None and _rank(costo, _PRIORIDAD_COSTO) == len(_PRIORIDAD_COSTO):
        costo = None
    ingreso_cands = [c for c in monetarias if c != costo]
    ingreso = min(ingreso_cands, key=lambda c: _rank(c, _PRIORIDAD_INGRESO), default=None)

    cliente = next((c for c in by_role.get("persona", []) + by_role.get("identificador", [])
                    if any(t in c.lower() for t in ("client", "customer", "comprador"))), None)
    producto = next((p.name for p in profiles.values()
                     if any(t in p.name.lower()
                            for t in ("producto", "product", "sku", "articulo", "item"))), None)
    estatus = next((p.name for p in profiles.values()
                    if p.semantic in ("categorico", "booleano") and p.n_unique <= 10
                    and p.role != "geografia"
                    and any(t in p.name.lower() for t in ("estatus", "status", "estado_"))), None)
    if estatus is None:
        estatus = next((p.name for p in profiles.values()
                        if p.semantic in ("categorico", "booleano") and p.n_unique <= 10
                        and p.role != "geografia" and p.name.lower() in ("estado", "estatus", "status")
                        and any(t in str(p.stats.get("moda", "")).lower()
                                for t in ("pagad", "cancel", "activ", "pendien", "cerrad"))), None)

    usados = {cliente, producto, estatus}
    segmento = next((p.name for p in profiles.values()
                     if p.semantic == "categorico" and 1 < p.n_unique <= 30
                     and p.name not in usados), None)
    return {
        "fecha": first("fecha", {"fecha"}),
        "ingreso": ingreso,
        "costo": costo,
        "cantidad": first("cantidad", {"numerico"}),
        "cliente": cliente,
        "producto": producto,
        "transaccion": first("identificador"),
        "segmento": segmento,
        "estatus": estatus,
    }
