"""El tablero de gráficas: rejilla de tarjetas y vista a detalle.

Dos ideas:

1. Las gráficas se ven juntas, chicas, en una rejilla — no apiladas una debajo
   de otra. De un vistazo se ve todo el negocio.
2. Se le pica a cualquiera (a la gráfica misma o a su botón) y se abre en
   grande, con su lectura, sus valores raros, sus cifras y la tabla de datos
   que hay detrás.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import streamlit as st

import charts
import deployment
import estado
import explicaciones as expl
import insights as ins_mod
import kpis as kpi_mod
import profiling
from utils import fmt_num, to_datetime_series, to_numeric_series

MAX_VISTAS = 6
MAX_VISTAS_AUTO = 3        # cuando el usuario trae sus propios indicadores
MAX_CATEGORIAS_KPI = 40    # arriba de esto, agrupar por esa columna no dice nada
ALTO_CHICA = 215
ALTO_GRANDE = 430


TIPOS_GRAFICA = {
    "line": "Línea",
    "area": "Área",
    "bar": "Barras",
    "pie": "Pastel",
    "hist": "Histograma",
    "box": "Caja (box plot)",
    "scatter": "Dispersión",
    "mapa": "Mapa",
}

# columnas que se pueden pintar en un mapa sin traer un archivo de fronteras
RE_PAIS = ("pais", "país", "country", "nacion", "nación")


@dataclass
class Vista:
    id: str
    titulo: str
    tipo: str                                  # la recomendada para estos datos
    figura: Callable[[int], object]            # alto -> go.Figure
    resumen: str = ""                          # una línea para la tarjeta chica
    lectura: str = ""
    atipicos: list[str] = field(default_factory=list)
    cifras: list[tuple[str, str]] = field(default_factory=list)
    tabla: pd.DataFrame | None = None
    nota_tabla: str = ""
    # material para volver a dibujarla de otra forma en la vista a detalle
    datos: dict = field(default_factory=dict)


# ------------------------------------------- cambiar el tipo de gráfica


def tipos_posibles(datos: dict) -> dict[str, str]:
    """{tipo: ''} si se puede dibujar, o {tipo: motivo} si no.

    El motivo se le enseña al usuario tal cual, así que dice qué le falta a
    los datos y no «tipo no soportado».
    """
    serie = datos.get("serie")
    cats = datos.get("categorias")
    vals = datos.get("valores")
    hay_serie = serie is not None and len(serie) >= 3
    hay_cats = cats is not None and len(cats) >= 2

    sin_tiempo = ("necesita una columna de fecha para poner los puntos en orden. "
                  "En estos datos no la encontramos.")
    sin_cats = ("necesita una columna de categorías —canal, producto, forma de pago— "
                "para comparar entre ellas.")
    sin_crudos = ("necesita los valores uno por uno de una columna numérica; aquí solo "
                  "hay totales ya calculados por periodo.")

    fuera: dict[str, str] = {}
    fuera["line"] = "" if hay_serie else sin_tiempo
    fuera["area"] = "" if hay_serie else sin_tiempo
    fuera["bar"] = "" if (hay_cats or hay_serie) else sin_cats

    if not hay_cats:
        fuera["pie"] = sin_cats
    elif not datos.get("aditivo", True):
        fuera["pie"] = ("este indicador es un promedio o un porcentaje, y esos no se "
                        "reparten en rebanadas: las partes no suman el total.")
    elif float(cats.min()) < 0:
        fuera["pie"] = "hay valores negativos, y un pastel no puede representarlos."
    elif len(cats) > 12:
        fuera["pie"] = (f"hay {len(cats)} categorías; un pastel con más de 12 rebanadas "
                        "deja de leerse. Las barras sí lo aguantan.")
    else:
        fuera["pie"] = ""

    fuera["hist"] = "" if (vals is not None and len(vals) >= 20) else sin_crudos
    fuera["box"] = "" if datos.get("grupo") else (
        "necesita los valores uno por uno más una columna para agruparlos.")
    fuera["scatter"] = "" if datos.get("par") else (
        "necesita dos columnas numéricas que cruzar; en estos datos no hay una segunda.")
    fuera["mapa"] = "" if datos.get("geo") is not None else (
        "necesita una columna de países que podamos ubicar. Los estados y municipios "
        "de México requieren un archivo de fronteras que la app no trae.")
    return fuera


def figura_de_tipo(tipo: str, datos: dict, alto: int):
    """Dibuja los mismos datos con el tipo que pidió el usuario."""
    pref = datos.get("prefijo", "")
    etq = datos.get("etiqueta", "")
    try:
        if tipo in ("line", "area"):
            s = datos["serie"]
            d = pd.DataFrame({"x": s.index, "y": s.values})
            if tipo == "area":
                return charts.area_time(d, "x", "y", ylab=etq, height=alto)
            return charts.line_time(d, "x", "y", ylab=etq, height=alto)
        if tipo == "bar":
            cats = datos.get("categorias")
            if cats is not None and len(cats) >= 2:
                e, v, _ = charts.top_con_otros(cats.clip(lower=0), 10)
                return charts.bar_ranked(e, v, "", etq, height=alto, prefijo=pref)
            s = datos["serie"]
            return charts.bar_ranked(datos.get("serie_etq") or [str(i) for i in s.index],
                                     list(s.values), "", etq, horizontal=False,
                                     height=alto, prefijo=pref,
                                     etiquetas_valor=len(s) <= 14)
        if tipo == "pie":
            e, v, _ = charts.top_con_otros(datos["categorias"].clip(lower=0), 8)
            return charts.pastel(e, v, "", height=alto, prefijo=pref)
        if tipo == "hist":
            vals = datos["valores"]
            return charts.histogram(vals, "", etq, height=alto, median=float(vals.median()))
        if tipo == "box":
            d, dim, val = datos["grupo"]
            return charts.box_by_group(d, dim, val, "", height=alto)
        if tipo == "scatter":
            d, x, y = datos["par"]
            return charts.scatter(d, x, y, title="", height=alto)
        if tipo == "mapa":
            g = datos["geo"]
            return charts.mapa_paises(list(g.index), list(g.values), "", height=alto,
                                      prefijo=pref)
    except Exception:  # noqa: BLE001 - si algo no cuadra, se cae a la recomendada
        return None
    return None


# ------------------------------------------------------------------ armado


def _freq_y_etiqueta(span_dias: int) -> tuple[str, str]:
    if span_dias <= 60:
        return "D", "día"
    if span_dias <= 365:
        return "W", "semana"
    return "ME", "mes"


def _serie_por_periodo(df, fecha, valor_col=None):
    """(serie, freq, etiqueta) agregada por periodo, o None si no se puede."""
    f = to_datetime_series(df[fecha])
    mask = ins_mod.robust_date_mask(f)
    d = df[mask].assign(_f=f[mask])
    if len(d) <= 3:
        return None
    span = (d["_f"].max() - d["_f"].min()).days
    freq, etq = _freq_y_etiqueta(span)
    if valor_col is None:
        serie = d.set_index("_f").resample(freq).size()
    else:
        serie = (d.assign(_v=to_numeric_series(d[valor_col]))
                  .set_index("_f")["_v"].resample(freq).sum())
    # el último periodo casi siempre está incompleto y dibuja una caída falsa
    if len(serie) > 2 and serie.index[-1] > d["_f"].max():
        serie = serie.iloc[:-1]
    if len(serie) < 3:
        return None
    return serie, freq, etq


def _material(df, profiles, mapping, metrica: str | None, prefijo: str,
              aditivo: bool, etiqueta: str) -> dict:
    """Junta todo lo que se puede pintar con estos datos, no solo lo que se pinta.

    Así la vista a detalle puede ofrecer barras, pastel, histograma o caja sin
    volver a calcular nada.
    """
    dim_cols = profiling.suggest_dimension_columns(profiles)
    num_cols = profiling.suggest_metric_columns(profiles)
    datos: dict = {"prefijo": prefijo, "aditivo": aditivo, "etiqueta": etiqueta}

    dim = next((d for d in [mapping.get("segmento"), mapping.get("producto"),
                            mapping.get("cliente")] + list(dim_cols)
                if d in dim_cols and 1 < df[d].nunique() <= MAX_CATEGORIAS_KPI), None)

    if metrica:
        v = to_numeric_series(df[metrica]).dropna()
        if len(v) >= 20:
            datos["valores"] = v
        if dim:
            agg = (df.assign(_v=to_numeric_series(df[metrica])).groupby(dim, observed=True)["_v"]
                     .sum().sort_values(ascending=False).dropna())
            if len(agg) >= 2:
                datos["categorias"] = agg
            d = df[[dim, metrica]].copy()
            d[metrica] = to_numeric_series(d[metrica])
            d = d.dropna()
            if len(d) > 20:
                datos["grupo"] = (d, dim, metrica)
        otras = [c for c in num_cols if c != metrica]
        if otras:
            y2 = mapping.get("cantidad") if mapping.get("cantidad") in otras else otras[0]
            par = df[[metrica, y2]].apply(to_numeric_series).dropna()
            if len(par) > 20:
                datos["par"] = (par, y2, metrica)

        pais = next((c for c in dim_cols
                     if any(p in str(c).lower() for p in RE_PAIS)), None)
        if pais:
            geo = (df.assign(_v=to_numeric_series(df[metrica])).groupby(pais, observed=True)["_v"]
                     .sum().sort_values(ascending=False).dropna())
            if len(geo) >= 2:
                datos["geo"] = geo
    return datos


# ------------------------------------------- gráficas de los KPIs del usuario


def _fmt_valor(v: float, fmt: str, moneda: str) -> str:
    """Formatea igual que la tarjeta del indicador, para que los números cuadren."""
    return kpi_mod.KPIResult("", float(v), fmt).display(moneda)


@st.cache_data(show_spinner=False, max_entries=32)
def _serie_de_formula(df: pd.DataFrame, formula: str, fecha: str, freq: str) -> pd.Series:
    """Evalúa la fórmula del usuario periodo por periodo.

    Sirve para cualquier fórmula —promedio, margen, porcentaje— porque no
    interpreta la operación: la calcula sobre las filas de cada periodo, igual
    que la tarjeta la calcula sobre todas.
    """
    f = to_datetime_series(df[fecha])
    mask = ins_mod.robust_date_mask(f)
    # to_period no acepta los alias de resample: "ME" es de resample, "M" de periodo
    periodo = {"D": "D", "W": "W", "ME": "M"}.get(freq, "M")
    d = df[mask].assign(_p=f[mask].dt.to_period(periodo))
    valores: dict = {}
    for periodo, sub in d.groupby("_p", observed=True):
        try:
            v = float(kpi_mod.FormulaEvaluator(sub).evaluate(formula))
        except Exception:  # noqa: BLE001 - fórmula del usuario sobre pocas filas
            continue
        if pd.notna(v) and abs(v) != float("inf"):
            valores[periodo.to_timestamp()] = v
    serie = pd.Series(valores).sort_index()
    # el último periodo casi siempre está incompleto y dibuja una caída falsa
    if len(serie) > 3:
        serie = serie.iloc[:-1]
    return serie


@st.cache_data(show_spinner=False, max_entries=32)
def _formula_por_grupo(df: pd.DataFrame, formula: str, dim: str) -> pd.Series:
    valores: dict = {}
    for valor, sub in df.groupby(dim, observed=True):
        try:
            v = float(kpi_mod.FormulaEvaluator(sub).evaluate(formula))
        except Exception:  # noqa: BLE001
            continue
        if pd.notna(v) and abs(v) != float("inf"):
            valores[str(valor)] = v
    return pd.Series(valores).sort_values(ascending=False)


_NO_ADITIVAS = ("promedio(", "mediana(", "maximo(", "minimo(", "unicos(", "percentil(")


def _es_aditiva(formula: str) -> bool:
    """¿Las partes suman el total? Solo así tiene sentido un pastel.

    Sumas y conteos sí: el total por canal suma el total general. Un promedio
    o un porcentaje no: el promedio de los canales no es el promedio global.
    """
    f = formula.lower()
    return "/" not in f and not any(t in f for t in _NO_ADITIVAS)


def _atipicos_simples(serie: pd.Series, etq: str, fmt: str, moneda: str) -> list[str]:
    """Periodos que se salen de lo normal, con mediana y MAD (no promedio)."""
    if len(serie) < 6:
        return []
    med = float(serie.median())
    mad = float((serie - med).abs().median())
    if mad <= 0:
        return []
    fuera = serie[(0.6745 * (serie - med).abs() / mad) > 3.5]
    salida = []
    for ts, v in fuera.sort_values(ascending=False).head(3).items():
        veces = abs(v / med) if med else 0
        salida.append(
            f"**{expl._etiqueta_fecha(ts, 'ME')}**: {_fmt_valor(v, fmt, moneda)} contra "
            f"{_fmt_valor(med, fmt, moneda)} de un {etq} típico"
            + (f" — {veces:.1f} veces." if veces > 1.2 else "."))
    return salida


def vistas_de_kpis(df, profiles, mapping, moneda_simbolo: str, customs) -> list[Vista]:
    """Una gráfica por cada indicador que el usuario definió.

    Es lo que faltaba: el tablero armaba sus gráficas por su cuenta e ignoraba
    por completo los indicadores propios. Si alguien se tomó el trabajo de
    definir «Ticket Promedio Neto», eso es lo que quiere ver graficado.
    """
    vistas: list[Vista] = []
    date_cols = profiling.suggest_date_columns(profiles)
    dim_cols = profiling.suggest_dimension_columns(profiles)
    fecha = mapping.get("fecha") if mapping.get("fecha") in date_cols else (
        date_cols[0] if date_cols else None)

    dim = next((d for d in [mapping.get("segmento"), mapping.get("producto")] + list(dim_cols)
                if d in dim_cols and df[d].nunique() <= MAX_CATEGORIAS_KPI), None)

    freq = etq = None
    if fecha:
        f = to_datetime_series(df[fecha])
        d = df[ins_mod.robust_date_mask(f)].assign(_f=f[ins_mod.robust_date_mask(f)])
        if len(d) > 3:
            freq, etq = _freq_y_etiqueta((d["_f"].max() - d["_f"].min()).days)

    for n, k in enumerate(customs):
        try:
            total = float(kpi_mod.FormulaEvaluator(df).evaluate(k.formula))
        except Exception:  # noqa: BLE001
            continue
        moneda = moneda_simbolo if k.fmt == kpi_mod.FMT_MONEY else ""

        # material para poder redibujarla de otra forma en la vista a detalle
        col = next((c for c in re.findall(r'"([^"]+)"', k.formula) if c in df.columns), None)
        datos = _material(df, profiles, mapping, col, moneda, _es_aditiva(k.formula), k.name)
        if dim:
            # las categorías se calculan con LA FÓRMULA del usuario, no sumando
            # la columna: el promedio por canal no es la suma por canal
            agg_k = _formula_por_grupo(df, k.formula, dim)
            if len(agg_k) >= 2:
                datos["categorias"] = agg_k

        serie = _serie_de_formula(df, k.formula, fecha, freq) if freq else pd.Series(dtype=float)
        if len(serie) >= 3:
            mejor, peor = serie.idxmax(), serie.idxmin()
            vistas.append(Vista(
                id=f"kpi{n}", tipo="line", titulo=f"{k.name} por {etq}",
                figura=lambda h, s=serie, nom=k.name: charts.line_time(
                    pd.DataFrame({"x": s.index, "y": s.values}), "x", "y", ylab=nom, height=h),
                resumen=f"Ahora: {_fmt_valor(total, k.fmt, moneda_simbolo)} · "
                        f"mejor {etq}: {expl._etiqueta_fecha(mejor, freq)}",
                lectura=(f"**Cómo leerla:** cada punto es «{k.name}» calculado solo con los "
                         f"registros de ese {etq}, con la misma fórmula de la tarjeta. "
                         f"El mejor fue {expl._etiqueta_fecha(mejor, freq)} con "
                         f"{_fmt_valor(serie.max(), k.fmt, moneda_simbolo)} y el más flojo "
                         f"{expl._etiqueta_fecha(peor, freq)} con "
                         f"{_fmt_valor(serie.min(), k.fmt, moneda_simbolo)}."),
                atipicos=_atipicos_simples(serie, etq, k.fmt, moneda_simbolo),
                cifras=[("Con todos los datos", _fmt_valor(total, k.fmt, moneda_simbolo)),
                        (f"Promedio por {etq}",
                         _fmt_valor(float(serie.mean()), k.fmt, moneda_simbolo)),
                        (f"Mejor {etq}", expl._etiqueta_fecha(mejor, freq)),
                        (f"Peor {etq}", expl._etiqueta_fecha(peor, freq))],
                tabla=pd.DataFrame({
                    etq.capitalize(): [expl._etiqueta_fecha(t, freq) for t in serie.index],
                    k.name: serie.values}),
                nota_tabla=f"Se calcula «{k.help or k.formula}» dentro de cada {etq}.",
                datos={**datos, "serie": serie,
                       "serie_etq": [expl._etiqueta_fecha(t, freq) for t in serie.index]}))
            continue

        # sin fecha usable: se compara el indicador entre categorías
        if dim:
            agg = _formula_por_grupo(df, k.formula, dim)
            if len(agg) > 1:
                etiquetas, vals, _ = charts.top_con_otros(agg.clip(lower=0), 8)
                vistas.append(Vista(
                    id=f"kpi{n}", tipo="bar", titulo=f"{k.name} por {dim.lower()}",
                    figura=(lambda h, e=etiquetas, v=vals, nom=k.name, p=moneda:
                            charts.bar_ranked(e, v, "", nom, height=h, prefijo=p)),
                    resumen=f"Arriba: {agg.index[0]} "
                            f"({_fmt_valor(agg.iloc[0], k.fmt, moneda_simbolo)})",
                    lectura=(f"**Cómo leerla:** «{k.name}» calculado por separado para cada "
                             f"valor de «{dim}», con la misma fórmula de la tarjeta."),
                    cifras=[("Con todos los datos", _fmt_valor(total, k.fmt, moneda_simbolo)),
                            (f"{dim} distintos", str(len(agg))),
                            ("El más alto", str(agg.index[0])),
                            ("El más bajo", str(agg.index[-1]))],
                    tabla=pd.DataFrame({dim: agg.index.astype(str), k.name: agg.values}),
                    nota_tabla=f"Se calcula «{k.help or k.formula}» dentro de cada {dim}.",
                    datos={**datos, "categorias": agg}))
    return vistas


def construir_vistas(df, profiles, mapping, moneda_simbolo: str) -> list[Vista]:
    """Arma hasta seis gráficas con lo que los datos permitan."""
    num_cols = profiling.suggest_metric_columns(profiles)
    date_cols = profiling.suggest_date_columns(profiles)
    dim_cols = profiling.suggest_dimension_columns(profiles)
    vistas: list[Vista] = []
    if not num_cols:
        return vistas

    metrica = mapping.get("ingreso") if mapping.get("ingreso") in num_cols else num_cols[0]
    moneda = moneda_simbolo if mapping.get("ingreso") == metrica else None
    pre = moneda or ""
    fecha = mapping.get("fecha") if mapping.get("fecha") in date_cols else (
        date_cols[0] if date_cols else None)
    base = _material(df, profiles, mapping, metrica, pre, True, metrica)

    # -------------------------------------------------- 1. evolución en el tiempo
    if fecha:
        res = _serie_por_periodo(df, fecha, metrica)
        if res:
            serie, freq, etq = res
            tabla = pd.DataFrame({
                etq.capitalize(): [expl._etiqueta_fecha(t, freq) for t in serie.index],
                metrica: serie.values})
            mejor, peor = serie.idxmax(), serie.idxmin()
            vistas.append(Vista(
                id="tiempo", tipo="line",
                titulo=f"{metrica} por {etq}",
                figura=lambda h, s=serie: charts.line_time(
                    pd.DataFrame({"x": s.index, "y": s.values}), "x", "y",
                    ylab=metrica, height=h),
                resumen=f"Mejor {etq}: {expl._etiqueta_fecha(mejor, freq)} "
                        f"({pre}{fmt_num(serie.max())})",
                lectura=expl.leer_serie(serie, freq, etq, metrica, moneda),
                atipicos=expl.atipicos_serie(serie, df, fecha, metrica, mapping,
                                             freq, etq, moneda),
                cifras=[("Total del periodo", f"{pre}{fmt_num(serie.sum())}"),
                        (f"Promedio por {etq}", f"{pre}{fmt_num(serie.mean())}"),
                        (f"Mejor {etq}", expl._etiqueta_fecha(mejor, freq)),
                        (f"Peor {etq}", expl._etiqueta_fecha(peor, freq))],
                tabla=tabla,
                nota_tabla=f"El total de {metrica} en cada {etq}.",
                datos={**base, "serie": serie,
                       "serie_etq": [expl._etiqueta_fecha(t, freq) for t in serie.index]}))

        # -------------------------------------------- 2. volumen de operaciones
        res = _serie_por_periodo(df, fecha, None)
        if res:
            serie, freq, etq = res
            vistas.append(Vista(
                id="volumen", tipo="line",
                titulo=f"Cuántos registros por {etq}",
                figura=lambda h, s=serie: charts.line_time(
                    pd.DataFrame({"x": s.index, "y": s.values}), "x", "y",
                    ylab="Registros", height=h),
                resumen=f"Promedio: {fmt_num(serie.mean())} por {etq}",
                lectura=("Aquí no se mide dinero sino actividad: cuántos registros "
                         "entraron en cada " + etq + ". Comparada con la gráfica de "
                         "importe te dice si un mes bueno fue por vender más veces "
                         "o por vender más caro."),
                cifras=[("Registros en total", fmt_num(float(serie.sum()))),
                        (f"Promedio por {etq}", fmt_num(float(serie.mean()))),
                        (f"Mejor {etq}", expl._etiqueta_fecha(serie.idxmax(), freq)),
                        (f"{etq.capitalize()}s con datos", str(len(serie)))],
                tabla=pd.DataFrame({
                    etq.capitalize(): [expl._etiqueta_fecha(t, freq) for t in serie.index],
                    "Registros": serie.values}),
                datos={**base, "etiqueta": "Registros", "prefijo": "", "serie": serie,
                       "serie_etq": [expl._etiqueta_fecha(t, freq) for t in serie.index]}))

    # ------------------------------------------------------- 3 y 4. rankings
    preferidas = [mapping.get("segmento"), mapping.get("producto"), mapping.get("cliente")]
    # dict.fromkeys quita repetidos sin perder el orden: un mismo campo puede
    # estar mapeado a dos ranuras (segmento y producto), y entonces se armaban
    # dos gráficas idénticas con la misma llave de widget
    orden_dims = list(dict.fromkeys([d for d in preferidas if d in dim_cols]
                                    + list(dim_cols)))
    for dim in orden_dims[:2]:
        agg = (df.assign(_v=to_numeric_series(df[metrica])).groupby(dim)["_v"].sum()
                 .sort_values(ascending=False))
        if len(agg) < 2:
            continue
        etq_b, vals, n_resto = charts.top_con_otros(agg, 8)
        total = float(agg.sum()) or 1.0
        vistas.append(Vista(
            id=f"rank_{dim}", tipo="bar",
            titulo=f"{metrica} por {dim.lower()}",
            figura=(lambda h, e=etq_b, v=vals, m=metrica, p=pre:
                    charts.bar_ranked(e, v, "", m, height=h, prefijo=p)),
            resumen=f"Arriba: {agg.index[0]} ({agg.iloc[0] / total:.0%} del total)",
            lectura=expl.leer_ranking(agg, dim, metrica, moneda, n_resto),
            atipicos=expl.atipicos_ranking(agg, dim, moneda),
            cifras=[("Valores distintos", str(len(agg))),
                    ("El más grande", str(agg.index[0])),
                    ("Su parte del total", f"{agg.iloc[0] / total:.1%}"),
                    # con pocas categorías "los 5 primeros" serían el 100%: no dice nada
                    (("Los 5 primeros juntan", f"{agg.head(5).sum() / total:.1%}")
                     if len(agg) > 6 else ("El más chico", str(agg.index[-1])))],
            tabla=pd.DataFrame({dim: agg.index.astype(str), metrica: agg.values,
                                "% del total": (agg.values / total * 100).round(1)}),
            nota_tabla="La tabla trae todas las categorías, no solo las del top.",
            datos={**base, "categorias": agg}))

    # ------------------------------------------------------ 5. distribución
    s = to_numeric_series(df[metrica]).dropna()
    if len(s) > 20:
        mediana = float(s.median())
        # sin recortar la cola, un solo valor gigante mete todo lo demás en la
        # primera barra y el histograma no dice absolutamente nada
        lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
        s_vista = s[(s >= lo) & (s <= hi)] if hi > lo else s
        fuera = len(s) - len(s_vista)
        nota = (f" Se dejó fuera el {fuera / len(s):.1%} de los registros —los "
                f"valores más extremos— para que las barras se alcancen a ver."
                if fuera else "")
        vistas.append(Vista(
            id="dist", tipo="hist",
            titulo=f"Cómo se reparte {metrica.lower()}",
            figura=lambda h, v=s_vista, m=mediana, c=metrica: charts.histogram(
                v, "", c, height=h, median=m),
            resumen=f"La mitad está por debajo de {pre}{fmt_num(mediana)}",
            lectura=expl.como_leer("hist") + nota,
            cifras=[("Mediana (el caso típico)", f"{pre}{fmt_num(mediana)}"),
                    ("Promedio", f"{pre}{fmt_num(float(s.mean()))}"),
                    ("El 10% más alto pasa de", f"{pre}{fmt_num(float(s.quantile(0.9)))}"),
                    ("El más alto de todos", f"{pre}{fmt_num(float(s.max()))}")],
            nota_tabla=("Si el promedio y la mediana están muy separados, hay unos "
                        "pocos registros enormes jalando el promedio."),
            datos={**base, "valores": s_vista}))

    # ---------------------------------------------------- 6. caja por grupo
    if orden_dims:
        dim = orden_dims[0]
        d = df[[dim, metrica]].copy()
        d[metrica] = to_numeric_series(d[metrica])
        d = d.dropna()
        if len(d) > 20 and d[dim].nunique() > 1:
            med = d.groupby(dim)[metrica].median().sort_values(ascending=False)
            vistas.append(Vista(
                id="caja", tipo="box",
                titulo=f"{metrica} típico por {dim.lower()}",
                figura=lambda h, dd=d, g=dim, v=metrica: charts.box_by_group(
                    dd, g, v, "", height=h),
                resumen=f"El más alto en el caso típico: {med.index[0]}",
                lectura=expl.como_leer("box"),
                cifras=[("Caso típico más alto", str(med.index[0])),
                        ("Su valor típico", f"{pre}{fmt_num(float(med.iloc[0]))}"),
                        ("Caso típico más bajo", str(med.index[-1])),
                        ("Su valor típico", f"{pre}{fmt_num(float(med.iloc[-1]))}")],
                tabla=pd.DataFrame({dim: med.index.astype(str),
                                    f"{metrica} típico": med.values}),
                nota_tabla="«Típico» es la mediana: la mitad queda arriba y la mitad abajo.",
                datos={**base, "grupo": (d, dim, metrica), "aditivo": False}))

    # -------------------------------------------------------- 7. dispersión
    otras = [c for c in num_cols if c != metrica]
    if otras:
        y2 = mapping.get("cantidad") if mapping.get("cantidad") in otras else otras[0]
        par = df[[metrica, y2]].apply(to_numeric_series).dropna()
        if len(par) > 20:
            corr = float(par[metrica].corr(par[y2])) if par[y2].std() else 0.0
            junto = ("se mueven juntas" if corr > 0.5 else
                     "se mueven al revés" if corr < -0.5 else "no se mueven juntas")
            vistas.append(Vista(
                id="disp", tipo="scatter",
                titulo=f"{metrica} contra {y2.lower()}",
                figura=lambda h, p=par, a=metrica, b=y2: charts.scatter(
                    p, b, a, title="", height=h),
                resumen=f"Por lo que se ve, {junto}",
                lectura=expl.como_leer("scatter"),
                cifras=[("Registros graficados", fmt_num(float(len(par)))),
                        ("Qué tanto van juntas", f"{corr:+.2f} (de −1 a 1)"),
                        (f"{metrica} promedio", f"{pre}{fmt_num(float(par[metrica].mean()))}"),
                        (f"{y2} promedio", fmt_num(float(par[y2].mean())))],
                nota_tabla=("Que dos cosas se muevan juntas no quiere decir que una "
                            "cause la otra."),
                datos={**base, "par": (par, y2, metrica)}))

    return vistas[:MAX_VISTAS]


# ------------------------------------------------------------------ pintado


def _abrir(vid: str):
    st.session_state["_vista"] = vid
    st.session_state["_gen"] = st.session_state.get("_gen", 0) + 1


def _cerrar():
    st.session_state["_vista"] = None
    # llaves nuevas para los widgets: si no, el clic que abrió la gráfica sigue
    # registrado y la volvería a abrir sola en cuanto regresas
    st.session_state["_gen"] = st.session_state.get("_gen", 0) + 1


def _pintar(fig, key: str, clickable: bool):
    """Dibuja la gráfica. Devuelve True si el usuario le picó encima."""
    cfg = {"displayModeBar": False}
    if not clickable:
        st.plotly_chart(fig, use_container_width=True, key=key, config=cfg)
        return False
    try:
        ev = st.plotly_chart(fig, use_container_width=True, key=key, config=cfg,
                             on_select="rerun", selection_mode=("points",))
        pts = (ev or {}).get("selection", {}).get("points", [])
        return bool(pts)
    except TypeError:
        # versión de Streamlit sin selección en gráficas: queda el botón
        st.plotly_chart(fig, use_container_width=True, key=key + "_p", config=cfg)
        return False


def _tarjeta(v: Vista, alto: int):
    """Una gráfica en su tarjeta. Se abre al picarle encima o al botón."""
    gen = st.session_state.get("_gen", 0)
    with st.container(border=True):
        st.markdown(f"**{v.titulo}**")
        if _pintar(v.figura(alto), f"m_{v.id}_{gen}", clickable=True):
            _abrir(v.id)
            st.rerun()
        if v.resumen:
            st.markdown(f"<div class='lect'>{v.resumen}</div>", unsafe_allow_html=True)
        if st.button("🔍 Ver a detalle", key=f"b_{v.id}_{gen}",
                     use_container_width=True):
            _abrir(v.id)
            st.rerun()


def _panel_cifras(v: Vista):
    """El panel de al lado de la gráfica principal, con sus números."""
    tiles = "".join(f"<div class='tile'><span class='t'>{estado.esc(t)}</span>"
                    f"<span class='v'>{estado.esc(val)}</span></div>"
                    for t, val in v.cifras[:4])
    st.markdown(
        f"<div class='hero hero-col'><span class='et'>En números</span>"
        f"<div class='nom'>{estado.esc(v.titulo)}</div>"
        f"<div class='tiles tiles-col'>{tiles}</div></div>", unsafe_allow_html=True)


def _fila(vistas: list[Vista], por_fila: int, alto: int):
    for inicio in range(0, len(vistas), por_fila):
        cols = st.columns(por_fila, gap="medium")
        for col, v in zip(cols, vistas[inicio:inicio + por_fila]):
            with col:
                _tarjeta(v, alto)


def _rejilla(vistas: list[Vista], propias: int = 0):
    st.caption("Pícale a cualquier gráfica —o a su botón— para abrirla en grande "
               "con su explicación, sus cifras y los datos de atrás.")
    principal, resto = vistas[0], vistas[1:]

    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        _tarjeta(principal, 290)
    with c2:
        _panel_cifras(principal)

    if propias:
        # el tablero arranca con lo que el usuario definió, no con lo automático
        mias, automaticas = resto[:propias - 1], resto[propias - 1:]
        if mias:
            estado.sec("Tus indicadores", "Los que tú definiste, graficados")
            _fila(mias, 3, ALTO_CHICA)
        if automaticas:
            estado.sec("También", "Lo que encontramos por nuestra cuenta")
            _fila(automaticas, 3, ALTO_CHICA)
        return

    if resto[:3]:
        estado.sec("Desglose", "De dónde sale el número")
        _fila(resto[:3], 3, ALTO_CHICA)
    if resto[3:]:
        estado.sec("Señales", "Cómo se reparte y qué se mueve junto")
        _fila(resto[3:], 3, ALTO_CHICA)


def _detalle(v: Vista):
    negritas = estado.negritas
    gen = st.session_state.get("_gen", 0)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("← Volver al tablero", use_container_width=True):
            _cerrar()
            st.rerun()
    c2.markdown(f"### {v.titulo}")

    fig = None
    if v.datos:
        posibles = tipos_posibles(v.datos)
        opciones = [t for t in TIPOS_GRAFICA]
        elegido = st.selectbox(
            "¿Cómo la quieres ver?", opciones,
            index=opciones.index(v.tipo) if v.tipo in opciones else 0,
            format_func=lambda t: TIPOS_GRAFICA[t] + ("" if not posibles[t] else "  ·  no aplica"),
            key=f"tipo_{v.id}",
            help="Puedes cambiar el tipo de gráfica. Los marcados «no aplica» no se "
                 "pueden dibujar con estos datos y te decimos por qué.")
        motivo = posibles.get(elegido, "")
        if motivo:
            st.info(f"**{TIPOS_GRAFICA[elegido]}** no se puede con estos datos: {motivo} "
                    f"Te dejo **{TIPOS_GRAFICA[v.tipo].lower()}**, que es la que "
                    "recomendamos aquí.", icon="💡")
        else:
            fig = figura_de_tipo(elegido, v.datos, ALTO_GRANDE)
    if fig is None:
        fig = v.figura(ALTO_GRANDE)

    with st.container(border=True):
        _pintar(fig, f"g_{v.id}_{gen}", clickable=False)

    if v.cifras:
        cols = st.columns(len(v.cifras))
        for col, (etq, val) in zip(cols, v.cifras):
            with col, st.container(border=True):
                st.metric(etq, val)

    if v.lectura:
        st.markdown(f"<div class='lect'>{negritas(v.lectura)}</div>",
                    unsafe_allow_html=True)
    if v.atipicos:
        st.markdown("<div class='atip-t'>Valores fuera de lo normal</div>",
                    unsafe_allow_html=True)
        for a in v.atipicos:
            st.markdown(f"<div class='atip'>{negritas(a)}</div>", unsafe_allow_html=True)
        st.caption("Son pistas de dónde salió el número, no la causa. La causa la sabes "
                   "tú: una campaña, un cliente grande, un cierre de mes o un error "
                   "de captura.")

    if v.tabla is not None and len(v.tabla):
        st.markdown("**Los datos de atrás**")
        vista_tabla = v.tabla.copy()
        for c in vista_tabla.columns:            # sin esto salen 246792.10000001
            if pd.api.types.is_float_dtype(vista_tabla[c]):
                vista_tabla[c] = vista_tabla[c].round(2)
        st.dataframe(vista_tabla, use_container_width=True, hide_index=True, height=280)
        st.download_button("📋 Bajar esta tabla en CSV",
                           deployment.to_csv_bytes(v.tabla),
                           f"{v.id}.csv", "text/csv", key=f"d_{v.id}_{gen}")
    if v.nota_tabla:
        st.caption(v.nota_tabla)


def render(df, profiles, mapping, moneda):
    customs = (st.session_state.get("custom") or []
               if st.session_state.get("kpi_modo") == "propios" else [])
    propias: list[Vista] = []
    if customs:
        with st.spinner("Graficando tus indicadores…"):
            propias = vistas_de_kpis(df, profiles, mapping, moneda, customs)

    automaticas = construir_vistas(df, profiles, mapping, moneda)
    if propias:
        # las automáticas pasan a segundo plano, pero no se tiran: el usuario
        # definió qué le importa, no que lo demás deje de existir
        vistas = propias + automaticas[:MAX_VISTAS_AUTO]
    else:
        vistas = automaticas

    if not vistas:
        estado.sec("Tablero", "Cómo se ve tu operación")
        st.info("Con estas columnas no alcanza para dibujar gráficas útiles. "
                "Revisa en **Calidad de datos** si interpretamos bien tus columnas.")
        return

    abierta = st.session_state.get("_vista")
    actual = next((v for v in vistas if v.id == abierta), None)
    if actual is not None:
        estado.sec("A detalle", "Una gráfica a fondo")
        _detalle(actual)
        return
    estado.sec("Tablero", "Cómo se ve tu operación", f"{len(vistas)} gráficas")
    _rejilla(vistas, propias=len(propias))
