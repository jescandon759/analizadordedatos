"""Lectura en palabras de cada gráfica, y atribución de los picos y caídas.

Dos cosas distintas, y conviene no confundirlas:

* **Qué significa la gráfica** — una descripción de lo que se ve, con los números
  reales: el mejor periodo, el peor, el promedio, hacia dónde va.
* **De dónde salió un valor atípico** — la app NO puede saber la causa de un pico.
  Lo que sí puede hacer, y es lo que hace aquí, es descomponerlo: si vino de una
  sola operación grande, de una categoría que ese día pesó el triple de lo normal,
  o de que simplemente hubo más operaciones. Eso son pistas verificables; la causa
  la pone quien conoce el negocio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import etiqueta, fmt_num, fmt_pct, to_datetime_series, to_numeric_series

DIAS = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
MAX_ATIPICOS = 3


def _money(v, moneda: str | None) -> str:
    return f"{moneda}{fmt_num(v)}" if moneda else fmt_num(v)


def _etiqueta_fecha(ts: pd.Timestamp, freq: str) -> str:
    if freq == "D":
        return f"{ts.day} de {MESES[ts.month - 1]} de {ts.year} ({DIAS[ts.dayofweek]})"
    if freq == "W":
        return f"la semana del {ts.day} de {MESES[ts.month - 1]} de {ts.year}"
    if freq in ("QE", "Q"):
        return f"el trimestre {ts.quarter} de {ts.year}"
    return f"{MESES[ts.month - 1]} de {ts.year}"


# ------------------------------------------------------------------ series


def leer_serie(serie: pd.Series, freq: str, etq: str, metrica: str,
               moneda: str | None = None) -> str:
    """Qué significa una gráfica de evolución en el tiempo."""
    s = serie.dropna()
    if len(s) < 2:
        return ""
    imax, imin = s.idxmax(), s.idxmin()
    prom = float(s.mean())

    # dirección: compara el primer tercio contra el último
    k = max(1, len(s) // 3)
    ini, fin = float(s.iloc[:k].mean()), float(s.iloc[-k:].mean())
    cambio = (fin - ini) / abs(ini) if ini else np.nan
    if not np.isfinite(cambio) or abs(cambio) < 0.08:
        rumbo = "En conjunto se mantiene parejo, sin subir ni bajar de forma clara."
    else:
        rumbo = (f"En conjunto va {'hacia arriba' if cambio > 0 else 'hacia abajo'}: "
                 f"el último tercio del periodo promedia {fmt_pct(abs(cambio))} "
                 f"{'más' if cambio > 0 else 'menos'} que el primero.")

    return (f"**Cómo leerla:** cada punto es el total de {metrica} de un {etq}. "
            f"Tu mejor {etq} fue {_etiqueta_fecha(imax, freq)} con {_money(s.max(), moneda)}, "
            f"y el más flojo {_etiqueta_fecha(imin, freq)} con {_money(s.min(), moneda)}. "
            f"El {etq} promedio son {_money(prom, moneda)}. {rumbo}")


def _pistas_periodo(sub: pd.DataFrame, df: pd.DataFrame, valor_col: str | None,
                    mapping: dict, total: float, etq: str,
                    conteo_habitual: float, moneda: str | None) -> list[str]:
    """De dónde salió el valor de ese periodo. Hechos, no causas."""
    pistas: list[str] = []
    if valor_col and valor_col in sub.columns:
        v = to_numeric_series(sub[valor_col]).dropna()
    else:
        v = pd.Series(dtype=float)

    # ¿fueron más operaciones o operaciones más grandes?
    if conteo_habitual > 0:
        ratio = len(sub) / conteo_habitual
        if ratio >= 1.4:
            pistas.append(f"Hubo **{len(sub):,} operaciones**, contra las "
                          f"{conteo_habitual:,.0f} de un {etq} normal: fue volumen, "
                          "no operaciones más grandes.")
        elif ratio <= 0.7 and len(v):
            pistas.append(f"Solo hubo **{len(sub):,} operaciones** (lo normal son "
                          f"{conteo_habitual:,.0f}), pero de mayor tamaño: el promedio por "
                          f"operación fue {_money(v.mean(), moneda)}.")

    # ¿una sola operación explica el periodo?
    if len(v) and total:
        parte = float(v.max()) / total
        if parte >= 0.2:
            fila = sub.loc[v.idxmax()]
            detalles = []
            for slot in ("cliente", "producto", "segmento"):
                col = mapping.get(slot)
                if col and col in sub.columns and pd.notna(fila.get(col)):
                    detalles.append(etiqueta(fila[col]))
            quien = f" ({', '.join(detalles[:2])})" if detalles else ""
            pistas.append(f"**Una sola operación de {_money(v.max(), moneda)}**{quien} "
                          f"explica el {fmt_pct(parte)} de todo el {etq}. "
                          "Vale la pena confirmar que la cifra sea correcta.")

    # contexto de calendario: ¿es una temporada que siempre pega fuerte?
    if "_f" in sub.columns and len(sub) and "_f" in df.columns:
        ts = sub["_f"].iloc[0]
        serie_val = (to_numeric_series(df[valor_col])
                     if valor_col and valor_col in df.columns
                     else pd.Series(1.0, index=df.index))
        if etq == "mes" and (df["_f"].max() - df["_f"].min()).days >= 500:
            por_mes = serie_val.groupby(df["_f"].dt.month).sum()
            if len(por_mes) >= 6 and por_mes.sum() > 0:
                cuota = por_mes / por_mes.sum()
                if int(cuota.idxmax()) == ts.month and cuota.max() > 1.4 / 12:
                    pistas.append(f"Es **{MESES[ts.month - 1]}**, que año con año es tu mes más "
                                  f"fuerte ({fmt_pct(cuota.max())} del total anual). "
                                  "Probablemente sea temporada, no algo excepcional.")
        elif etq == "día":
            por_dia = serie_val.groupby(df["_f"].dt.dayofweek).sum()
            if len(por_dia) >= 5 and por_dia.sum() > 0:
                cuota = por_dia / por_dia.sum()
                if int(cuota.idxmax()) == ts.dayofweek and cuota.max() > 1.4 / 7:
                    pistas.append(f"Cae en **{DIAS[ts.dayofweek]}**, tu día más fuerte de la "
                                  f"semana ({fmt_pct(cuota.max())} del total).")

    # ¿se concentró en alguna categoría fuera de lo habitual?
    for slot in ("segmento", "producto", "cliente"):
        col = mapping.get(slot)
        if not col or col not in sub.columns or sub[col].nunique() < 2:
            continue
        if valor_col and valor_col in sub.columns:
            en_periodo = sub.assign(_v=to_numeric_series(sub[valor_col])).groupby(col)["_v"].sum()
            global_ = df.assign(_v=to_numeric_series(df[valor_col])).groupby(col)["_v"].sum()
        else:
            en_periodo, global_ = sub[col].value_counts(), df[col].value_counts()
        if en_periodo.sum() <= 0 or global_.sum() <= 0:
            continue
        sp, sg = en_periodo / en_periodo.sum(), global_ / global_.sum()
        cat = sp.idxmax()
        if sp[cat] >= 0.4 and sg.get(cat, 0) > 0 and sp[cat] >= sg[cat] * 1.5:
            pistas.append(f"Se concentró en **{etiqueta(cat)}**: {fmt_pct(sp[cat])} del {etq}, "
                          f"cuando normalmente representa {fmt_pct(sg[cat])}.")
            break

    return pistas


def atipicos_serie(serie: pd.Series, df: pd.DataFrame, fecha_col: str,
                   valor_col: str | None, mapping: dict, freq: str, etq: str,
                   moneda: str | None = None) -> list[str]:
    """Periodos que se salen de lo normal, con la descomposición de cada uno."""
    s = serie.dropna()
    if len(s) < 6:
        return []
    # mediana y MAD: no se dejan arrastrar por el propio pico que buscamos
    med = float(s.median())
    mad = float(np.median(np.abs(s.values - med)))
    if mad <= 0:
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        if q3 - q1 <= 0:
            return []
        z = (s - med) / (q3 - q1)
        umbral = 2.5
    else:
        z = 0.6745 * (s - med) / mad
        umbral = 3.5

    raros = z[abs(z) >= umbral].sort_values(key=abs, ascending=False)
    if raros.empty:
        return []

    f = to_datetime_series(df[fecha_col])
    d = df.assign(_f=f).dropna(subset=["_f"])
    periodos = d["_f"].dt.to_period({"D": "D", "W": "W", "ME": "M", "QE": "Q"}.get(freq, "M"))
    conteo_habitual = float(periodos.value_counts().median())

    salida = []
    for ts in list(raros.index)[:MAX_ATIPICOS]:
        valor = float(s.loc[ts])
        alto = valor > med
        veces = valor / med if med else np.nan
        cabecera = (f"**{_etiqueta_fecha(ts, freq).capitalize()}** se salió de lo normal: "
                    f"{_money(valor, moneda)} contra los {_money(med, moneda)} de un {etq} "
                    f"típico"
                    + (f" — {veces:.1f} veces más." if alto and np.isfinite(veces)
                       else f" — {fmt_pct(1 - veces)} menos." if np.isfinite(veces) else "."))

        per = ts.to_period({"D": "D", "W": "W", "ME": "M", "QE": "Q"}.get(freq, "M"))
        sub = d[periodos == per]
        pistas = _pistas_periodo(sub, d, valor_col, mapping, valor, etq,
                                 conteo_habitual, moneda) if len(sub) else []
        if not alto and not pistas:
            pistas.append("No encontramos una concentración que lo explique: revisa si a ese "
                          f"{etq} simplemente le faltan registros por capturar.")
        salida.append(cabecera + ("<br>" + "<br>".join("· " + p for p in pistas)
                                  if pistas else ""))
    return salida


# ------------------------------------------------------------------ ranking


def leer_ranking(agg: pd.Series, dim: str, metrica: str,
                 moneda: str | None = None, n_resto: int = 0) -> str:
    a = agg.dropna().sort_values(ascending=False)
    if len(a) < 2:
        return ""
    total = float(a.sum())
    if total <= 0:
        return ""
    primero = etiqueta(a.index[0])
    s1 = float(a.iloc[0]) / total
    dos = float(a.iloc[:2].sum()) / total
    cola = (f" La última barra, **Otros**, junta los {n_resto:,} valores restantes de "
            f"«{dim}», que entre todos son {fmt_pct(1 - float(a.iloc[:7].sum()) / total)} "
            "del total." if n_resto else "")
    return (f"**Cómo leerla:** cada barra es el total de {metrica} de un valor de "
            f"«{dim}», de mayor a menor, con su cifra escrita al lado. "
            f"**{primero}** encabeza con {_money(a.iloc[0], moneda)}, "
            f"el {fmt_pct(s1)} del total. Los dos primeros juntos son "
            f"{fmt_pct(dos)}.{cola}")


def atipicos_ranking(agg: pd.Series, dim: str, moneda: str | None = None) -> list[str]:
    a = agg.dropna()
    if len(a) < 4:
        return []
    resto = a.iloc[1:]
    med = float(resto.median())
    if med <= 0:
        return []
    if float(a.iloc[0]) >= med * 3:
        return [f"**{etiqueta(a.index[0])}** está muy por encima del resto: "
                f"{float(a.iloc[0]) / med:.1f} veces la barra típica "
                f"({_money(med, moneda)}). Si es correcto, ahí está tu concentración; "
                "si no lo esperabas, revisa que no se estén agrupando dos cosas distintas "
                f"bajo el mismo valor de «{dim}»."]
    return []


# ------------------------------------------------------------------ otras formas

COMO_LEER = {
    "hist": ("Cada barra cuenta cuántos registros caen en ese rango. Mientras más alta, "
             "más común es ese valor. La línea punteada marca la mediana: la mitad de tus "
             "registros está a cada lado."),
    "box": ("Cada caja resume una categoría: la línea de en medio es el valor típico y la "
            "caja abarca a la mitad central de los registros. Cajas a distinta altura "
            "significan que esa categoría se comporta distinto. La vista está recortada "
            "para que las cajas se vean: puede haber valores sueltos más allá del borde."),
    "scatter": ("Cada punto es un registro. Si los puntos siguen la línea punteada, las dos "
                "variables se mueven juntas. Que vayan juntas no significa que una cause "
                "la otra."),
    "bar": ("Cada barra es el total del grupo. Compara las alturas para ver dónde se "
            "concentra y dónde casi no hay nada."),
    "line": ("Cada punto es el total de un periodo. Fíjate en la dirección general, no en "
             "el sube y baja de un punto al siguiente."),
}


def como_leer(tipo: str) -> str:
    return COMO_LEER.get(tipo, "")
