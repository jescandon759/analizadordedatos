"""Motor de insights por reglas.

No hay generación de texto libre: cada hallazgo nace de una prueba estadística o
de un umbral explícito, y el texto se arma con los números reales. Así el mismo
archivo produce siempre el mismo diagnóstico y todo hallazgo es verificable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from utils import (etiqueta, fmt_num, fmt_pct, safe_div, to_datetime_series,
                   to_numeric_series)

OPORTUNIDAD, RIESGO, HALLAZGO, CONTEXTO = "oportunidad", "riesgo", "hallazgo", "contexto"
ICON = {OPORTUNIDAD: "▲", RIESGO: "▼", HALLAZGO: "◆", CONTEXTO: "●"}


@dataclass
class Insight:
    kind: str
    title: str
    text: str                    # versión técnica (modo avanzado)
    impact: float = 0.5          # 0-1, ordena la lista
    evidence: dict = field(default_factory=dict)
    chart: dict | None = None    # {"type": ..., "data": ...} para dibujar en la app
    simple: str = ""             # misma idea sin jerga (modo sencillo)
    titulo_simple: str = ""

    @property
    def texto_llano(self) -> str:
        return self.simple or self.text

    @property
    def titulo_llano(self) -> str:
        return self.titulo_simple or self.title


def _period_freq(span_days: int) -> tuple[str, str]:
    if span_days <= 45:
        return "D", "día"
    if span_days <= 200:
        return "W", "semana"
    if span_days <= 1200:
        return "ME", "mes"
    return "QE", "trimestre"


def robust_date_mask(f: pd.Series) -> pd.Series:
    """Marca las fechas dentro del rango creíble (descarta 1900-01-01 y similares).

    Usa el rango intercuartílico de la propia distribución de fechas: con datos
    limpios no descarta nada, y con un puñado de fechas basura las aparta sin
    tocar el resto.
    """
    valid = f.dropna()
    if len(valid) < 20:
        return f.notna()
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr <= pd.Timedelta(0):
        return f.notna()
    lo, hi = q1 - 5 * iqr, q3 + 5 * iqr
    return f.notna() & (f >= lo) & (f <= hi)


def _agg_series(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str) -> pd.Series:
    f = to_datetime_series(df[date_col])
    d = df.assign(_f=f)[robust_date_mask(f)].set_index("_f")
    if d.empty:
        return pd.Series(dtype=float)
    if value_col and value_col in df.columns:
        s = d[value_col].pipe(to_numeric_series).resample(freq).sum()
    else:
        s = d.resample(freq).size().astype(float)
    # el último periodo suele estar incompleto: incluirlo simula una caída
    if len(s) and s.index[-1] > d.index.max():
        s = s.iloc[:-1]
    # unas pocas fechas basura (1900-01-01, 1899-12-31) generan cientos de
    # periodos vacíos que envenenan promedios y desviaciones: se recortan
    if len(s) and (s != 0).any():
        no_cero = np.flatnonzero(s.values != 0)
        s = s.iloc[no_cero[0]: no_cero[-1] + 1]
    return s


def _robust_span(f: pd.Series) -> int:
    """Rango temporal ignorando fechas atípicas en los extremos."""
    if len(f) < 20:
        return max((f.max() - f.min()).days, 1)
    lo, hi = f.quantile(0.01), f.quantile(0.99)
    return max((hi - lo).days, 1)


# ---------------------------------------------------------------- reglas


def _r_temporal(df, mapping, out: list[Insight]) -> None:
    date_col, val_col = mapping.get("fecha"), mapping.get("ingreso")
    if not date_col or date_col not in df.columns:
        return
    f_all = to_datetime_series(df[date_col])
    f = f_all[robust_date_mask(f_all)]
    descartadas = int(f_all.notna().sum() - len(f))
    if len(f) < 10:
        return
    span = _robust_span(f)
    freq, label = _period_freq(span)
    s = _agg_series(df, date_col, val_col, freq).dropna()
    if len(s) < 4:
        return
    metric = f"'{val_col}'" if val_col else "el número de registros"
    if descartadas:
        out.append(Insight(
            CONTEXTO, "Se apartaron fechas fuera de rango para el análisis temporal",
            f"{descartadas:,} registro(s) tienen una fecha imposible respecto al resto de la "
            "serie (típicamente 1900-01-01 por un campo vacío). Se excluyeron del análisis de "
            "tendencia para no distorsionarlo, pero siguen contando en los totales: "
            "corrígelos en la fase de Preparación.",
            impact=0.45, evidence={"descartadas": descartadas},
            titulo_simple="Algunos registros tienen una fecha imposible",
            simple=(f"{descartadas:,} registro(s) traen una fecha que no puede ser real "
                    "(normalmente 1900, que es lo que aparece cuando el campo se dejó vacío). "
                    "Los dejamos fuera de las gráficas de tiempo para que no las deformen, "
                    "pero siguen contando en los totales."),
        ))

    # actualidad
    dias_sin = (pd.Timestamp.today().normalize() - f.max().normalize()).days
    if dias_sin > 45:
        out.append(Insight(
            RIESGO, "Los datos no están actualizados",
            f"El último registro es del {f.max():%d/%m/%Y}, hace {dias_sin} días. "
            "Cualquier conclusión describe el pasado, no la operación actual.",
            impact=0.6, evidence={"dias_sin_datos": dias_sin},
            titulo_simple="Tus datos no están al día",
            simple=(f"El registro más reciente es del {f.max():%d/%m/%Y}, hace {dias_sin} días. "
                    "Lo que ves aquí describe cómo iban las cosas entonces, no cómo van hoy."),
        ))

    # tendencia (regresión sobre el índice del periodo)
    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)
    sl, ic, r, p, se = stats.linregress(x, y)
    promedio = float(np.mean(y)) or 1.0
    cambio_rel = safe_div(sl, abs(promedio))
    if p < 0.05 and abs(cambio_rel) > 0.01:
        direccion = "creciendo" if sl > 0 else "cayendo"
        kind = OPORTUNIDAD if sl > 0 else RIESGO
        out.append(Insight(
            kind, f"Tendencia sostenida a la {'alza' if sl>0 else 'baja'}",
            f"{metric.capitalize()} viene {direccion} {fmt_pct(abs(cambio_rel))} por {label} "
            f"en promedio ({len(s)} {label}s analizados, R²={r**2:.2f}, p={p:.3f}). "
            f"De mantenerse el ritmo, el próximo {label} rondaría {fmt_num(y[-1] + sl)}.",
            impact=min(0.95, 0.55 + abs(cambio_rel) * 2 + r**2 * 0.2),
            evidence={"pendiente": sl, "r2": r**2, "p": p, "periodos": len(s)},
            chart={"type": "line", "x": list(s.index), "y": list(y),
                   "title": f"{val_col or 'Registros'} por {label}"},
            titulo_simple=("Vas creciendo de forma sostenida" if sl > 0
                           else "Vas cayendo de forma sostenida"),
            simple=(f"{'Subes' if sl > 0 else 'Bajas'} alrededor de "
                    f"{fmt_pct(abs(cambio_rel))} cada {label}, y no es casualidad: el patrón se "
                    f"repite a lo largo de los {len(s)} {label}es analizados. Si sigue el mismo "
                    f"ritmo, el próximo {label} andaría cerca de {fmt_num(y[-1] + sl)}."),
        ))
    else:
        out.append(Insight(
            CONTEXTO, "Sin tendencia estadísticamente significativa",
            f"{metric.capitalize()} oscila sin dirección clara (p={p:.2f}). "
            f"La variación entre {label}s es ruido, no señal: no hay base para proyectar.",
            impact=0.3, evidence={"p": p},
            chart={"type": "line", "x": list(s.index), "y": list(y),
                   "title": f"{val_col or 'Registros'} por {label}"},
            titulo_simple="No hay una dirección clara",
            simple=(f"Los altibajos de un {label} a otro son normales, no una señal de que "
                    "vayas subiendo o bajando. Con esto no conviene hacer proyecciones."),
        ))

    # último periodo cerrado vs anterior y vs media histórica
    if len(s) >= 4:
        ultimo, previo = float(s.iloc[-1]), float(s.iloc[-2])
        hist = s.iloc[:-1]
        mu, sd = float(hist.mean()), float(hist.std(ddof=1) or 0)
        var = safe_div(ultimo - previo, abs(previo)) if previo else np.nan
        if sd > 0:
            z = (ultimo - mu) / sd
            if abs(z) >= 2:
                out.append(Insight(
                    RIESGO if z < 0 else OPORTUNIDAD,
                    f"El último {label} se sale del comportamiento normal",
                    f"Cerró en {fmt_num(ultimo)} contra un promedio histórico de {fmt_num(mu)} "
                    f"(±{fmt_num(sd)}): {abs(z):.1f} desviaciones estándar "
                    f"{'por debajo' if z<0 else 'por encima'}. "
                    + (f"Variación vs. el {label} previo: {fmt_pct(var)}. " if np.isfinite(var) else "")
                    + ("Vale la pena revisar si el periodo está completo antes de reaccionar."
                       if z < 0 else "Conviene entender qué lo provocó para intentar repetirlo."),
                    impact=min(0.95, 0.6 + abs(z) * 0.08),
                    evidence={"z": z, "ultimo": ultimo, "media": mu},
                    titulo_simple=(f"El último {label} fue muy "
                                   f"{'flojo' if z < 0 else 'bueno'}"),
                    simple=(f"Cerró en {fmt_num(ultimo)}, cuando lo normal ronda "
                            f"{fmt_num(mu)}. Es una diferencia grande, fuera de lo que suele "
                            f"variar de un {label} a otro. "
                            + ("Antes de alarmarte, confirma que el periodo esté completo."
                               if z < 0 else
                               "Vale la pena averiguar qué lo provocó para intentar repetirlo.")),
                ))


def _r_estacionalidad(df, mapping, out: list[Insight]) -> None:
    date_col, val_col = mapping.get("fecha"), mapping.get("ingreso")
    if not date_col or date_col not in df.columns:
        return
    f = to_datetime_series(df[date_col])
    d = df.assign(_f=f)[robust_date_mask(f)]
    if len(d) < 60:
        return
    span = _robust_span(d["_f"])
    val = to_numeric_series(d[val_col]) if val_col and val_col in df.columns else pd.Series(1.0, index=d.index)
    unidad = f"'{val_col}'" if val_col else "los registros"

    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
    if span >= 28:
        by_dow = val.groupby(d["_f"].dt.dayofweek).sum()
        by_dow = by_dow.reindex(range(7)).fillna(0)
        if by_dow.sum() > 0:
            share = by_dow / by_dow.sum()
            mejor, peor = int(share.idxmax()), int(share.idxmin())
            if share.max() > 1.6 / 7:
                out.append(Insight(
                    HALLAZGO, "Hay un patrón claro por día de la semana",
                    f"{dias_es[mejor]} concentra {fmt_pct(share.max())} de {unidad}, contra "
                    f"{fmt_pct(share.min())} de {dias_es[peor]} (si no hubiera patrón, cada día "
                    f"tendría ~14.3%). Es información directa para programar personal, "
                    "inventario o campañas.",
                    impact=0.6,
                    evidence={"mejor_dia": dias_es[mejor], "peor_dia": dias_es[peor]},
                    chart={"type": "bar", "labels": dias_es, "values": list(by_dow.values),
                           "title": f"{val_col or 'Registros'} por día de la semana"},
                    titulo_simple=f"El {dias_es[mejor].lower()} es tu mejor día",
                    simple=(f"{dias_es[mejor]} concentra {fmt_pct(share.max())} del total y "
                            f"{dias_es[peor].lower()} apenas {fmt_pct(share.min())}. Si todos los "
                            "días fueran iguales, cada uno tendría un 14%. Sirve directo para "
                            "decidir turnos, inventario y cuándo lanzar promociones."),
                ))

    if span >= 365:
        meses = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                 "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        by_m = val.groupby(d["_f"].dt.month).sum().reindex(range(1, 13)).fillna(0)
        if by_m.sum() > 0:
            share = by_m / by_m.sum()
            if share.max() > 1.5 / 12:
                out.append(Insight(
                    HALLAZGO, "Hay estacionalidad mensual",
                    f"{meses[int(share.idxmax())-1]} es el mes más fuerte con {fmt_pct(share.max())} "
                    f"del total anual y {meses[int(share.idxmin())-1]} el más débil con "
                    f"{fmt_pct(share.min())}. Planea flujo de efectivo e inventario contra esta curva, "
                    "no contra el promedio.",
                    impact=0.65,
                    chart={"type": "bar", "labels": meses, "values": list(by_m.values),
                           "title": f"{val_col or 'Registros'} por mes"},
                    titulo_simple=f"Tu año se concentra en {meses[int(share.idxmax())-1]}",
                    simple=(f"{meses[int(share.idxmax())-1]} vale {fmt_pct(share.max())} del año "
                            f"y {meses[int(share.idxmin())-1]} solo {fmt_pct(share.min())}. "
                            "Planea tu efectivo y tu inventario con esa curva, no con el "
                            "promedio anual."),
                ))


def _r_concentracion(df, mapping, profiles, out: list[Insight]) -> None:
    val_col = mapping.get("ingreso")
    dims = list(dict.fromkeys(
        c for c in (mapping.get("cliente"), mapping.get("producto"), mapping.get("segmento"))
        if c and c in df.columns))
    for dim in dims:
        if df[dim].nunique() < 5:
            continue
        if val_col and val_col in df.columns:
            agg = df.groupby(dim)[val_col].apply(lambda x: to_numeric_series(x).sum())
            unidad, medida = f"del total de '{val_col}'", val_col
        else:
            agg = df.groupby(dim).size().astype(float)
            unidad, medida = "de los registros", "registros"
        agg = agg[agg > 0].sort_values(ascending=False)
        if len(agg) < 5 or agg.sum() <= 0:
            continue

        k20 = max(1, int(round(len(agg) * 0.2)))
        share20 = float(agg.head(k20).sum() / agg.sum())
        top1 = float(agg.iloc[0] / agg.sum())

        if share20 >= 0.7:
            # con muchos elementos, "el 20% hace el 72%" es Pareto normal, no
            # dependencia de nadie: el titular tiene que distinguir los dos casos
            depende_de_uno = top1 > 0.3
            out.append(Insight(
                RIESGO if depende_de_uno else HALLAZGO,
                f"Concentración fuerte en '{dim}'",
                f"El 20% superior ({k20} de {len(agg)}) genera {fmt_pct(share20)} {unidad}. "
                f"El primero solo, '{agg.index[0]}', aporta {fmt_pct(top1)}. "
                + ("Es una dependencia peligrosa: perder ese elemento tumba el resultado."
                   if top1 > 0.3 else "Clásico patrón de Pareto: enfoca ahí el esfuerzo comercial."),
                impact=min(0.9, 0.5 + share20 * 0.4 + top1),
                evidence={"share_top20": share20, "top1": str(agg.index[0]), "share_top1": top1},
                chart={"type": "bar", "labels": [etiqueta(i) for i in agg.head(10).index],
                       "values": list(agg.head(10).values),
                       "title": f"Top 10 de '{dim}' por {medida}", "highlight": k20},
                titulo_simple=(f"Dependes demasiado de '{etiqueta(agg.index[0])}'" if depende_de_uno
                               else f"El 20% de '{dim}' genera el {fmt_pct(share20)}"),
                simple=(
                    (f"'{etiqueta(agg.index[0])}' por sí solo aporta {fmt_pct(top1)} del total, y entre "
                     f"los {k20} más grandes juntan {fmt_pct(share20)}. Eso es riesgoso: si ese "
                     "se va, se cae buena parte del negocio. Vale la pena cuidarlo mucho y, en "
                     "paralelo, buscar diversificar.")
                    if depende_de_uno else
                    (f"{k20} de {len(agg):,} valores de '{dim}' generan {fmt_pct(share20)} del "
                     f"total. Ninguno manda por sí solo — el más grande es apenas "
                     f"{fmt_pct(top1)} — así que no hay dependencia de nadie en particular. "
                     "Es el patrón 80/20 de siempre: ese grupo es donde rinde más el esfuerzo "
                     "comercial.")),
            ))

        cola = agg[agg.cumsum() / agg.sum() > 0.95]
        if len(cola) >= max(5, len(agg) * 0.3):
            out.append(Insight(
                OPORTUNIDAD, f"Cola larga improductiva en '{dim}'",
                f"{len(cola)} de {len(agg)} valores ({len(cola)/len(agg):.0%}) aportan en conjunto "
                f"menos del 5% {unidad}. Cada uno consume atención, inventario o costo de servicio; "
                "depurar esa cola libera recursos sin afectar el resultado.",
                impact=0.55, evidence={"n_cola": len(cola), "n_total": len(agg)},
                titulo_simple=f"Tienes mucha cola improductiva en {dim}",
                simple=(f"{len(cola)} de {len(agg)} ({len(cola)/len(agg):.0%}) aportan, entre "
                        "todos juntos, menos del 5% del total. Cada uno te consume atención, "
                        "inventario o costo de atender. Depurar esa lista te libera recursos "
                        "casi sin afectar el resultado."),
            ))


def _r_segmentos(df, mapping, profiles, out: list[Insight]) -> None:
    val_col = mapping.get("ingreso")
    if not val_col or val_col not in df.columns:
        return
    v = to_numeric_series(df[val_col])
    dims = list(dict.fromkeys(
        c for c in (mapping.get("segmento"), mapping.get("producto"))
        if c and c in df.columns and 1 < df[c].nunique() <= 40))
    for dim in dims:
        d = pd.DataFrame({"g": df[dim].astype(str), "v": v}).dropna()
        if len(d) < 30:
            continue
        grupos = [g["v"].values for _, g in d.groupby("g") if len(g) >= 5]
        if len(grupos) < 2:
            continue
        try:
            f_stat, p = stats.f_oneway(*grupos)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(p) or p >= 0.05:
            out.append(Insight(
                CONTEXTO, f"'{dim}' no explica diferencias en '{val_col}'",
                f"Las medias por categoría no son estadísticamente distintas (ANOVA p={p:.2f}). "
                "Segmentar por aquí no aporta: busca otra variable explicativa.",
                impact=0.25,
                titulo_simple=f"Separar por '{dim}' no explica nada",
                simple=(f"Todas las categorías de '{dim}' se comportan prácticamente igual en "
                        f"'{val_col}'. Las diferencias que veas ahí son casualidad, no una "
                        "señal. Busca otra forma de segmentar."),
            ))
            continue

        medias = d.groupby("g")["v"].agg(["mean", "count", "sum"]).sort_values("mean", ascending=False)
        global_mean = float(d["v"].mean())
        mejor, peor = medias.index[0], medias.index[-1]
        lift = safe_div(medias.loc[mejor, "mean"] - global_mean, abs(global_mean))
        drop = safe_div(medias.loc[peor, "mean"] - global_mean, abs(global_mean))
        out.append(Insight(
            OPORTUNIDAD, f"'{dim}' sí discrimina el desempeño",
            f"El promedio de '{val_col}' varía de forma significativa entre categorías "
            f"(ANOVA p={p:.4f}). **{mejor}** promedia {fmt_num(medias.loc[mejor,'mean'])} "
            f"({fmt_pct(lift)} sobre la media general de {fmt_num(global_mean)}, "
            f"{int(medias.loc[mejor,'count'])} registros), mientras **{peor}** promedia "
            f"{fmt_num(medias.loc[peor,'mean'])} ({fmt_pct(drop)}). "
            "Replicar lo que hace el primero es la palanca más directa.",
            impact=min(0.9, 0.6 + abs(lift) * 0.3),
            evidence={"p": p, "mejor": str(mejor), "peor": str(peor)},
            chart={"type": "box", "dim": dim, "value": val_col,
                   "title": f"Distribución de '{val_col}' por '{dim}'"},
            titulo_simple=f"'{mejor}' rinde mucho más que el resto",
            simple=(f"Separando por '{dim}', **{mejor}** promedia "
                    f"{fmt_num(medias.loc[mejor,'mean'])} contra "
                    f"{fmt_num(medias.loc[peor,'mean'])} de **{peor}**. La diferencia es real, "
                    "no ruido de los datos. Entender qué hace distinto al primero y copiarlo "
                    "es la palanca más directa que tienes aquí."),
        ))


def _r_crecimiento_por_segmento(df, mapping, out: list[Insight]) -> None:
    date_col, val_col = mapping.get("fecha"), mapping.get("ingreso")
    dim = mapping.get("segmento") or mapping.get("producto")
    if not (date_col and dim) or date_col not in df.columns or dim not in df.columns:
        return
    f = to_datetime_series(df[date_col])
    d = df.assign(_f=f)[robust_date_mask(f)]
    if len(d) < 60 or d[dim].nunique() > 40:
        return
    corte = d["_f"].median()   # la mediana parte la muestra en dos mitades reales
    val = to_numeric_series(d[val_col]) if val_col and val_col in df.columns else pd.Series(1.0, index=d.index)
    d = d.assign(_v=val)
    a = d[d["_f"] <= corte].groupby(dim)["_v"].sum()
    b = d[d["_f"] > corte].groupby(dim)["_v"].sum()
    comp = pd.DataFrame({"antes": a, "despues": b}).fillna(0)
    comp = comp[comp["antes"] > 0]
    if len(comp) < 3:
        return
    comp["var"] = (comp["despues"] - comp["antes"]) / comp["antes"]
    comp["peso"] = comp["antes"] / comp["antes"].sum()
    rel = comp[comp["peso"] >= 0.05]
    if rel.empty:
        return
    sube = rel.sort_values("var", ascending=False).iloc[0]
    baja = rel.sort_values("var").iloc[0]
    medida = f"'{val_col}'" if val_col else "el volumen"
    if baja["var"] < -0.15:
        out.append(Insight(
            RIESGO, f"Caída concentrada en '{baja.name}'",
            f"Comparando la primera mitad del periodo contra la segunda, {medida} de "
            f"**{baja.name}** cayó {fmt_pct(abs(baja['var']))} "
            f"({fmt_num(baja['antes'])} → {fmt_num(baja['despues'])}). Pesaba "
            f"{fmt_pct(baja['peso'])} del total, así que arrastra el resultado global.",
            impact=min(0.92, 0.6 + abs(baja["var"]) * 0.3 + baja["peso"]),
            evidence={"segmento": str(baja.name), "variacion": float(baja["var"])},
            titulo_simple=f"'{baja.name}' se está cayendo",
            simple=(f"Comparando la primera mitad del periodo contra la segunda, "
                    f"**{baja.name}** bajó {fmt_pct(abs(baja['var']))} "
                    f"({fmt_num(baja['antes'])} → {fmt_num(baja['despues'])}). Como pesaba "
                    f"{fmt_pct(baja['peso'])} del total, esa caída arrastra el resultado "
                    "general."),
        ))
    if sube["var"] > 0.15:
        out.append(Insight(
            OPORTUNIDAD, f"Crecimiento acelerado en '{sube.name}'",
            f"{medida.capitalize()} de **{sube.name}** subió {fmt_pct(sube['var'])} entre la primera "
            f"y la segunda mitad del periodo ({fmt_num(sube['antes'])} → {fmt_num(sube['despues'])}). "
            "Vale la pena verificar si es sostenible antes de asignarle más recursos.",
            impact=min(0.88, 0.55 + sube["var"] * 0.2 + sube["peso"]),
            evidence={"segmento": str(sube.name), "variacion": float(sube["var"])},
            titulo_simple=f"'{sube.name}' está despegando",
            simple=(f"Subió {fmt_pct(sube['var'])} entre la primera y la segunda mitad del "
                    f"periodo ({fmt_num(sube['antes'])} → {fmt_num(sube['despues'])}). "
                    "Antes de meterle más recursos, confirma que el crecimiento sea sostenible "
                    "y no un pico puntual."),
        ))


def _r_dispersion(df, mapping, out: list[Insight]) -> None:
    val_col = mapping.get("ingreso")
    if not val_col or val_col not in df.columns:
        return
    v = to_numeric_series(df[val_col]).dropna()
    if len(v) < 30 or v.mean() == 0:
        return
    cv = float(v.std(ddof=1) / abs(v.mean()))
    p90, p50 = float(v.quantile(0.9)), float(v.median())
    if cv > 1.2:
        out.append(Insight(
            HALLAZGO, f"'{val_col}' es muy heterogéneo",
            f"El coeficiente de variación es {cv:.1f} (arriba de 1 ya es alta dispersión). "
            f"La mediana es {fmt_num(p50)} pero el 10% superior arranca en {fmt_num(p90)}: "
            "hablar de 'el promedio' aquí engaña. Conviene analizar por rangos o segmentos.",
            impact=0.55, evidence={"cv": cv, "p50": p50, "p90": p90},
            chart={"type": "hist", "col": val_col, "title": f"Distribución de '{val_col}'"},
            titulo_simple="Hablar del 'promedio' aquí te va a engañar",
            simple=(f"Tus registros son muy dispares: la mitad está por debajo de "
                    f"{fmt_num(p50)}, pero el 10% más grande arranca en {fmt_num(p90)}. "
                    "El promedio queda en tierra de nadie. Conviene ver por rangos o por "
                    "segmentos, no un solo número."),
        ))

    top = v.sort_values(ascending=False)
    k = max(1, int(len(v) * 0.01))
    share = float(top.head(k).sum() / top.sum()) if top.sum() else 0
    if share > 0.2:
        out.append(Insight(
            RIESGO, "Unos pocos registros dominan el total",
            f"El 1% de los registros más grandes ({k} de {len(v):,}) representa {fmt_pct(share)} "
            f"de la suma de '{val_col}'. Verifica que no sean errores de captura antes de "
            "usarlos como base de cualquier proyección.",
            impact=0.6, evidence={"share_top1pct": share},
            titulo_simple="Unos pocos registros mueven casi todo el total",
            simple=(f"Los {k} registros más grandes (el 1%) valen {fmt_pct(share)} de la suma. "
                    "Antes de usar estos totales para proyectar, revisa a mano que no sean "
                    "errores de captura: un cero de más en una venta cambia todo."),
        ))


def _r_correlaciones(df, profiles, mapping, out: list[Insight]) -> None:
    num = [p.name for p in profiles.values()
           if p.semantic == "numerico" and p.role != "identificador" and p.n_unique > 5]
    if len(num) < 2:
        return
    sub = df[num].apply(to_numeric_series).dropna()
    if len(sub) < 20:
        return
    corr = sub.corr()
    pares = []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and 0.5 <= abs(r) < 0.98:
                pares.append((a, b, float(r)))
    pares.sort(key=lambda t: -abs(t[2]))
    for a, b, r in pares[:3]:
        signo = "en el mismo sentido" if r > 0 else "en sentido contrario"
        out.append(Insight(
            HALLAZGO, f"'{a}' y '{b}' se mueven juntos",
            f"Correlación de {r:.2f}: cuando una sube, la otra se mueve {signo} de forma "
            f"consistente (explica {r**2:.0%} de la variación). "
            "Correlación no es causalidad — sirve como hipótesis, no como conclusión.",
            impact=0.4 + abs(r) * 0.2, evidence={"a": a, "b": b, "r": r},
            chart={"type": "scatter", "x": a, "y": b, "title": f"{a} vs. {b}"},
            titulo_simple=f"'{a}' y '{b}' se mueven al mismo ritmo",
            simple=(f"Cuando una sube, la otra {'sube' if r > 0 else 'baja'} de forma bastante "
                    "consistente. Ojo: que vayan juntas no significa que una cause la otra. "
                    "Tómalo como una pista para investigar, no como una conclusión."),
        ))


def _r_desbalance(df, profiles, out: list[Insight]) -> None:
    for p in profiles.values():
        if p.semantic not in ("categorico", "booleano") or not (1 < p.n_unique <= 20):
            continue
        vc = df[p.name].astype(str).value_counts(normalize=True)
        if vc.iloc[0] > 0.9:
            out.append(Insight(
                CONTEXTO, f"'{p.name}' está casi siempre en el mismo valor",
                f"'{vc.index[0]}' cubre {fmt_pct(vc.iloc[0])} de los registros. Como dimensión de "
                "análisis aporta poco, y si fuera la variable objetivo de un modelo, la exactitud "
                "sería engañosa (acertar siempre esa clase ya da esa cifra).",
                impact=0.3, evidence={"columna": p.name, "dominante": str(vc.index[0])},
                titulo_simple=f"'{p.name}' casi no varía",
                simple=(f"El {fmt_pct(vc.iloc[0])} de los registros dice '{vc.index[0]}'. "
                        "Como no hay variedad, esa columna no sirve para comparar ni para "
                        "explicar nada."),
            ))


def _r_calidad(issues, out: list[Insight]) -> None:
    criticos = [i for i in issues if i.severity == "crítico"]
    if criticos:
        top = criticos[0]
        out.append(Insight(
            RIESGO, "La calidad de los datos limita las conclusiones",
            f"Hay {len(criticos)} problema(s) crítico(s) sin resolver. El de mayor alcance: "
            f"{top.title.lower()} — {top.detail} Resuélvelos en la fase de Preparación antes de "
            "tomar decisiones con estos números.",
            impact=0.98, evidence={"n_criticos": len(criticos)},
            titulo_simple="Hay problemas en los datos que afectan estos números",
            simple=(f"Detectamos {len(criticos)} problema(s) serio(s). El más grande: "
                    f"{top.title.lower()}. Revisa la sección «Qué revisar en tus datos» antes "
                    "de tomar decisiones con las cifras de arriba."),
        ))


# ---------------------------------------------------------------- API


def generate_insights(df: pd.DataFrame, profiles: dict, mapping: dict,
                      issues: list | None = None, limit: int = 12) -> list[Insight]:
    out: list[Insight] = []
    if len(df) == 0:
        return out
    if issues:
        _r_calidad(issues, out)
    for rule in (
        lambda: _r_temporal(df, mapping, out),
        lambda: _r_estacionalidad(df, mapping, out),
        lambda: _r_concentracion(df, mapping, profiles, out),
        lambda: _r_segmentos(df, mapping, profiles, out),
        lambda: _r_crecimiento_por_segmento(df, mapping, out),
        lambda: _r_dispersion(df, mapping, out),
        lambda: _r_correlaciones(df, profiles, mapping, out),
        lambda: _r_desbalance(df, profiles, out),
    ):
        try:
            rule()
        except Exception as e:  # noqa: BLE001 — una regla que falla no debe tumbar el análisis
            out.append(Insight(CONTEXTO, "Una regla de análisis no pudo ejecutarse",
                               f"Detalle técnico: {type(e).__name__}: {e}", impact=0.05))
    out.sort(key=lambda i: -i.impact)
    return out[:limit]


def executive_summary(insights: list[Insight], overview: dict, score: int) -> str:
    riesgos = [i for i in insights if i.kind == RIESGO]
    oportunidades = [i for i in insights if i.kind == OPORTUNIDAD]
    partes = [
        f"Se analizaron **{overview['filas']:,} registros** y **{overview['columnas']} variables**. "
        f"La calidad de los datos obtuvo **{score}/100**."
    ]
    if riesgos:
        partes.append(f"**Riesgos ({len(riesgos)}):** " + riesgos[0].title.lower() +
                      (f"; {riesgos[1].title.lower()}" if len(riesgos) > 1 else "") + ".")
    if oportunidades:
        partes.append(f"**Oportunidades ({len(oportunidades)}):** " + oportunidades[0].title.lower() +
                      (f"; {oportunidades[1].title.lower()}" if len(oportunidades) > 1 else "") + ".")
    if not riesgos and not oportunidades:
        partes.append("No se detectaron patrones accionables con la configuración actual. "
                      "Mapea las columnas de negocio en la fase 1 para habilitar más reglas.")
    return " ".join(partes)
