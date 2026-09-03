"""Gráficas Plotly con una paleta validada, consistente en modo claro y oscuro.

Reglas aplicadas: hues categóricos en orden fijo (nunca cíclico), un solo eje Y,
secuencial = un solo tono, divergente = azul↔rojo con punto neutro gris,
leyenda siempre que haya ≥2 series, rejilla discreta y tope de categorías.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import etiqueta, fmt_num

try:  # detección de tema; la app siempre corre en Streamlit
    import streamlit as st
except Exception:  # pragma: no cover
    st = None

# ------------------------------------------------------------------ paleta

CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_DARK = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

STATUS = {"ok": "#008300", "cerca": "#eda100", "bajo": "#e34948", "neutro": "#8a8a85"}
SEVERITY_COLOR = {"crítico": "#e34948", "advertencia": "#eda100", "info": "#2a78d6"}

MAX_CATEGORIES = 8          # más allá de esto se agrupa en "Otros"
MAX_CATEGORIES_SCATTER = 3  # el modo all-pairs solo valida 3 slots


def _dark() -> bool:
    try:
        return (st.get_option("theme.base") or "light").lower() == "dark"
    except Exception:
        return False


def palette() -> list[str]:
    return CATEGORICAL_DARK if _dark() else CATEGORICAL_LIGHT


def sequential() -> list[str]:
    return SEQ_DARK if _dark() else SEQ_LIGHT


def _ink() -> tuple[str, str, str]:
    if _dark():
        return "#ffffff", "#c3c2b7", "rgba(255,255,255,0.10)"
    return "#0b0b0b", "#52514e", "rgba(11,11,11,0.10)"


def _layout(fig: go.Figure, title: str = "", ylab: str = "", xlab: str = "",
            legend: bool = False, height: int = 360) -> go.Figure:
    primary, secondary, grid = _ink()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=primary), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=secondary, size=12,
                  family="Inter, -apple-system, Segoe UI, Roboto, sans-serif"),
        margin=dict(l=8, r=8, t=44 if title else 12, b=8),
        height=height,
        hovermode="x unified" if fig.data and fig.data[0].type in ("scatter", "bar") else "closest",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    font=dict(color=secondary, size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title_text=xlab, showgrid=False, zeroline=False,
                     linecolor=grid, tickfont=dict(color=secondary, size=11),
                     title_font=dict(color=secondary, size=11))
    fig.update_yaxes(title_text=ylab, gridcolor=grid, zeroline=False, linecolor="rgba(0,0,0,0)",
                     tickfont=dict(color=secondary, size=11),
                     title_font=dict(color=secondary, size=11))
    return fig


def top_con_otros(serie: pd.Series, n: int = 8, etiqueta_resto: str = "Otros"):
    """Top n de una serie agregada; el resto se suma en una barra 'Otros (N)'.

    Cortar la cola en silencio hace que las barras no sumen el total y el lector
    saque conclusiones equivocadas: mejor mostrarla agrupada.
    """
    s = serie.dropna().sort_values(ascending=False)
    if len(s) <= n:
        return [etiqueta(i) for i in s.index], list(s.values), 0
    cabeza, resto = s.iloc[:n - 1], s.iloc[n - 1:]
    labels = [etiqueta(i) for i in cabeza.index] + [f"{etiqueta_resto} ({len(resto):,})"]
    values = list(cabeza.values) + [float(resto.sum())]
    return labels, values, len(resto)


def _fold_other(s: pd.Series, top: int = MAX_CATEGORIES, label: str = "Otros") -> pd.Series:
    if s.nunique() <= top:
        return s
    keep = s.value_counts().head(top - 1).index
    return s.where(s.isin(keep), label)


# ------------------------------------------------------------------ formas


def line_time(df: pd.DataFrame, x: str, y: str, color: str | None = None,
              title: str = "", ylab: str = "", height: int = 360) -> go.Figure:
    """Evolución temporal. Una línea, o varias si se pasa `color` (máx. 8)."""
    fig = go.Figure()
    pal = palette()
    if color:
        d = df.copy()
        d[color] = _fold_other(d[color].astype(str))
        cats = list(d.groupby(color)[y].sum().sort_values(ascending=False).index)
        for i, cat in enumerate(cats[:MAX_CATEGORIES]):
            sub = d[d[color] == cat].sort_values(x)
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="lines", name=str(cat),
                line=dict(width=2, color=pal[i % len(pal)]),
                hovertemplate=f"<b>{cat}</b>: %{{y:,.2f}}<extra></extra>",
            ))
        fig = _layout(fig, title, ylab, "", legend=len(cats) >= 2, height=height)
        _eje_fechas_es(fig, df[x])
        return fig

    d = df.sort_values(x)
    # con pocos puntos se dibujan los marcadores: se ve dónde cae cada periodo
    # y, en el tablero, son el blanco al que se le pica para abrir la gráfica
    modo = "lines+markers" if len(d) <= 120 else "lines"
    fig.add_trace(go.Scatter(
        x=d[x], y=d[y], mode=modo, name=y,
        line=dict(width=2, color=pal[0]),
        marker=dict(size=6, color=pal[0]),
        fill="tozeroy", fillcolor=_alpha(pal[0], 0.12),
        hovertemplate="%{y:,.2f}<extra></extra>",
    ))
    fig = _layout(fig, title, ylab, "", legend=False, height=height)
    _eje_fechas_es(fig, d[x])
    return fig


MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun",
            "jul", "ago", "sep", "oct", "nov", "dic"]


def _eje_fechas_es(fig: go.Figure, x) -> None:
    """Pone los meses en español en el eje del tiempo.

    Plotly viene con los nombres de mes en inglés y no trae el paquete de
    idiomas, así que los rótulos se escriben a mano. En una app que habla
    español, un eje que dice «Apr 2024» se siente ajeno.
    """
    try:
        s = pd.Series(pd.to_datetime(pd.Series(list(x)), errors="coerce")).dropna()
        if len(s) < 2:
            return
        s = s.sort_values().reset_index(drop=True)
        span = (s.iloc[-1] - s.iloc[0]).days
        # seis marcas como máximo: en una tarjeta angosta, ocho se encinman
        # y Plotly las gira en diagonal
        n = int(min(6, max(3, len(s))))
        idx = sorted({int(round(i)) for i in np.linspace(0, len(s) - 1, n)})
        pts = [s.iloc[i] for i in idx]
        if span > 400:
            # "ene 24" y no "ene 2024": en una tarjeta angosta, el año completo
            # obliga a Plotly a girar los rótulos en diagonal
            etq = [f"{MESES_ES[t.month - 1]} {t.year % 100:02d}" for t in pts]
        elif span > 60:
            etq = [f"{MESES_ES[t.month - 1]} {t.year % 100:02d}" for t in pts]
        else:
            etq = [f"{t.day} {MESES_ES[t.month - 1]}" for t in pts]
        fig.update_xaxes(tickmode="array", tickvals=pts, ticktext=etq)
    except Exception:  # noqa: BLE001
        pass


def _alpha(hexcolor: str, a: float) -> str:
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def bar_ranked(labels, values, title: str = "", xlab: str = "", horizontal: bool = True,
               highlight: int = 0, height: int = 360, value_fmt: str = ",.2f",
               etiquetas_valor: bool = True, prefijo: str = "") -> go.Figure:
    """Ranking. Un solo color: el eje ya distingue las categorías.

    El eje de categorías se declara `type="category"` a la fuerza. Si no, unas
    etiquetas que parecen números (IDs de producto, folios) hacen que Plotly use
    una escala continua y las barras salgan como hilos en posiciones absurdas.
    """
    pal = palette()
    base = pal[0]
    etq = [etiqueta(l) for l in labels]
    vals = [float(v) for v in values]
    colors = [base if (highlight == 0 or i < highlight) else _alpha(base, 0.45)
              for i in range(len(etq))]
    _, secondary, _ = _ink()
    texto = [f"{prefijo}{fmt_num(v)}" for v in vals] if etiquetas_valor else None
    alto = max(height, 34 * len(etq) + 70) if horizontal else height

    fig = go.Figure()
    if horizontal:
        fig.add_trace(go.Bar(
            y=etq[::-1], x=vals[::-1], orientation="h",
            marker=dict(color=colors[::-1], line=dict(width=0)),
            text=(texto[::-1] if texto else None), textposition="outside",
            textfont=dict(color=secondary, size=11), cliponaxis=False,
            hovertemplate="<b>%{y}</b>: %{x:" + value_fmt + "}<extra></extra>",
        ))
        fig = _layout(fig, title, "", xlab, height=alto)
        fig.update_yaxes(type="category", categoryorder="array",
                         categoryarray=etq[::-1], showgrid=False)
        if vals:
            fig.update_xaxes(range=[0, max(vals) * (1.22 if etiquetas_valor else 1.02)])
    else:
        fig.add_trace(go.Bar(
            x=etq, y=vals,
            marker=dict(color=colors, line=dict(width=0)),
            text=texto, textposition="outside",
            textfont=dict(color=secondary, size=11), cliponaxis=False,
            hovertemplate="<b>%{x}</b>: %{y:" + value_fmt + "}<extra></extra>",
        ))
        fig = _layout(fig, title, xlab, "", height=height)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=etq)
        if vals:
            fig.update_yaxes(range=[0, max(vals) * (1.18 if etiquetas_valor else 1.02)])
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(bargap=0.32 if len(etq) > 3 else 0.55, hovermode="closest")
    return fig


def bar_grouped_time(df: pd.DataFrame, x: str, y: str, color: str,
                     title: str = "", ylab: str = "", height: int = 380) -> go.Figure:
    """Barras apiladas por periodo, con separación de 2px entre segmentos."""
    d = df.copy()
    d[color] = _fold_other(d[color].astype(str))
    pivot = d.pivot_table(index=x, columns=color, values=y, aggfunc="sum").fillna(0)
    order = pivot.sum().sort_values(ascending=False).index
    pal = palette()
    fig = go.Figure()
    for i, cat in enumerate(order):
        fig.add_trace(go.Bar(
            x=pivot.index, y=pivot[cat], name=str(cat),
            marker=dict(color=pal[i % len(pal)], line=dict(width=1, color="rgba(0,0,0,0)")),
            hovertemplate=f"<b>{cat}</b>: %{{y:,.2f}}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", bargap=0.25)
    fig = _layout(fig, title, ylab, "", legend=len(order) >= 2, height=height)
    return fig


def histogram(values, title: str = "", xlab: str = "", height: int = 300,
              median: float | None = None) -> go.Figure:
    pal = palette()
    v = pd.Series(values).dropna()
    fig = go.Figure(go.Histogram(
        x=v, marker=dict(color=pal[0], line=dict(width=0)), nbinsx=min(50, max(10, int(len(v) ** 0.5))),
        hovertemplate="%{x}: %{y} registros<extra></extra>",
    ))
    fig = _layout(fig, title, "Registros", xlab, height=height)
    fig.update_layout(bargap=0.04, hovermode="closest")
    if median is not None:
        _, secondary, _ = _ink()
        fig.add_vline(x=median, line_width=2, line_dash="dot", line_color=secondary,
                      annotation_text="mediana", annotation_font_color=secondary,
                      annotation_font_size=11)
    return fig


def box_by_group(df: pd.DataFrame, group: str, value: str, title: str = "",
                 height: int = 360) -> go.Figure:
    d = df.copy()
    d[group] = _fold_other(d[group].astype(str))
    order = d.groupby(group)[value].median().sort_values(ascending=False).index
    pal = palette()
    fig = go.Figure()
    for i, cat in enumerate(order):
        fig.add_trace(go.Box(
            y=d.loc[d[group] == cat, value], name=str(cat),
            marker=dict(color=pal[i % len(pal)]), line=dict(width=2),
            boxpoints=False,
        ))
    fig = _layout(fig, title, value, "", legend=False, height=height)
    fig.update_xaxes(type="category")
    # sin acotar el eje, un solo valor extremo aplasta todas las cajas hasta
    # volverlas rayas: se recorta la vista al rango donde vive la mayoría
    q = d.groupby(group)[value].quantile([0.25, 0.75]).unstack()
    if not q.empty and q.notna().all().all():
        iqr = float((q[0.75] - q[0.25]).median())
        if iqr > 0:
            lo = float(q[0.25].min()) - 1.8 * iqr
            hi = float(q[0.75].max()) + 1.8 * iqr
            real_lo, real_hi = float(d[value].min()), float(d[value].max())
            fig.update_yaxes(range=[max(lo, real_lo - 0.02 * abs(real_lo)),
                                    min(hi, real_hi)])
    return fig


def heatmap_corr(corr: pd.DataFrame, title: str = "", height: int = 420) -> go.Figure:
    """Correlaciones: divergente azul↔rojo con gris neutro en cero."""
    mid = "#383835" if _dark() else "#f0efec"
    scale = [(0.0, "#e34948"), (0.25, "#e87ba4"), (0.5, mid),
             (0.75, "#6da7ec"), (1.0, "#0d366b")]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale=scale, xgap=2, ygap=2,
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="r", thickness=10, outlinewidth=0),
    ))
    fig = _layout(fig, title, "", "", height=height)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(showgrid=False)
    return fig


def heatmap_seq(z, x, y, title: str = "", height: int = 380, zlabel: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, colorscale=[[i / (len(sequential()) - 1), c]
                                   for i, c in enumerate(sequential())],
        xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}: %{z:,.2f}<extra></extra>",
        colorbar=dict(title=zlabel, thickness=10, outlinewidth=0),
    ))
    fig = _layout(fig, title, "", "", height=height)
    fig.update_layout(hovermode="closest")
    fig.update_yaxes(showgrid=False)
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None,
            title: str = "", height: int = 400, trend: bool = True) -> go.Figure:
    pal = palette()
    fig = go.Figure()
    if color:
        d = df.copy()
        d[color] = _fold_other(d[color].astype(str), top=MAX_CATEGORIES_SCATTER)
        cats = list(d[color].value_counts().index)
        for i, cat in enumerate(cats):
            sub = d[d[color] == cat]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="markers", name=str(cat),
                marker=dict(size=9, color=pal[i % len(pal)], opacity=0.75,
                            line=dict(width=2, color="rgba(0,0,0,0)")),
                hovertemplate=f"<b>{cat}</b><br>{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>",
            ))
        legend = len(cats) >= 2
    else:
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], mode="markers", name=y,
            marker=dict(size=9, color=pal[0], opacity=0.7,
                        line=dict(width=2, color="rgba(0,0,0,0)")),
            hovertemplate=f"{x}: %{{x:,.2f}}<br>{y}: %{{y:,.2f}}<extra></extra>",
        ))
        legend = False
    if trend and len(df) > 3:
        sub = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) > 3 and sub[x].nunique() > 1:
            b, a = np.polyfit(sub[x], sub[y], 1)
            xs = np.linspace(sub[x].min(), sub[x].max(), 50)
            _, secondary, _ = _ink()
            fig.add_trace(go.Scatter(
                x=xs, y=a + b * xs, mode="lines", name="tendencia",
                line=dict(width=2, dash="dot", color=secondary), hoverinfo="skip",
            ))
    fig = _layout(fig, title, y, x, legend=legend, height=height)
    fig.update_layout(hovermode="closest")
    return fig


def missing_bar(profiles, height: int = 320) -> go.Figure | None:
    data = [(p.name, p.pct_missing * 100) for p in profiles.values() if p.pct_missing > 0]
    if not data:
        return None
    data.sort(key=lambda t: -t[1])
    data = data[:20]
    return bar_ranked([d[0] for d in data], [d[1] for d in data],
                      title="Valores faltantes por columna", xlab="% de registros vacíos",
                      height=max(height, 24 * len(data) + 80), value_fmt=".1f")


def severity_donut(counts: dict, height: int = 220) -> go.Figure:
    labels = [k for k, v in counts.items() if v]
    values = [counts[k] for k in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=[SEVERITY_COLOR.get(l, "#8a8a85") for l in labels],
                    line=dict(width=2, color="rgba(0,0,0,0)")),
        textinfo="value", hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig = _layout(fig, "", "", "", legend=True, height=height)
    fig.update_layout(hovermode="closest", margin=dict(l=0, r=0, t=28, b=0))
    return fig


def gauge(value: float, target: float, title: str = "", height: int = 200) -> go.Figure:
    ratio = value / target if target else 0
    color = STATUS["ok"] if ratio >= 1 else (STATUS["cerca"] if ratio >= 0.9 else STATUS["bajo"])
    primary, secondary, grid = _ink()
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        gauge=dict(axis=dict(range=[0, max(target * 1.3, value * 1.1)],
                             tickfont=dict(color=secondary, size=10)),
                   bar=dict(color=color, thickness=0.7),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   threshold=dict(line=dict(color=secondary, width=3), value=target)),
        number=dict(font=dict(color=primary, size=26)),
        title=dict(text=title, font=dict(color=secondary, size=12)),
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=height,
                      margin=dict(l=16, r=16, t=40, b=8))
    return fig
