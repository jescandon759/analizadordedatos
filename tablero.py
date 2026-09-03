"""El tablero de gráficas: rejilla de tarjetas y vista a detalle.

Dos ideas:

1. Las gráficas se ven juntas, chicas, en una rejilla — no apiladas una debajo
   de otra. De un vistazo se ve todo el negocio.
2. Se le pica a cualquiera (a la gráfica misma o a su botón) y se abre en
   grande, con su lectura, sus valores raros, sus cifras y la tabla de datos
   que hay detrás.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
import streamlit as st

import charts
import deployment
import estado
import explicaciones as expl
import insights as ins_mod
import profiling
from utils import fmt_num, to_datetime_series, to_numeric_series

MAX_VISTAS = 6
ALTO_CHICA = 215
ALTO_GRANDE = 430


@dataclass
class Vista:
    id: str
    titulo: str
    tipo: str                                  # line | bar | hist | box | scatter
    figura: Callable[[int], object]            # alto -> go.Figure
    resumen: str = ""                          # una línea para la tarjeta chica
    lectura: str = ""
    atipicos: list[str] = field(default_factory=list)
    cifras: list[tuple[str, str]] = field(default_factory=list)
    tabla: pd.DataFrame | None = None
    nota_tabla: str = ""


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
                nota_tabla=f"El total de {metrica} en cada {etq}."))

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
                    "Registros": serie.values})))

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
            nota_tabla="La tabla trae todas las categorías, no solo las del top."))

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
                        "pocos registros enormes jalando el promedio.")))

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
                nota_tabla="«Típico» es la mediana: la mitad queda arriba y la mitad abajo."))

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
                            "cause la otra.")))

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


def _rejilla(vistas: list[Vista]):
    gen = st.session_state.get("_gen", 0)
    st.caption("Pícale a cualquier gráfica —o a su botón— para abrirla en grande "
               "con su explicación, sus cifras y los datos de atrás.")
    for fila in range(0, len(vistas), 2):
        cols = st.columns(2, gap="medium")
        for col, v in zip(cols, vistas[fila:fila + 2]):
            with col, st.container(border=True):
                st.markdown(f"**{v.titulo}**")
                if _pintar(v.figura(ALTO_CHICA), f"m_{v.id}_{gen}", clickable=True):
                    _abrir(v.id)
                    st.rerun()
                if v.resumen:
                    st.markdown(f"<div class='lect'>{v.resumen}</div>",
                                unsafe_allow_html=True)
                if st.button("🔍 Ver a detalle", key=f"b_{v.id}_{gen}",
                             use_container_width=True):
                    _abrir(v.id)
                    st.rerun()


def _detalle(v: Vista):
    negritas = estado.negritas
    gen = st.session_state.get("_gen", 0)
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("← Volver al tablero", use_container_width=True):
            _cerrar()
            st.rerun()
    c2.markdown(f"### {v.titulo}")

    with st.container(border=True):
        _pintar(v.figura(ALTO_GRANDE), f"g_{v.id}_{gen}", clickable=False)

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
    vistas = construir_vistas(df, profiles, mapping, moneda)
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
    _rejilla(vistas)
