"""Modo sencillo: una sola pantalla. Subes el archivo y ves el resultado.

Nada que configurar antes de obtener valor. Todo lo que se puede decidir solo,
se decide solo, y se dice claramente qué se hizo.
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

import charts
import deployment
import estado
import explicaciones as expl
import insights as ins_mod
import kpis as kpi_mod
import loader
import profiling
import quality
import textos
from demo import build_demo
from utils import fmt_num, fmt_pct, to_datetime_series, to_numeric_series

MAX_INSIGHTS = 5


def _negritas(texto: str) -> str:
    """Convierte **x** en <b>x</b>: dentro de HTML crudo Streamlit no lee markdown."""
    partes = texto.split("**")
    return "".join(p if i % 2 == 0 else f"<b>{p}</b>" for i, p in enumerate(partes))


def _esc(t) -> str:
    return html.escape(str(t))


def _sec(chip: str, titulo: str, sub: str = ""):
    """Encabezado de sección: etiqueta de color + título, como en un tablero."""
    st.markdown(
        f"<div class='sec'><span class='chip'>{_esc(chip)}</span>"
        f"<span class='tit'>{_esc(titulo)}</span>"
        + (f"<span class='sub'>{_esc(sub)}</span>" if sub else "")
        + "</div>", unsafe_allow_html=True)


MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]


def _periodo(df, mapping, corto: bool = False) -> str:
    """Rango de fechas en español, o '—' si no hay columna de fecha usable."""
    col = mapping.get("fecha")
    if not col or col not in df.columns:
        return "—"
    try:
        f = to_datetime_series(df[col])
        f = f[ins_mod.robust_date_mask(f)].dropna()
        if f.empty:
            return "—"
        a, b = f.min(), f.max()
        if corto:   # cabe en una casilla angosta: "ene 24 – dic 25"
            return (f"{MESES[a.month - 1]} {a.year % 100:02d} – "
                    f"{MESES[b.month - 1]} {b.year % 100:02d}")
        if a.year == b.year:
            return f"{MESES[a.month - 1]} – {MESES[b.month - 1]} {b.year}"
        return f"{MESES[a.month - 1]} {a.year} – {MESES[b.month - 1]} {b.year}"
    except Exception:  # noqa: BLE001
        return "—"


# ------------------------------------------------------------------ inicio


def _pantalla_inicio():
    st.markdown("# Analiza tus datos")
    st.markdown(
        "#### Sube tu archivo y en unos segundos ves qué está pasando en tu negocio.")
    st.write("")

    up = st.file_uploader(
        "Arrastra aquí tu archivo de Excel o CSV",
        type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Ver un ejemplo", use_container_width=True,
                     help="Genera un archivo de ventas de prueba para que veas cómo funciona."):
            estado.cargar(build_demo(), "ejemplo_ventas.csv")
            st.rerun()

    st.write("")
    _sec("Qué obtienes", "En una sola pantalla")
    a, b, c = st.columns(3, gap="medium")
    a.container(border=True).markdown(
        "**📊 Tus números**\n\nVentas, ticket promedio, clientes y márgenes, "
        "calculados solos.")
    b.container(border=True).markdown(
        "**💡 Qué está pasando**\n\nTendencias, temporadas, qué crece y qué se cae, "
        "explicado en español claro.")
    c.container(border=True).markdown(
        "**🔍 Qué está mal**\n\nDuplicados, datos faltantes y errores de captura. "
        "Los corregimos y te decimos qué corregimos.")

    if up is not None:
        mb = len(up.getvalue()) / 1024 ** 2
        aviso = ("Archivo grande: puede tardar un par de minutos." if mb > 8
                 else "Leyendo y analizando tu archivo…")
        try:
            with st.spinner(aviso):
                res = loader.load_bytes(up.getvalue(), up.name)
                if len(res.df) == 0:
                    st.error("El archivo no trae ninguna fila de datos.")
                    st.caption("Revisa que la tabla tenga encabezados en la primera fila.")
                    return
                estado.cargar(res.df, up.name)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            que, como = loader.mensaje_amigable(e, up.name)
            st.error(que, icon="🚫")
            st.markdown(f"**Qué puedes hacer:** {como}")
            with st.expander("Detalle técnico"):
                st.code(f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ bloques


def _topbar(df):
    """Barra fija arriba: qué app es, qué archivo se está viendo, y salida."""
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            "<div class='topbar'><span class='marca'>📊 Analiza tus datos</span>"
            "<span class='sep'>|</span>"
            f"<span class='arch'>{_esc(st.session_state.source)}</span>"
            f"<span class='meta'>{len(df):,} registros · {df.shape[1]} columnas</span>"
            "</div>", unsafe_allow_html=True)
    with c2:
        if st.button("Analizar otro archivo", use_container_width=True):
            for k in ("raw", "clean", "prep_log", "mapping", "model", "source"):
                st.session_state[k] = estado.DEFAULTS[k]
            st.session_state.usar_original = False
            st.session_state.pop("_excel_listo", None)
            st.session_state.pop("_excel_avisos", None)
            st.rerun()


def _hero(df, mapping, score, counts, principal, moneda):
    """El panel destacado: el número que más importa, con su contexto al pie."""
    nivel, _, emoji, _ = estado.confianza(score, counts)
    if principal is not None:
        etiqueta = "El número que más importa"
        valor = principal.display(moneda)
        nombre = principal.help
    else:
        etiqueta = "Tu archivo"
        valor = f"{len(df):,}"
        nombre = "registros analizados y listos para revisar."

    # el periodo va en el encabezado de la sección: aquí no cabe sin cortarse
    tiles = [("Registros", f"{len(df):,}"),
             ("Columnas", f"{df.shape[1]}"),
             ("Confianza", f"{emoji} {nivel}")]
    htm = "".join(f"<div class='tile'><span class='t'>{_esc(t)}</span>"
                  f"<span class='v'>{_esc(v)}</span></div>" for t, v in tiles)
    st.markdown(
        f"<div class='hero'><span class='et'>{_esc(etiqueta)}</span>"
        f"<div class='val'>{_esc(valor)}</div>"
        f"<div class='nom'>{_esc(nombre)}</div>"
        f"<div class='tiles'>{htm}</div></div>", unsafe_allow_html=True)


def _bloque_limpieza():
    log = st.session_state.prep_log
    if not log or st.session_state.clean is None:
        return
    acciones = [l for l in log if not l.startswith("Resultado:")]
    if not acciones:
        return

    if st.session_state.usar_original:
        c1, c2 = st.columns([3, 1])
        c1.warning("Estás viendo los datos **sin corregir**, tal como venían en el archivo.",
                   icon="⚠️")
        with c2:
            st.write("")
            if st.button("Volver a los corregidos", use_container_width=True):
                st.session_state.usar_original = False
                estado.olvidar_excel()
                st.rerun()
        return

    c1, c2 = st.columns([3, 1])
    c1.success(f"Corregimos **{len(acciones)} cosas** automáticamente para que los números "
               "cuadren.", icon="🔧")
    with c2:
        st.write("")
        if st.button("Ver datos sin corregir", use_container_width=True):
            st.session_state.usar_original = True
            estado.olvidar_excel()
            st.rerun()
    with st.expander("¿Qué corregimos exactamente?"):
        for l in acciones:
            st.markdown(f"- {l}")
        st.caption("Nada de esto modifica tu archivo original: la corrección vive solo aquí.")


def _bloque_confianza(score, counts):
    nivel, clase, emoji, frase = estado.confianza(score, counts)
    st.markdown(
        f"<div class='sem {clase}'><div class='punto'>{emoji}</div>"
        f"<div class='txt'><b>Confianza en los datos: {nivel}</b><span>{frase}</span></div></div>",
        unsafe_allow_html=True)


MODOS_KPI = {
    "recomendados": "Los que la app recomiende",
    "propios": "Yo defino los míos",
    "ninguno": "No quiero indicadores",
}

VACIO_KPI = {
    "ninguno": ("**Sin indicadores en pantalla.** El panel de la izquierda, los hallazgos "
                "y las gráficas siguen funcionando igual."),
    "propios": ("**Todavía no has creado ninguno.** Ármalo aquí abajo: eliges un nombre, "
                "qué calcular y de qué columna. No hay que escribir fórmulas."),
    "recomendados": ("**No pudimos calcular indicadores con estas columnas.** Dinos cuál es "
                     "el importe en la pestaña **Calidad de datos** y vuelve a intentar."),
}


def _pinta_kpis(kpis, moneda, por_fila: int = 3):
    """Cada indicador en su propia tarjeta, en rejilla."""
    for inicio in range(0, len(kpis), por_fila):
        cols = st.columns(por_fila)
        for col, k in zip(cols, kpis[inicio:inicio + por_fila]):
            delta = (fmt_pct(k.delta) if k.delta is not None and np.isfinite(k.delta) else None)
            with col, st.container(border=True):
                st.metric(k.name, k.display(moneda), delta=delta,
                          help=(k.help + (f" · {k.delta_label}" if delta else "")))
                if k.status:
                    marca = {"ok": "🟢 Cumples la meta", "cerca": "🟡 Cerca de la meta",
                             "bajo": "🔴 Debajo de la meta"}[k.status]
                    st.caption(f"{marca} ({moneda if k.fmt == kpi_mod.FMT_MONEY else ''}"
                               f"{fmt_num(k.target)})")


def _constructor_kpi(df, profiles, mapping):
    """Arma un KPI eligiendo de menús, sin escribir fórmulas."""
    columnas = list(df.columns)
    numericas = profiling.suggest_metric_columns(profiles) or columnas
    # lo primero que alguien quiere sumar es su importe: que salga por defecto
    for preferida in (mapping.get("costo"), mapping.get("ingreso")):
        if preferida in numericas:
            numericas = [preferida] + [c for c in numericas if c != preferida]
    with st.form("nuevo_kpi_simple", clear_on_submit=True):
        c1, c2 = st.columns([1, 1])
        nombre = c1.text_input("¿Cómo se llama tu indicador?",
                               placeholder="Margen, Ventas de mayoreo, Tasa de cancelación…")
        op = c2.selectbox("¿Qué quieres calcular?", list(kpi_mod.OPERACIONES),
                          format_func=lambda o: kpi_mod.OPERACIONES[o][0])
        _, n_cols, admite_filtro, fmt_sug, _ = kpi_mod.OPERACIONES[op]

        col_a = col_b = None
        if n_cols >= 1:
            cc = st.columns(2 if n_cols >= 2 else 1)
            col_a = cc[0].selectbox("¿De qué columna?", numericas if op != "distintos" else columnas)
            if n_cols >= 2:
                col_b = cc[1].selectbox(kpi_mod.ETIQUETA_B.get(op, "Segunda columna"),
                                        columnas if op == "promedio_por" else numericas)

        filtro_col = filtro_val = None
        if admite_filtro:
            categoricas = ["(sin condición)"] + profiling.suggest_dimension_columns(profiles)
            fc = st.selectbox(
                "¿Solo cuando se cumpla algo?" if op != "porcentaje" else "¿Qué condición?",
                categoricas)
            if fc != "(sin condición)":
                filtro_col = fc
                valores = sorted(df[fc].dropna().astype(str).unique())[:200]
                filtro_val = st.selectbox(f"«{fc}» igual a", valores)

        c3, c4 = st.columns(2)
        fmt = c3.selectbox("¿Cómo se muestra?",
                           [kpi_mod.FMT_MONEY, kpi_mod.FMT_NUM, kpi_mod.FMT_PCT, kpi_mod.FMT_INT],
                           index=[kpi_mod.FMT_MONEY, kpi_mod.FMT_NUM, kpi_mod.FMT_PCT,
                                  kpi_mod.FMT_INT].index(fmt_sug),
                           format_func=lambda f: {"moneda": "Como dinero", "numero": "Como número",
                                                  "porcentaje": "Como porcentaje",
                                                  "entero": "Como cantidad entera"}[f])
        meta = c4.number_input("Meta (opcional, 0 = sin meta)", value=0.0, step=1.0,
                               help="Si pones una meta, aparece un semáforo debajo del número.")

        if st.form_submit_button("Agregar indicador", type="primary"):
            if not nombre.strip():
                st.error("Ponle un nombre a tu indicador.")
                return
            try:
                formula = kpi_mod.construir_formula(op, col_a, col_b, filtro_col, filtro_val)
                kpi_mod.FormulaEvaluator(df).evaluate(formula)
            except kpi_mod.FormulaError as e:
                st.error(f"No se pudo calcular: {e}")
                return
            st.session_state.custom.append(kpi_mod.CustomKPI(
                nombre.strip(), formula, fmt, meta or None,
                kpi_mod.descripcion_kpi(op, col_a, col_b, filtro_col, filtro_val)))
            st.rerun()


def _selector_kpis():
    """El interruptor de tres posiciones, arriba de la rejilla."""
    st.radio("¿Qué indicadores quieres ver?", list(MODOS_KPI),
             format_func=lambda m: MODOS_KPI[m], horizontal=True, key="kpi_modo",
             label_visibility="collapsed")
    return st.session_state.kpi_modo


def _kpis_a_mostrar(df, mapping):
    """(indicadores que se pintan, catálogo completo, errores de los propios).

    Se calcula antes de dibujar nada porque el panel destacado necesita saber
    cuál es el indicador principal.
    """
    modo = st.session_state.kpi_modo
    if modo == "ninguno":
        return [], [], []

    catalogo = kpi_mod.compute_catalog(df, mapping)
    if modo == "recomendados":
        if not catalogo:
            return [], [], []
        nombres = [k.name for k in catalogo]
        elegidos = st.session_state.selected_kpis
        if elegidos is None:
            elegidos = [k.name for k in textos.ordenar_kpis(catalogo, 6)]
        elegidos = [n for n in elegidos if n in nombres] or nombres[:6]
        st.session_state.selected_kpis = elegidos
        orden = {k.name: i for i, k in enumerate(textos.ordenar_kpis(catalogo, len(catalogo)))}
        mostrar = sorted([k for k in catalogo if k.name in elegidos],
                         key=lambda k: orden.get(k.name, 99))
        kpi_mod.add_period_deltas(mostrar, df, mapping)
        return mostrar, catalogo, []

    propios, errores = kpi_mod.compute_custom(df, st.session_state.custom)
    if propios:
        kpi_mod.add_period_deltas(propios, df, mapping)
    return propios, catalogo, errores


def _kpis_extras(df, profiles, mapping, catalogo, errores, hay_propios):
    """Lo que se ajusta pocas veces: qué indicadores ver, o crear los tuyos."""
    modo = st.session_state.kpi_modo
    if modo == "ninguno":
        return

    if modo == "recomendados":
        if not catalogo:
            return
        nombres = [k.name for k in catalogo]
        with st.expander("¿Cuáles quieres ver? (elegimos estos por ti)"):
            elegidos = st.multiselect(
                "Indicadores", nombres,
                default=[n for n in (st.session_state.selected_kpis or []) if n in nombres],
                label_visibility="collapsed")
            st.caption("Solo aparecen los que tus datos permiten calcular. "
                       "Pasa el cursor sobre el ⓘ de cada uno para ver cómo se obtiene.")
        if elegidos and elegidos != st.session_state.selected_kpis:
            st.session_state.selected_kpis = elegidos
            st.rerun()
        return

    # --- modo "propios"
    for e in errores:
        st.error(e.replace("**", ""))
    if hay_propios:
        quitar = st.selectbox("Quitar un indicador",
                              ["(ninguno)"] + [k.name for k in st.session_state.custom],
                              key="quitar_kpi")
        if quitar != "(ninguno)" and st.button("Quitar"):
            st.session_state.custom = [k for k in st.session_state.custom if k.name != quitar]
            st.rerun()

    with st.expander("➕ Crear un indicador", expanded=not hay_propios):
        _constructor_kpi(df, profiles, mapping)


def _dibujar(spec, df, key):
    try:
        t = spec["type"]
        if t == "line":
            fig = charts.line_time(pd.DataFrame({"x": spec["x"], "y": spec["y"]}),
                                   "x", "y", title="", height=240)
        elif t == "bar":
            fig = charts.bar_ranked(spec["labels"], spec["values"], "",
                                    horizontal=len(spec["labels"]) > 5,
                                    highlight=spec.get("highlight", 0), height=260,
                                    prefijo=spec.get("prefijo", ""))
        elif t == "box":
            d = df[[spec["dim"], spec["value"]]].copy()
            d[spec["value"]] = to_numeric_series(d[spec["value"]])
            fig = charts.box_by_group(d.dropna(), spec["dim"], spec["value"], "", height=260)
        elif t == "hist":
            s = to_numeric_series(df[spec["col"]]).dropna()
            fig = charts.histogram(s, "", spec["col"], 240, median=float(s.median()))
        elif t == "scatter":
            d = df[[spec["x"], spec["y"]]].apply(to_numeric_series).dropna()
            fig = charts.scatter(d, spec["x"], spec["y"], title="", height=260)
        else:
            return
        st.plotly_chart(fig, use_container_width=True, key=key,
                        config={"displayModeBar": False})
        lectura = expl.como_leer(t)
        if lectura:
            st.markdown(f"<div class='lect'>{lectura}</div>", unsafe_allow_html=True)
    except Exception:  # noqa: BLE001
        pass


def _bloque_hallazgos(df, hallazgos):
    if not hallazgos:
        return
    _sec("Hallazgos", "Lo más importante",
         f"{min(len(hallazgos), MAX_INSIGHTS)} de {len(hallazgos)}")
    vista = list(enumerate(hallazgos[:MAX_INSIGHTS]))

    def tarjeta(n, h, plano=False):
        st.markdown(
            f"<div class='ins ins-{h.kind}{' ins-plano' if plano else ''}'>"
            f"<span class='ins-h'>{ins_mod.ICON[h.kind]} {_esc(h.titulo_llano)}</span>"
            f"<span>{_negritas(_esc(h.texto_llano))}</span></div>",
            unsafe_allow_html=True)
        if h.chart:
            _dibujar(h.chart, df, f"h{n}")

    # Los que traen gráfica ocupan la fila entera; si compartieran fila con uno
    # de puro texto, al lado quedaría un hueco blanco del alto de la gráfica.
    pendientes: list = []
    for n, h in vista:
        if h.chart:
            for fila in range(0, len(pendientes), 2):
                cols = st.columns(2, gap="medium")
                for col, (m, g) in zip(cols, pendientes[fila:fila + 2]):
                    with col:
                        tarjeta(m, g)
            pendientes = []
            with st.container(border=True):
                tarjeta(n, h, plano=True)
            st.write("")
        else:
            pendientes.append((n, h))
    for fila in range(0, len(pendientes), 2):
        cols = st.columns(2, gap="medium")
        for col, (m, g) in zip(cols, pendientes[fila:fila + 2]):
            with col:
                tarjeta(m, g)


def _bloque_graficas(df, profiles, mapping):
    date_cols = profiling.suggest_date_columns(profiles)
    num_cols = profiling.suggest_metric_columns(profiles)
    dim_cols = profiling.suggest_dimension_columns(profiles)
    if not num_cols:
        return
    metrica = mapping.get("ingreso") if mapping.get("ingreso") in num_cols else num_cols[0]

    moneda = (st.session_state.currency
              if mapping.get("ingreso") == metrica else None)
    hechas = []   # (título, figura, lectura, [atípicos])

    fecha = mapping.get("fecha") if mapping.get("fecha") in date_cols else (
        date_cols[0] if date_cols else None)
    if fecha:
        f = to_datetime_series(df[fecha])
        d = df.assign(_f=f, _v=to_numeric_series(df[metrica]))[ins_mod.robust_date_mask(f)]
        if len(d) > 3:
            span = (d["_f"].max() - d["_f"].min()).days
            freq, etq = (("D", "día") if span <= 60 else
                         ("W", "semana") if span <= 365 else ("ME", "mes"))
            serie = d.set_index("_f")["_v"].resample(freq).sum()
            # el último periodo casi siempre está incompleto y dibuja una caída falsa
            if len(serie) > 2 and serie.index[-1] > d["_f"].max():
                serie = serie.iloc[:-1]
            hechas.append((
                f"Cómo va {metrica} por {etq}",
                charts.line_time(pd.DataFrame({"x": serie.index, "y": serie.values}),
                                 "x", "y", title="", ylab=metrica, height=300),
                expl.leer_serie(serie, freq, etq, metrica, moneda),
                expl.atipicos_serie(serie, df, fecha, metrica, mapping, freq, etq, moneda)))

    dim = mapping.get("segmento") or mapping.get("producto")
    if dim not in dim_cols:
        dim = dim_cols[0] if dim_cols else None
    if dim:
        agg = (df.assign(_v=to_numeric_series(df[metrica])).groupby(dim)["_v"].sum()
                 .sort_values(ascending=False))
        if len(agg) > 1:
            etq, vals, n_resto = charts.top_con_otros(agg, 8)
            hechas.append((
                f"{metrica} por {dim.lower()}",
                charts.bar_ranked(etq, vals, "", metrica, height=300,
                                  prefijo=moneda or ""),
                expl.leer_ranking(agg, dim, metrica, moneda, n_resto),
                expl.atipicos_ranking(agg, dim, moneda)))

    if not hechas:
        st.info("Con estas columnas no alcanza para dibujar una gráfica útil. "
                "Revisa en **Calidad de datos** si interpretamos bien tus columnas.")
        return
    _sec("Gráficas", "Cómo se ve tu operación", f"{len(hechas)} vistas")
    for i, (titulo, fig, lectura, atipicos) in enumerate(hechas):
        with st.container(border=True):
            st.markdown(f"**{titulo}**")
            st.plotly_chart(fig, use_container_width=True, key=f"g{i}",
                            config={"displayModeBar": False})
            if lectura:
                st.markdown(f"<div class='lect'>{_negritas(lectura)}</div>",
                            unsafe_allow_html=True)
            if atipicos:
                st.markdown("<div class='atip-t'>Valores fuera de lo normal</div>",
                            unsafe_allow_html=True)
                for a in atipicos:
                    st.markdown(f"<div class='atip'>{_negritas(a)}</div>",
                                unsafe_allow_html=True)
                st.caption("Estas son pistas de dónde salió el número, no la causa. "
                           "La causa la sabes tú: una campaña, un cliente grande, un cierre "
                           "de mes o un error de captura.")
        st.write("")


def _bloque_problemas(issues, counts):
    if not issues:
        st.success("No encontramos ningún problema en tus datos.", icon="✅")
        return
    corregidos = [i for i in issues if textos.es_corregible(i)]
    pendientes = [i for i in issues if not textos.es_corregible(i)]
    aplicado = st.session_state.clean is not None and not st.session_state.usar_original

    resumen = (f"{len(pendientes)} cosa(s) por revisar" if pendientes
               else "todo en orden")
    with st.expander(f"🔍 Qué revisar en tus datos — {resumen}", expanded=bool(
            [i for i in pendientes if i.severity == "crítico"])):
        if pendientes:
            st.markdown("**Esto no lo podemos arreglar solos — necesita tu criterio:**")
            for i in pendientes[:14]:
                icono = {"crítico": "🔴", "advertencia": "🟡"}.get(i.severity, "🔵")
                st.markdown(f"<div class='fix'>{icono} {_negritas(textos.explicar(i))}</div>",
                            unsafe_allow_html=True)
            if len(pendientes) > 14:
                st.caption(f"…y {len(pendientes) - 14} más. El reporte descargable los trae todos.")
        if corregidos:
            st.markdown("")
            st.markdown("**Esto ya lo corregimos:**" if aplicado
                        else "**Esto lo podemos corregir (actívalo arriba):**")
            for i in corregidos[:12]:
                st.markdown(
                    f"<div class='fix'>{'✅' if aplicado else '⬜️'} "
                    f"{_negritas(textos.explicar(i))}</div>", unsafe_allow_html=True)


def _bloque_columnas(df, profiles, mapping):
    with st.expander("⚙️ ¿Interpretamos bien tus columnas?"):
        st.caption("De esto dependen los números y los hallazgos. Si algo está mal, corrígelo "
                   "aquí y todo se recalcula.")
        opciones = ["(ninguna)"] + list(df.columns)
        etiquetas = {
            "fecha": "Fecha de la venta o del registro",
            "ingreso": "Importe / venta / ingreso",
            "costo": "Costo",
            "cantidad": "Unidades o piezas",
            "cliente": "Cliente",
            "producto": "Producto o servicio",
            "segmento": "Canal, sucursal, vendedor o categoría",
        }
        nuevo = dict(mapping)
        cols = st.columns(2)
        for i, (slot, etiqueta) in enumerate(etiquetas.items()):
            with cols[i % 2]:
                actual = mapping.get(slot)
                idx = opciones.index(actual) if actual in opciones else 0
                elegido = st.selectbox(etiqueta, opciones, index=idx, key=f"s_{slot}")
                nuevo[slot] = None if elegido == "(ninguna)" else elegido
        if nuevo != mapping:
            st.session_state.mapping = {**mapping, **nuevo}
            st.rerun()


@st.cache_data(show_spinner=False, max_entries=3)
def _armar_excel(df, source, confianza, resumen, prep_log, problemas, kpis, hallazgos):
    return deployment.build_excel(
        df, source=source, confianza=confianza, resumen=resumen, prep_log=prep_log,
        problemas=problemas, kpis=kpis, hallazgos=hallazgos)


def _bloque_descargas(df, profiles, issues, score, counts, hallazgos, mapping, overview):
    _sec("Descargas", "Llévate el resultado")
    catalogo = kpi_mod.compute_catalog(df, mapping)
    propios, _ = kpi_mod.compute_custom(df, st.session_state.custom)
    kpis_mostrar = textos.ordenar_kpis(catalogo, 8) + propios
    moneda = st.session_state.currency
    nivel, _, _, frase = estado.confianza(score, counts)
    resumen_txt = textos.resumen(hallazgos, overview, score).replace("**", "")

    kpis_df = pd.DataFrame([{"KPI": k.name, "Valor": k.display(moneda)}
                            for k in kpis_mostrar])
    hall_df = pd.DataFrame([{"Hallazgo": h.titulo_llano,
                             "Detalle": h.texto_llano.replace("**", "")}
                            for h in hallazgos])
    prob_df = pd.DataFrame([
        {"Severidad": {"crítico": "Revisar ya", "advertencia": "Revisar"}.get(i.severity, "Aviso"),
         "Qué pasa": textos.explicar(i).replace("**", "")}
        for i in issues])

    # El Excel se arma solo cuando lo piden: con archivos grandes tarda y consume
    # memoria, y no tiene por qué costarle eso a quien solo vino a ver sus números.
    avisos: list[str] = []
    c1, c2, c3 = st.columns(3, gap="medium")
    tarjeta_1, tarjeta_2, tarjeta_3 = (c1.container(border=True),
                                       c2.container(border=True),
                                       c3.container(border=True))
    tarjeta_1.markdown("**📗 Excel limpio**")
    tarjeta_1.caption("Tus datos corregidos y con filtros, más las hojas de qué "
                      "corregimos y qué revisar.")
    tarjeta_2.markdown("**📄 Reporte para leer**")
    tarjeta_2.caption("Todo lo de esta pantalla en una página que se abre en cualquier "
                      "navegador y se puede imprimir.")
    tarjeta_3.markdown("**📋 Datos en CSV**")
    tarjeta_3.caption("Todas las filas corregidas, sin formato. Útil si el archivo es "
                      "enorme o si lo vas a cargar en otro sistema.")

    if st.session_state.get("_excel_listo"):
        tarjeta_1.download_button(
            "Descargar Excel limpio", st.session_state["_excel_listo"],
            "datos_limpios.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True, type="primary")
        avisos = st.session_state.get("_excel_avisos", [])
    elif tarjeta_1.button("📗 Preparar Excel limpio", use_container_width=True,
                          type="primary"):
        with st.spinner("Armando tu Excel… (puede tardar si el archivo es grande)"):
            try:
                excel, avisos = _armar_excel(
                    df, st.session_state.source, f"{nivel} — {frase}", resumen_txt,
                    st.session_state.prep_log, prob_df, kpis_df, hall_df)
                st.session_state["_excel_listo"] = excel
                st.session_state["_excel_avisos"] = avisos
            except MemoryError:
                st.session_state["_excel_avisos"] = [
                    "El archivo es demasiado grande para armar el Excel con la memoria "
                    "disponible. Usa la descarga en CSV: trae exactamente los mismos datos "
                    "corregidos y la abre Excel sin problema."]
            except Exception as e:  # noqa: BLE001
                st.session_state["_excel_avisos"] = [
                    f"No se pudo generar el Excel ({type(e).__name__}). Usa la descarga en CSV, "
                    "que trae los mismos datos corregidos."]
        st.rerun()
    reporte = deployment.build_html_report(
        source=st.session_state.source, overview=overview, score=score, counts=counts,
        kpis=kpis_mostrar, insights=hallazgos,
        issues_df=quality.issues_table(issues),
        profile_df=profiling.profiles_table(profiles),
        prep_log=st.session_state.prep_log, model_report=st.session_state.model,
        summary=textos.resumen(hallazgos, overview, score), currency=moneda)
    tarjeta_2.download_button("📄 Reporte para leer o imprimir", reporte.encode("utf-8"),
                              "reporte.html", "text/html", use_container_width=True)
    tarjeta_3.download_button("📋 Datos en CSV", deployment.to_csv_bytes(df),
                              "datos_corregidos.csv", "text/csv", use_container_width=True)
    for a in avisos:
        st.caption(f"ℹ️ {a}")


# ------------------------------------------------------------------ render


def render():
    df = estado.active_df()
    if df is None:
        _pantalla_inicio()
        return

    profiles, issues, score, counts, overview = estado.analyze(df)
    mapping = estado.mapping_actual(profiles)
    moneda = st.session_state.currency

    _topbar(df)

    with st.spinner("Buscando patrones…"):
        hallazgos = ins_mod.generate_insights(df, profiles, mapping, issues)

    pendientes = len([i for i in issues if not textos.es_corregible(i)])
    t_resumen, t_graficas, t_calidad, t_descargas = st.tabs(
        ["Resumen", "Gráficas", f"Calidad de datos ({pendientes})", "Descargas"])

    # ---------------------------------------------------------------- resumen
    with t_resumen:
        _bloque_limpieza()
        st.info(textos.resumen(hallazgos, overview, score).replace("**", ""), icon="📌")

        _sec("Panorama", "Tus números", _periodo(df, mapping))
        _selector_kpis()
        mostrar, catalogo, errores = _kpis_a_mostrar(df, mapping)
        principal = mostrar[0] if mostrar else None

        c_hero, c_kpis = st.columns([1, 2.15], gap="medium")
        with c_hero:
            _hero(df, mapping, score, counts, principal, moneda)
        with c_kpis:
            if mostrar:
                _pinta_kpis(mostrar, moneda, por_fila=2)
            else:
                st.container(border=True).markdown(VACIO_KPI[st.session_state.kpi_modo])
        _kpis_extras(df, profiles, mapping, catalogo, errores, bool(mostrar))

        _bloque_hallazgos(df, hallazgos)

    # --------------------------------------------------------------- gráficas
    with t_graficas:
        _bloque_graficas(df, profiles, mapping)

    # ---------------------------------------------------------------- calidad
    with t_calidad:
        _sec("Calidad", "Qué tan confiables son estos números")
        _bloque_confianza(score, counts)
        st.write("")
        _bloque_problemas(issues, counts)
        _bloque_columnas(df, profiles, mapping)

    # -------------------------------------------------------------- descargas
    with t_descargas:
        _bloque_descargas(df, profiles, issues, score, counts, hallazgos, mapping, overview)

    st.write("")
    st.caption("¿Necesitas más control — limpiar paso a paso, definir tus propios KPIs o "
               "entrenar un modelo? Abre el menú lateral (la flecha « » de arriba a la "
               "izquierda) y activa **Modo avanzado**.")
