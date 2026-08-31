"""Modo sencillo: una sola pantalla. Subes el archivo y ves el resultado.

Nada que configurar antes de obtener valor. Todo lo que se puede decidir solo,
se decide solo, y se dice claramente qué se hizo.
"""
from __future__ import annotations

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
from utils import fmt_pct, to_datetime_series, to_numeric_series

MAX_INSIGHTS = 5


def _negritas(texto: str) -> str:
    """Convierte **x** en <b>x</b>: dentro de HTML crudo Streamlit no lee markdown."""
    partes = texto.split("**")
    return "".join(p if i % 2 == 0 else f"<b>{p}</b>" for i, p in enumerate(partes))


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
    a, b, c = st.columns(3)
    a.markdown("**📊 Tus números**\n\nVentas, ticket promedio, clientes y márgenes, "
               "calculados solos.")
    b.markdown("**💡 Qué está pasando**\n\nTendencias, temporadas, qué crece y qué se cae, "
               "explicado en español claro.")
    c.markdown("**🔍 Qué está mal**\n\nDuplicados, datos faltantes y errores de captura. "
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


def _encabezado(df):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"# {st.session_state.source}")
        st.caption(f"{len(df):,} registros · {df.shape[1]} columnas")
    with c2:
        st.write("")
        if st.button("Analizar otro archivo", use_container_width=True):
            for k in ("raw", "clean", "prep_log", "mapping", "model", "source"):
                st.session_state[k] = estado.DEFAULTS[k]
            st.session_state.usar_original = False
            st.rerun()


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
                st.rerun()
        return

    c1, c2 = st.columns([3, 1])
    c1.success(f"Corregimos **{len(acciones)} cosas** automáticamente para que los números "
               "cuadren.", icon="🔧")
    with c2:
        st.write("")
        if st.button("Ver datos sin corregir", use_container_width=True):
            st.session_state.usar_original = True
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


def _bloque_kpis(df, mapping):
    catalogo = kpi_mod.compute_catalog(df, mapping)
    propios, _ = kpi_mod.compute_custom(df, st.session_state.custom)
    mostrar = textos.ordenar_kpis(catalogo, 6) + propios[:2]
    if not mostrar:
        return
    kpi_mod.add_period_deltas(mostrar, df, mapping)
    st.markdown("### Tus números")
    for inicio in range(0, len(mostrar), 3):
        cols = st.columns(3)
        for col, k in zip(cols, mostrar[inicio:inicio + 3]):
            delta = (fmt_pct(k.delta) if k.delta is not None and np.isfinite(k.delta) else None)
            col.metric(k.name, k.display(st.session_state.currency), delta=delta,
                       help=(k.help + (f" · {k.delta_label}" if delta else "")))


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
    st.markdown("### Lo más importante")
    for n, h in enumerate(hallazgos[:MAX_INSIGHTS]):
        st.markdown(
            f"<div class='ins ins-{h.kind}'>"
            f"<span class='ins-h'>{ins_mod.ICON[h.kind]} {h.titulo_llano}</span>"
            f"<span>{_negritas(h.texto_llano)}</span></div>",
            unsafe_allow_html=True)
        if h.chart:
            _dibujar(h.chart, df, f"h{n}")


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
        return
    st.markdown("### Tus gráficas")
    for i, (titulo, fig, lectura, atipicos) in enumerate(hechas):
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
    st.markdown("### Llévatelo")
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

    with st.spinner("Preparando tu Excel…"):
        try:
            excel, avisos = _armar_excel(
                df, st.session_state.source, f"{nivel} — {frase}", resumen_txt,
                st.session_state.prep_log, prob_df, kpis_df, hall_df)
        except Exception as e:  # noqa: BLE001
            excel, avisos = None, [f"No se pudo generar el Excel ({type(e).__name__})."]

    c1, c2, c3 = st.columns(3)
    if excel is not None:
        c1.download_button("📗 Descargar Excel limpio", excel,
                           "datos_limpios.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True, type="primary",
                           help="Tus datos ya corregidos, más una hoja con todo lo que "
                                "corregimos y otra con lo que falta revisar.")
    html = deployment.build_html_report(
        source=st.session_state.source, overview=overview, score=score, counts=counts,
        kpis=kpis_mostrar, insights=hallazgos,
        issues_df=quality.issues_table(issues),
        profile_df=profiling.profiles_table(profiles),
        prep_log=st.session_state.prep_log, model_report=st.session_state.model,
        summary=textos.resumen(hallazgos, overview, score), currency=moneda)
    c2.download_button("📄 Reporte para leer o imprimir", html.encode("utf-8"),
                       "reporte.html", "text/html", use_container_width=True,
                       help="Se abre en cualquier navegador.")
    c3.download_button("📋 Datos en CSV", deployment.to_csv_bytes(df),
                       "datos_corregidos.csv", "text/csv", use_container_width=True,
                       help="Todas las filas, sin formato. Útil si el archivo es enorme.")
    for a in avisos:
        st.caption(f"ℹ️ {a}")
    st.caption("El Excel trae cuatro hojas: **Resumen**, **Datos limpios** (con filtros ya "
               "puestos), **Qué corregimos** y **Qué revisar**.")


# ------------------------------------------------------------------ render


def render():
    df = estado.active_df()
    if df is None:
        _pantalla_inicio()
        return

    profiles, issues, score, counts, overview = estado.analyze(df)
    mapping = estado.mapping_actual(profiles)

    _encabezado(df)
    _bloque_limpieza()
    _bloque_confianza(score, counts)

    with st.spinner("Buscando patrones…"):
        hallazgos = ins_mod.generate_insights(df, profiles, mapping, issues)

    st.info(textos.resumen(hallazgos, overview, score).replace("**", ""), icon="📌")
    st.write("")

    _bloque_kpis(df, mapping)
    st.write("")
    _bloque_hallazgos(df, hallazgos)
    st.write("")
    _bloque_graficas(df, profiles, mapping)
    st.write("")
    _bloque_problemas(issues, counts)
    _bloque_columnas(df, profiles, mapping)
    st.write("")
    _bloque_descargas(df, profiles, issues, score, counts, hallazgos, mapping, overview)
    st.write("")
    st.caption("¿Necesitas más control — limpiar paso a paso, definir tus propios KPIs o "
               "entrenar un modelo? Abre el menú lateral (la flecha « » de arriba a la "
               "izquierda) y activa **Modo avanzado**.")
