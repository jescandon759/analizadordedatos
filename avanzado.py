"""Modo avanzado: las seis fases de CRISP-DM, una por una.

Es literalmente la misma lógica que usa el modo sencillo, pero con cada decisión
expuesta: mapeo de columnas, elección de reparaciones, modelado y despliegue.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import charts
import deployment
import estado
import insights as ins_mod
import kpis as kpi_mod
import loader
import prep
import profiling
import quality
from demo import build_demo
from estado import active_df, analyze, goto, reset_downstream
from modeling import (
    ANOMALIAS, CLASIFICACION, CLUSTERING, FORECAST, REGRESION,
    infer_task, run_anomalies, run_clustering, run_forecast, run_supervised,
)
from utils import fmt_num, fmt_pct, to_datetime_series, to_numeric_series


def render():
    labels = [p[1] for p in estado.PHASES]
    keys = [p[0] for p in estado.PHASES]
    with st.sidebar:
        st.markdown("**Fases CRISP-DM**")
        st.radio("Fase", labels, key="nav", label_visibility="collapsed")
    phase = keys[labels.index(st.session_state.nav)]

    # ================================================================== 0. DATOS
    if phase == "datos":
        st.title("Carga de datos")
        st.caption("CSV, TSV o Excel. La app detecta la codificación, el separador, la hoja y "
                   "la fila de encabezado, incluso si el archivo viene desordenado.")

        c1, c2 = st.columns([2, 1])
        with c1:
            up = st.file_uploader("Arrastra tu archivo", type=["csv", "tsv", "txt", "xlsx", "xlsm", "xls"],
                                  label_visibility="collapsed")
        with c2:
            st.write("")
            if st.button("🎲 Usar datos de ejemplo", use_container_width=True,
                         help="Genera un archivo de ventas con errores típicos para probar la app."):
                reset_downstream()
                st.session_state.raw = build_demo()
                st.session_state.source = "ventas_demo.csv (generado)"
                st.session_state.meta = None
                st.rerun()

        if up is not None:
            raw_bytes = up.getvalue()
            sheet = None
            if up.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                try:
                    hojas = loader.list_excel_sheets(raw_bytes)
                    if len(hojas) > 1:
                        sheet = st.selectbox("Hoja de cálculo", hojas)
                except Exception as e:
                    que, como = loader.mensaje_amigable(e, up.name)
                    st.error(que, icon="🚫")
                    st.markdown(f"**Qué puedes hacer:** {como}")
                    st.stop()
            with st.expander("Opciones avanzadas de lectura"):
                manual_header = st.number_input(
                    "Fila de encabezado (0 = automático·primera fila)", min_value=0, max_value=50, value=0)
                sep_choice = st.selectbox("Separador (solo CSV)",
                                          ["Automático", ",", ";", "tabulador", "|"])
            try:
                sep = {"Automático": None, "tabulador": "\t"}.get(sep_choice, sep_choice)
                res = loader.load_bytes(raw_bytes, up.name, sheet=sheet,
                                        header_row=manual_header if manual_header else None,
                                        separator=sep)
            except Exception as e:
                que, como = loader.mensaje_amigable(e, up.name)
                st.error(que, icon="🚫")
                st.markdown(f"**Qué puedes hacer:** {como}")
                with st.expander("Detalle técnico"):
                    st.code(f"{type(e).__name__}: {e}")
                st.stop()

            if st.session_state.source != f"{up.name}|{sheet}":
                pass
            st.success(f"Leído: **{len(res.df):,} filas × {res.df.shape[1]} columnas**", icon="✅")
            detalles = [f"Tipo: {res.kind}"]
            if res.encoding:
                detalles.append(f"Codificación: {res.encoding}")
            if res.separator:
                detalles.append(f"Separador: {res.separator!r}")
            if res.sheet:
                detalles.append(f"Hoja: {res.sheet}")
            st.caption(" · ".join(detalles))
            for w in res.warnings:
                st.warning(w, icon="⚠️")

            st.dataframe(res.df.head(50), use_container_width=True, height=340)

            if st.button("Usar este archivo →", type="primary"):
                reset_downstream()
                st.session_state.raw = res.df
                st.session_state.source = up.name + (f" · {res.sheet}" if res.sheet else "")
                st.session_state.meta = res
                goto("negocio")
                st.rerun()

        elif st.session_state.raw is not None:
            st.divider()
            st.subheader("Datos cargados")
            st.caption(st.session_state.source)
            st.dataframe(active_df().head(50), use_container_width=True, height=340)


    # ------------------------------------------------------------------ guardas
    elif st.session_state.raw is None:
        st.title("Analizador CRISP-DM")
        st.info("Primero carga un archivo en la sección **📁 Datos**.", icon="📁")
        if st.button("Ir a la carga de datos", type="primary"):
            goto("datos")
            st.rerun()


    # ============================================================ 1. NEGOCIO/KPIs
    elif phase == "negocio":
        df = active_df()
        profiles, issues, score, counts, overview = analyze(df)
        st.title("Fase 1 — Comprensión del negocio")
        st.caption("Antes de analizar hay que decir qué significa cada columna y qué se quiere medir. "
                   "Esto es lo que separa un análisis de un montón de gráficas.")

        if not st.session_state.mapping:
            st.session_state.mapping = kpi_mod.suggest_mapping(profiles)

        st.subheader("Mapeo de columnas de negocio")
        st.caption("La app propuso un mapeo automático leyendo los nombres y el contenido. "
                   "Corrígelo: de aquí dependen todos los KPIs y los insights.")
        opciones = ["(ninguna)"] + list(df.columns)
        cols = st.columns(3)
        nuevo = {}
        for i, (slot, desc) in enumerate(kpi_mod.SLOTS.items()):
            with cols[i % 3]:
                actual = st.session_state.mapping.get(slot)
                idx = opciones.index(actual) if actual in opciones else 0
                nuevo[slot] = st.selectbox(slot.capitalize(), opciones, index=idx, help=desc,
                                           key=f"map_{slot}")
        st.session_state.mapping = {k: (None if v == "(ninguna)" else v) for k, v in nuevo.items()}
        mapping = st.session_state.mapping

        if not any(mapping.values()):
            st.warning("Sin al menos una columna mapeada, los KPIs y varios insights no se pueden "
                       "calcular.", icon="⚠️")

        st.divider()
        st.subheader("KPIs del catálogo")
        catalogo = kpi_mod.compute_catalog(df, mapping)
        if not catalogo:
            st.info("Mapea al menos la columna de importe para ver KPIs.")
        else:
            nombres = [k.name for k in catalogo]
            if st.session_state.selected_kpis is None:
                st.session_state.selected_kpis = nombres[:8]
            sel = st.multiselect("¿Cuáles quieres en el dashboard?", nombres,
                                 default=[n for n in st.session_state.selected_kpis if n in nombres],
                                 help="Opcional. Si no eliges nada, se muestran todos los disponibles.")
            st.session_state.selected_kpis = sel or nombres
            mostrados = [k for k in catalogo if k.name in st.session_state.selected_kpis]
            kpi_mod.add_period_deltas(mostrados, df, mapping)
            rows = [mostrados[i:i + 4] for i in range(0, len(mostrados), 4)]
            for row in rows:
                cc = st.columns(4)
                for c, k in zip(cc, row):
                    c.metric(k.name, k.display(st.session_state.currency),
                             delta=(fmt_pct(k.delta) if k.delta is not None and np.isfinite(k.delta) else None),
                             help=k.help)

        st.divider()
        st.subheader("KPIs propios")
        st.caption("Define tus propias métricas con una fórmula. Se calculan sobre los datos activos.")

        with st.expander("Cómo escribir una fórmula", expanded=not st.session_state.custom):
            st.markdown("Usa nombres de columna entre comillas y estas funciones:")
            st.code("\n".join(kpi_mod.FUNCTIONS.values()), language="text")
            st.markdown("**Ejemplos**")
            st.code('suma("Importe") - suma("Costo")\n'
                    'suma("Importe") / unicos("Cliente")\n'
                    '(suma("Importe") - suma("Costo")) / suma("Importe")\n'
                    'suma_si("Importe", "Canal", "Mayoreo") / suma("Importe")\n'
                    'conteo_si("Estatus", "Cancelado") / conteo()', language="python")

        with st.form("nuevo_kpi", clear_on_submit=True):
            c1, c2 = st.columns([1, 2])
            nombre = c1.text_input("Nombre del KPI", placeholder="Margen bruto")
            formula = c2.text_input("Fórmula",
                                    placeholder='(suma("Importe") - suma("Costo")) / suma("Importe")')
            c3, c4 = st.columns([1, 1])
            formato = c3.selectbox("Formato", [kpi_mod.FMT_NUM, kpi_mod.FMT_MONEY,
                                               kpi_mod.FMT_PCT, kpi_mod.FMT_INT])
            meta = c4.number_input("Meta (opcional)", value=0.0, step=1.0,
                                   help="Deja 0 si no aplica. Sirve para el semáforo.")
            if st.form_submit_button("Agregar KPI", type="primary"):
                if not nombre or not formula:
                    st.error("Falta el nombre o la fórmula.")
                else:
                    try:
                        kpi_mod.FormulaEvaluator(df).evaluate(formula)
                        st.session_state.custom.append(
                            kpi_mod.CustomKPI(nombre, formula, formato, meta or None))
                        st.rerun()
                    except kpi_mod.FormulaError as e:
                        st.error(f"Fórmula inválida: {e}")

        if st.session_state.custom:
            res, errs = kpi_mod.compute_custom(df, st.session_state.custom)
            for e in errs:
                st.error(e)
            cc = st.columns(min(4, max(len(res), 1)))
            for i, k in enumerate(res):
                with cc[i % len(cc)]:
                    st.metric(k.name, k.display(st.session_state.currency), help=k.help)
                    if k.target:
                        st.plotly_chart(charts.gauge(k.value, k.target, f"Meta: {fmt_num(k.target)}"),
                                        use_container_width=True,
                                        key=f"g_{i}", config={"displayModeBar": False})
            borrar = st.selectbox("Eliminar un KPI propio",
                                  ["(ninguno)"] + [k.name for k in st.session_state.custom])
            if borrar != "(ninguno)" and st.button("Eliminar"):
                st.session_state.custom = [k for k in st.session_state.custom if k.name != borrar]
                st.rerun()


    # ========================================================= 2. COMPRENSIÓN
    elif phase == "comprension":
        df = active_df()
        profiles, issues, score, counts, overview = analyze(df)
        st.title("Fase 2 — Comprensión de los datos")
        st.caption("Qué hay dentro del archivo y qué está mal. Los problemas se listan de mayor a "
                   "menor severidad, con el número de registros afectados.")

        c = st.columns([1.1, 1, 1, 1, 1.4])
        color = "🟢" if score >= 80 else ("🟡" if score >= 55 else "🔴")
        c[0].metric("Calidad de datos", f"{score}/100", help="100 = sin problemas detectados.")
        c[1].metric("Registros", f"{overview['filas']:,}")
        c[2].metric("Variables", overview["columnas"])
        c[3].metric("Celdas vacías", fmt_pct(overview["pct_vacias"]))
        c[4].metric("Problemas", f"{counts['crítico']} críticos · {counts['advertencia']} advert.")

        if score < 55:
            st.error(f"{color} Calidad baja. Resuelve los críticos en la fase 3 antes de sacar "
                     "conclusiones: los totales y promedios de arriba están afectados.", icon="🚨")
        elif score < 80:
            st.warning(f"{color} Calidad aceptable con reservas. Revisa los puntos marcados.", icon="⚠️")
        else:
            st.success(f"{color} Calidad buena. Puedes analizar con confianza razonable.", icon="✅")

        t1, t2, t3, t4 = st.tabs(["🚨 Errores detectados", "🔎 Perfil de variables",
                                  "📈 Distribuciones", "🧮 Vista previa"])

        with t1:
            if not issues:
                st.success("No se detectó ningún problema de calidad.")
            else:
                cc = st.columns([2, 1])
                with cc[1]:
                    st.plotly_chart(charts.severity_donut(counts), use_container_width=True,
                                    key="donut", config={"displayModeBar": False})
                    fig_missing = charts.missing_bar(profiles)
                    if fig_missing:
                        st.plotly_chart(fig_missing, use_container_width=True, key="missing",
                                        config={"displayModeBar": False})
                with cc[0]:
                    filtro = st.multiselect("Filtrar por severidad",
                                            ["crítico", "advertencia", "info"],
                                            default=["crítico", "advertencia"])
                    pill = {"crítico": "p-crit", "advertencia": "p-adv", "info": "p-info"}
                    mostrados = [i for i in issues if i.severity in filtro]
                    st.caption(f"{len(mostrados)} de {len(issues)} problemas")
                    for i in mostrados:
                        st.markdown(
                            f"<div class='ins-card {'ins-riesgo' if i.severity=='crítico' else ''}'>"
                            f"<b><span class='pill {pill[i.severity]}'>{i.severity.upper()}</span> "
                            f"&nbsp;{i.title}</b><span>{i.detail}</span></div>",
                            unsafe_allow_html=True)
                st.download_button("⬇️ Descargar lista de problemas (CSV)",
                                   deployment.to_csv_bytes(quality.issues_table(issues)),
                                   "problemas_calidad.csv", "text/csv")

        with t2:
            st.dataframe(profiling.profiles_table(profiles), use_container_width=True, height=420)
            st.caption("El *tipo detectado* no es el dtype: la app infiere qué es realmente cada "
                       "columna (una fecha guardada como texto se marca como fecha).")

        with t3:
            num_cols = profiling.suggest_metric_columns(profiles)
            dim_cols = profiling.suggest_dimension_columns(profiles)
            if not num_cols:
                st.info("No hay columnas numéricas analizables.")
            else:
                col = st.selectbox("Variable numérica", num_cols)
                serie = to_numeric_series(df[col]).dropna()
                cc = st.columns(4)
                cc[0].metric("Media", fmt_num(serie.mean()))
                cc[1].metric("Mediana", fmt_num(serie.median()))
                cc[2].metric("Desv. estándar", fmt_num(serie.std(ddof=1)))
                cc[3].metric("Rango", f"{fmt_num(serie.min())} – {fmt_num(serie.max())}")
                st.plotly_chart(charts.histogram(serie, f"Distribución de '{col}'", col,
                                                 median=float(serie.median())),
                                use_container_width=True, key="hist", config={"displayModeBar": False})
                if dim_cols:
                    dim = st.selectbox("Comparar por", dim_cols)
                    d = df[[dim, col]].copy()
                    d[col] = to_numeric_series(d[col])
                    st.plotly_chart(charts.box_by_group(d.dropna(), dim, col,
                                                        f"'{col}' por '{dim}'"),
                                    use_container_width=True, key="box",
                                    config={"displayModeBar": False})
            if len(num_cols) >= 2:
                sub = df[num_cols].apply(to_numeric_series)
                corr = sub.corr(numeric_only=True).dropna(how="all").dropna(axis=1, how="all")
                if len(corr) >= 2:
                    st.plotly_chart(charts.heatmap_corr(corr.round(2), "Correlación entre variables"),
                                    use_container_width=True, key="corr",
                                    config={"displayModeBar": False})

        with t4:
            st.dataframe(df.head(200), use_container_width=True, height=460)


    # ========================================================= 3. PREPARACIÓN
    elif phase == "preparacion":
        raw = st.session_state.raw
        profiles, issues, score, counts, overview = analyze(raw)
        st.title("Fase 3 — Preparación de los datos")
        st.caption("Cada acción queda registrada en una bitácora: el resultado es auditable y "
                   "reproducible. Nada se aplica sobre el archivo original.")

        reparables = [i for i in issues if i.fix]
        plan = st.session_state.plan

        st.subheader("Reparaciones sugeridas")
        if not reparables:
            st.success("No hay nada que reparar automáticamente.")
        else:
            c1, c2 = st.columns([1, 1])
            if c1.button("Seleccionar reparaciones seguras", use_container_width=True,
                         help="Solo lo que no cambia el significado del dato: duplicados, tipos, "
                              "espacios, codificación y categorías equivalentes."):
                st.session_state.plan = prep.plan_from_issues(issues, aggressive=False)
                st.rerun()
            if c2.button("Limpiar selección", use_container_width=True):
                st.session_state.plan = prep.PrepPlan()
                st.rerun()

            marcados = set()
            for i in reparables:
                key = f"fix_{i.code}_{i.column}"
                ya = (
                    (i.fix == "drop_duplicates" and plan.drop_duplicates)
                    or (i.fix == "drop_empty_rows" and plan.drop_empty_rows)
                    or (i.fix == "drop_column" and i.column in plan.drop_columns)
                    or (i.fix == "trim" and i.column in plan.trim)
                    or (i.fix == "fix_mojibake" and i.column in plan.fix_mojibake)
                    or (i.fix == "normalize_categories" and i.column in plan.normalize_categories)
                    or (i.fix == "to_numeric" and i.column in plan.to_numeric)
                    or (i.fix == "to_datetime" and i.column in plan.to_datetime)
                    or (i.fix == "null_dates" and i.column in plan.null_dates)
                    or (i.fix == "clip_outliers" and i.column in plan.clip_outliers)
                )
                check = st.checkbox(f"**{i.fix_label}** — {i.title}", value=ya, key=key,
                                    help=i.detail)
                if check:
                    marcados.add((i.fix, i.column))

            nuevo = prep.PrepPlan()
            for fix, col in marcados:
                if fix == "drop_duplicates":
                    nuevo.drop_duplicates = True
                elif fix == "drop_empty_rows":
                    nuevo.drop_empty_rows = True
                elif fix == "drop_column":
                    nuevo.drop_columns.append(col)
                elif fix == "trim":
                    nuevo.trim.append(col)
                elif fix == "fix_mojibake":
                    nuevo.fix_mojibake.append(col)
                elif fix == "normalize_categories":
                    nuevo.normalize_categories.append(col)
                elif fix == "to_numeric":
                    nuevo.to_numeric.append(col)
                elif fix == "to_datetime":
                    nuevo.to_datetime.append(col)
                elif fix == "null_dates":
                    nuevo.null_dates.append(col)
                elif fix == "clip_outliers":
                    nuevo.clip_outliers[col] = "iqr"
            nuevo.impute = dict(plan.impute)
            nuevo.snake_case = plan.snake_case
            st.session_state.plan = nuevo
            plan = nuevo

        st.divider()
        st.subheader("Tratamiento de valores faltantes")
        con_nulos = [p.name for p in profiles.values() if p.n_missing > 0]
        if not con_nulos:
            st.caption("Ninguna columna tiene valores faltantes.")
        else:
            cols = st.columns(2)
            for i, c in enumerate(con_nulos):
                p = profiles[c]
                opciones = ["nada", "mediana", "media", "moda", "cero", "desconocido",
                            "ffill", "eliminar_filas"]
                if p.semantic not in ("numerico",):
                    opciones = [o for o in opciones if o not in ("mediana", "media", "cero")]
                with cols[i % 2]:
                    val = st.selectbox(
                        f"{c} — {p.n_missing:,} nulos ({p.pct_missing:.0%})",
                        opciones, index=opciones.index(plan.impute.get(c, "nada"))
                        if plan.impute.get(c, "nada") in opciones else 0,
                        format_func=lambda o: prep.IMPUTE_STRATEGIES[o], key=f"imp_{c}")
                    if val != "nada":
                        plan.impute[c] = val
                    else:
                        plan.impute.pop(c, None)

        st.divider()
        c1, c2 = st.columns(2)
        plan.snake_case = c1.checkbox("Normalizar nombres de columna a snake_case", plan.snake_case)
        extra_drop = c2.multiselect("Eliminar columnas adicionales",
                                    [c for c in raw.columns if c not in plan.drop_columns])
        plan_final = prep.PrepPlan(**{**plan.__dict__,
                                      "drop_columns": list(plan.drop_columns) + list(extra_drop)})

        st.divider()
        if plan_final.is_empty():
            st.info("No hay ninguna transformación seleccionada.")
        else:
            preview, log = prep.apply_plan(raw, plan_final)
            st.subheader("Vista previa del resultado")
            cc = st.columns(3)
            cc[0].metric("Filas", f"{len(preview):,}", delta=f"{len(preview)-len(raw):,}")
            cc[1].metric("Columnas", preview.shape[1], delta=preview.shape[1] - raw.shape[1])
            _, _, new_score, _, _ = analyze(preview)
            cc[2].metric("Calidad", f"{new_score}/100", delta=new_score - score)
            st.dataframe(preview.head(50), use_container_width=True, height=300)
            with st.expander("Bitácora de transformaciones", expanded=True):
                for l in log:
                    st.markdown(f"- {l}")
            if st.button("✅ Aplicar y usar estos datos", type="primary"):
                st.session_state.clean = preview
                st.session_state.prep_log = log
                st.session_state.model = None
                goto("dashboard")
                st.rerun()

        if st.session_state.prep_log:
            st.divider()
            st.subheader("Bitácora aplicada")
            for l in st.session_state.prep_log:
                st.markdown(f"- {l}")


    # ============================================================ 4. DASHBOARD
    elif phase == "dashboard":
        df = active_df()
        profiles, issues, score, counts, overview = analyze(df)
        mapping = st.session_state.mapping or kpi_mod.suggest_mapping(profiles)
        st.session_state.mapping = mapping

        st.title("Dashboard")
        st.caption(f"{st.session_state.source} · {len(df):,} registros · calidad {score}/100")

        catalogo = kpi_mod.compute_catalog(df, mapping)
        if st.session_state.selected_kpis:
            catalogo = [k for k in catalogo if k.name in st.session_state.selected_kpis]
        kpi_mod.add_period_deltas(catalogo, df, mapping)
        propios, errs = kpi_mod.compute_custom(df, st.session_state.custom)
        todos = catalogo + propios

        with st.spinner("Generando insights…"):
            hallazgos = ins_mod.generate_insights(df, profiles, mapping, issues)
        resumen = ins_mod.executive_summary(hallazgos, overview, score)

        st.info(resumen.replace("**", ""), icon="📌")

        if todos:
            for row_start in range(0, min(len(todos), 8), 4):
                cc = st.columns(4)
                for c, k in zip(cc, todos[row_start:row_start + 4]):
                    c.metric(k.name, k.display(st.session_state.currency),
                             delta=(fmt_pct(k.delta) if k.delta is not None and np.isfinite(k.delta) else None),
                             help=k.help)
        else:
            st.warning("Mapea columnas de negocio en la fase 1 para ver KPIs.", icon="⚠️")

        st.divider()
        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Hallazgos")
            if not hallazgos:
                st.info("Sin hallazgos. Mapea fecha, importe y una dimensión en la fase 1.")
            for n, h in enumerate(hallazgos):
                st.markdown(
                    f"<div class='ins-card ins-{h.kind}'><b>{ins_mod.ICON[h.kind]} {h.title}</b>"
                    f"<span>{h.text.replace('**','')}</span></div>", unsafe_allow_html=True)
                if h.chart:
                    spec = h.chart
                    try:
                        if spec["type"] == "line":
                            fig = charts.line_time(pd.DataFrame({"x": spec["x"], "y": spec["y"]}),
                                                   "x", "y", title=spec["title"], height=260)
                        elif spec["type"] == "bar":
                            fig = charts.bar_ranked(spec["labels"], spec["values"], spec["title"],
                                                    horizontal=len(spec["labels"]) > 7,
                                                    highlight=spec.get("highlight", 0), height=280)
                        elif spec["type"] == "box":
                            d = df[[spec["dim"], spec["value"]]].copy()
                            d[spec["value"]] = to_numeric_series(d[spec["value"]])
                            fig = charts.box_by_group(d.dropna(), spec["dim"], spec["value"],
                                                      spec["title"], height=280)
                        elif spec["type"] == "hist":
                            s = to_numeric_series(df[spec["col"]]).dropna()
                            fig = charts.histogram(s, spec["title"], spec["col"], 260,
                                                   median=float(s.median()))
                        elif spec["type"] == "scatter":
                            d = df[[spec["x"], spec["y"]]].apply(to_numeric_series).dropna()
                            fig = charts.scatter(d, spec["x"], spec["y"], title=spec["title"], height=300)
                        else:
                            fig = None
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True, key=f"ins_{n}",
                                            config={"displayModeBar": False})
                    except Exception as e:  # noqa: BLE001
                        st.caption(f"(No se pudo dibujar la gráfica: {type(e).__name__})")

        with right:
            st.subheader("Exploración")
            date_cols = profiling.suggest_date_columns(profiles)
            num_cols = profiling.suggest_metric_columns(profiles)
            dim_cols = profiling.suggest_dimension_columns(profiles)

            if date_cols and num_cols:
                c1, c2 = st.columns(2)
                dcol = c1.selectbox("Fecha", date_cols,
                                    index=date_cols.index(mapping["fecha"])
                                    if mapping.get("fecha") in date_cols else 0)
                mcol = c2.selectbox("Métrica", num_cols,
                                    index=num_cols.index(mapping["ingreso"])
                                    if mapping.get("ingreso") in num_cols else 0)
                freq_lbl = st.radio("Agrupar por", ["Día", "Semana", "Mes", "Trimestre"],
                                    index=2, horizontal=True)
                freq = {"Día": "D", "Semana": "W", "Mes": "ME", "Trimestre": "QE"}[freq_lbl]
                _fecha = to_datetime_series(df[dcol])
                _mask = ins_mod.robust_date_mask(_fecha)
                d = df.assign(_f=_fecha, _v=to_numeric_series(df[mcol]))[_mask]
                _fuera = int(_fecha.notna().sum() - len(d))
                if _fuera:
                    st.caption(f"⚠️ {_fuera} registro(s) con fecha fuera de rango se excluyeron de "
                               "las gráficas temporales.")
                if len(d):
                    serie = d.set_index("_f")["_v"].resample(freq).sum()
                    st.plotly_chart(
                        charts.line_time(pd.DataFrame({"x": serie.index, "y": serie.values}),
                                         "x", "y", title=f"{mcol} por {freq_lbl.lower()}",
                                         ylab=mcol, height=300),
                        use_container_width=True, key="ts", config={"displayModeBar": False})
                    if dim_cols:
                        dim = st.selectbox("Desglosar por", dim_cols)
                        dd = d.assign(_p=d["_f"].dt.to_period(
                            {"D": "D", "W": "W", "ME": "M", "QE": "Q"}[freq]).dt.to_timestamp())
                        st.plotly_chart(
                            charts.bar_grouped_time(dd, "_p", "_v", dim,
                                                    f"{mcol} por {freq_lbl.lower()} y {dim}",
                                                    ylab=mcol, height=320),
                            use_container_width=True, key="stack", config={"displayModeBar": False})
            elif num_cols and dim_cols:
                dim = st.selectbox("Dimensión", dim_cols)
                mcol = st.selectbox("Métrica", num_cols)
                agg = df.assign(_v=to_numeric_series(df[mcol])).groupby(dim)["_v"].sum()
                agg = agg.sort_values(ascending=False).head(12)
                st.plotly_chart(charts.bar_ranked(agg.index, agg.values, f"{mcol} por {dim}", mcol),
                                use_container_width=True, key="rank",
                                config={"displayModeBar": False})
            else:
                st.info("No hay combinación de fecha/métrica/dimensión para explorar.")

            if dim_cols and num_cols:
                st.markdown("**Ranking**")
                dim2 = st.selectbox("Agrupar por", dim_cols, key="rk_dim")
                met2 = st.selectbox("Sumar", num_cols, key="rk_met")
                g = (df.assign(_v=to_numeric_series(df[met2]))
                       .groupby(dim2)["_v"].agg(["sum", "count", "mean"])
                       .sort_values("sum", ascending=False))
                g.columns = [f"Total {met2}", "Registros", f"Promedio {met2}"]
                g["% del total"] = (g[f"Total {met2}"] / g[f"Total {met2}"].sum() * 100).round(1)
                st.dataframe(g.head(15).style.format({c: "{:,.2f}" for c in g.columns[:3]}),
                             use_container_width=True, height=320)

        st.divider()
        if st.button("📄 Preparar reporte para descargar"):
            goto("despliegue")
            st.rerun()


    # ============================================================ 5. MODELADO
    elif phase == "modelado":
        df = active_df()
        profiles, issues, score, counts, overview = analyze(df)
        st.title("Fase 4 — Modelado")
        st.caption("Aquí es fácil producir un modelo con buenas métricas y cero valor. "
                   "La app entrena siempre un baseline tonto y compara contra él, y busca fugas de "
                   "información antes de reportar nada.")

        if score < 55:
            st.warning("La calidad de los datos es baja. Modelar sobre datos sucios amplifica los "
                       "errores: pasa antes por la fase 3.", icon="⚠️")

        tipo = st.radio("¿Qué quieres hacer?", [
            "Predecir una columna (clasificación o regresión)",
            "Agrupar registros parecidos (segmentación)",
            "Encontrar registros atípicos (anomalías)",
            "Pronosticar una serie de tiempo",
        ], index=0)

        st.divider()

        if tipo.startswith("Predecir"):
            candidatas = [c for c in df.columns
                          if profiles[c].semantic not in ("vacia", "constante", "identificador", "texto")]
            if not candidatas:
                st.error("No hay ninguna columna que sirva como objetivo.")
                st.stop()
            target = st.selectbox("Columna objetivo (lo que quieres predecir)", candidatas)
            tarea = infer_task(df[target].dropna())
            st.caption(f"Tarea detectada: **{tarea}** "
                       f"({df[target].nunique()} valores distintos en el objetivo).")
            test_size = st.slider("Porcentaje reservado para prueba", 10, 40, 25, 5) / 100
            if st.button("Entrenar modelos", type="primary"):
                with st.spinner("Entrenando y validando…"):
                    st.session_state.model = run_supervised(df, profiles, target, test_size)
                st.session_state.model_target = target
                goto("evaluacion")
                st.rerun()

        elif tipo.startswith("Agrupar"):
            auto = st.checkbox("Elegir el número de grupos automáticamente (mejor silueta)", True)
            k = None if auto else st.slider("Número de segmentos", 2, 8, 4)
            if st.button("Segmentar", type="primary"):
                with st.spinner("Buscando segmentos…"):
                    st.session_state.model = run_clustering(df, profiles, k)
                st.session_state.model_target = None
                goto("evaluacion")
                st.rerun()

        elif tipo.startswith("Encontrar"):
            cont = st.slider("Porcentaje esperado de atípicos", 0.5, 10.0, 2.0, 0.5) / 100
            if st.button("Detectar anomalías", type="primary"):
                with st.spinner("Analizando…"):
                    st.session_state.model = run_anomalies(df, profiles, cont)
                st.session_state.model_target = None
                goto("evaluacion")
                st.rerun()

        else:
            date_cols = profiling.suggest_date_columns(profiles)
            num_cols = profiling.suggest_metric_columns(profiles)
            if not date_cols:
                st.error("No hay ninguna columna de fecha. Conviértela en la fase 3.")
                st.stop()
            c1, c2, c3 = st.columns(3)
            dcol = c1.selectbox("Columna de fecha", date_cols)
            mcol = c2.selectbox("Métrica a pronosticar", ["(conteo de registros)"] + num_cols)
            horizonte = c3.slider("Meses a pronosticar", 1, 18, 6)
            if st.button("Pronosticar", type="primary"):
                with st.spinner("Ajustando el modelo…"):
                    st.session_state.model = run_forecast(
                        df, dcol, None if mcol.startswith("(") else mcol, horizonte)
                st.session_state.model_target = None
                goto("evaluacion")
                st.rerun()

        if st.session_state.model is not None:
            st.divider()
            st.info(f"Último modelo entrenado: **{st.session_state.model.task}** — "
                    f"{st.session_state.model.headline}")


    # ============================================================ 6. EVALUACIÓN
    elif phase == "evaluacion":
        st.title("Fase 5 — Evaluación")
        rep = st.session_state.model
        if rep is None:
            st.info("Todavía no has entrenado ningún modelo.", icon="🧠")
            if st.button("Ir a Modelado", type="primary"):
                goto("modelado")
                st.rerun()
            st.stop()

        if not rep.ok:
            st.error(f"**{rep.headline}** — {rep.verdict}")
            st.stop()

        df = active_df()
        st.caption(f"Tarea: **{rep.task}** · {rep.headline}")

        if rep.beats_baseline is False:
            st.error(rep.verdict.replace("**", ""), icon="🚨")
        elif rep.verdict.startswith("⚠️"):
            st.error(rep.verdict.replace("**", ""), icon="⚠️")
        elif rep.verdict.startswith("🟡"):
            st.warning(rep.verdict.replace("**", ""), icon="🟡")
        else:
            st.success(rep.verdict.replace("**", ""), icon="✅")

        cc = st.columns(max(len(rep.metrics), 1))
        for c, (k, v) in zip(cc, rep.metrics.items()):
            base = rep.baseline.get(k)
            es_pct = k in ("Exactitud", "Exactitud balanceada", "F1 (macro)", "ROC AUC",
                           "MAPE", "% del total", "Silueta")
            valor = fmt_pct(v) if es_pct and abs(v) <= 1 else fmt_num(v)
            delta = None
            if base is not None and np.isfinite(base):
                d = v - base
                delta = f"{d:+.3f} vs baseline"
            c.metric(k, valor, delta=delta,
                     delta_color="normal" if k not in ("MAE", "RMSE", "MAPE") else "inverse")

        if rep.baseline:
            with st.expander("¿Qué es el baseline?"):
                st.markdown(
                    "Un modelo tonto que ignora todas las variables: en clasificación siempre "
                    "responde la clase más común, en regresión siempre el promedio. **Si tu modelo "
                    "no le gana con holgura, no aprendió nada útil.** Es la comparación que la "
                    "mayoría de las herramientas automáticas omite.")
                st.dataframe(pd.DataFrame({"Modelo": rep.metrics, "Baseline": rep.baseline}),
                             use_container_width=True)

        for w in rep.warnings:
            st.warning(w, icon="⚠️")

        st.divider()

        if rep.task == CLASIFICACION and "confusion" in rep.extra:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Matriz de confusión")
                cm, labels = rep.extra["confusion"], rep.extra["labels"]
                st.plotly_chart(charts.heatmap_seq(cm, labels, labels,
                                                   "Real (filas) vs. predicho (columnas)",
                                                   zlabel="Casos"),
                                use_container_width=True, key="cm", config={"displayModeBar": False})
                st.caption("La diagonal son los aciertos. Fuera de la diagonal está el tipo de "
                           "error que comete el modelo — más informativo que la exactitud global.")
            with c2:
                if rep.importance is not None:
                    st.subheader("Variables que más aportan")
                    st.plotly_chart(
                        charts.bar_ranked(rep.importance["Variable"], rep.importance["Importancia"],
                                          "Importancia por permutación", "Caída del desempeño al "
                                          "desordenar la variable", height=380, value_fmt=".4f"),
                        use_container_width=True, key="imp", config={"displayModeBar": False})

        elif rep.task == REGRESION and "y_true" in rep.extra:
            c1, c2 = st.columns(2)
            yt, yp = rep.extra["y_true"], rep.extra["y_pred"]
            with c1:
                st.subheader("Predicho vs. real")
                d = pd.DataFrame({"Real": yt, "Predicho": yp})
                st.plotly_chart(charts.scatter(d, "Real", "Predicho",
                                               title="Cada punto es un registro de prueba", height=380),
                                use_container_width=True, key="pvr", config={"displayModeBar": False})
                st.caption("En un modelo perfecto los puntos caerían sobre la diagonal.")
            with c2:
                st.subheader("Residuos")
                st.plotly_chart(charts.histogram(yt - yp, "Distribución del error", "Real − Predicho",
                                                 height=380, median=float(np.median(yt - yp))),
                                use_container_width=True, key="res", config={"displayModeBar": False})
                st.caption("Deben centrarse en cero y ser simétricos. Un sesgo indica que el modelo "
                           "sobre o subestima de forma sistemática.")
            if rep.importance is not None:
                st.plotly_chart(
                    charts.bar_ranked(rep.importance["Variable"], rep.importance["Importancia"],
                                      "Variables que más aportan", "Importancia por permutación",
                                      height=360, value_fmt=".4f"),
                    use_container_width=True, key="imp2", config={"displayModeBar": False})

        elif rep.task == CLUSTERING:
            st.subheader("Perfil de los segmentos")
            st.dataframe(rep.extra["resumen"].style.format(precision=2),
                         use_container_width=True)
            sil = rep.extra.get("sil_por_k", {})
            if len(sil) > 1:
                st.plotly_chart(
                    charts.bar_ranked([f"k={k}" for k in sil], list(sil.values()),
                                      "Calidad de la separación según el número de grupos",
                                      "Coeficiente de silueta", horizontal=False, height=280,
                                      value_fmt=".3f"),
                    use_container_width=True, key="sil", config={"displayModeBar": False})
            etiquetas = rep.extra["labels"]
            out = df.loc[etiquetas.index].copy()
            out["Segmento"] = etiquetas.values
            st.download_button("⬇️ Descargar datos con su segmento",
                               deployment.to_csv_bytes(out), "datos_segmentados.csv", "text/csv")

        elif rep.task == ANOMALIAS:
            st.subheader("Registros atípicos")
            idx = rep.extra["index"]
            anom = df.loc[idx].copy()
            anom.insert(0, "Score", rep.extra["score"].loc[idx].round(4))
            st.dataframe(anom.sort_values("Score").head(200), use_container_width=True, height=420)
            st.download_button("⬇️ Descargar registros atípicos",
                               deployment.to_csv_bytes(anom), "anomalias.csv", "text/csv")

        elif rep.task == FORECAST:
            s, test, pred, fut = (rep.extra["serie"], rep.extra["test"],
                                  rep.extra["pred_test"], rep.extra["futuro"])
            st.subheader("Pronóstico")
            hist = pd.DataFrame({"x": s.index, "y": s.values})
            fig = charts.line_time(hist, "x", "y", title="Histórico y pronóstico", height=380)
            pal = charts.palette()
            fig.add_scatter(x=list(fut.index), y=list(fut.values), mode="lines",
                            name="pronóstico", line=dict(width=2, dash="dot", color=pal[1]))
            fig.add_scatter(x=list(pred.index), y=list(pred.values), mode="lines",
                            name="validación", line=dict(width=2, color=pal[2]))
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True, key="fc", config={"displayModeBar": False})
            st.dataframe(pd.DataFrame({"Periodo": fut.index.strftime("%Y-%m"),
                                       "Pronóstico": fut.values.round(2)}),
                         use_container_width=True, height=260)

        st.divider()
        if st.button("Ir a Despliegue →", type="primary"):
            goto("despliegue")
            st.rerun()


    # ============================================================ 7. DESPLIEGUE
    elif phase == "despliegue":
        df = active_df()
        profiles, issues, score, counts, overview = analyze(df)
        mapping = st.session_state.mapping or kpi_mod.suggest_mapping(profiles)
        st.title("Fase 6 — Despliegue")
        st.caption("Sacar el resultado del análisis: datos limpios, reporte, modelo empaquetado y "
                   "calificación de archivos nuevos.")

        t1, t2, t3 = st.tabs(["📦 Exportar", "📄 Reporte", "🎯 Calificar datos nuevos"])

        with t1:
            st.subheader("Datos y tablas")
            catalogo = kpi_mod.compute_catalog(df, mapping)
            propios, _ = kpi_mod.compute_custom(df, st.session_state.custom)
            kpi_df = pd.DataFrame([{"KPI": k.name, "Valor": k.value,
                                    "Formato": k.fmt, "Origen": k.source}
                                   for k in catalogo + propios])
            hallazgos = ins_mod.generate_insights(df, profiles, mapping, issues)
            ins_df = pd.DataFrame([{"Tipo": h.kind, "Hallazgo": h.title, "Detalle": h.text.replace("**", "")}
                                   for h in hallazgos])

            c1, c2 = st.columns(2)
            c1.download_button("⬇️ Datos actuales (CSV)", deployment.to_csv_bytes(df),
                               "datos_preparados.csv", "text/csv", use_container_width=True)
            libro = {
                "Datos": df,
                "KPIs": kpi_df,
                "Insights": ins_df,
                "Problemas": quality.issues_table(issues),
                "Perfil": profiling.profiles_table(profiles),
                "Bitacora": pd.DataFrame({"Transformación": st.session_state.prep_log or ["(ninguna)"]}),
            }
            c2.download_button("⬇️ Libro completo (Excel)", deployment.to_excel_bytes(libro),
                               "analisis_crispdm.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

            rep = st.session_state.model
            st.divider()
            st.subheader("Modelo entrenado")
            if rep is None or not rep.ok or rep.model is None:
                st.caption("Entrena un modelo en la fase 4 para poder empaquetarlo.")
            else:
                if rep.beats_baseline is False:
                    st.warning("Este modelo no supera al baseline. Puedes descargarlo, pero no "
                               "deberías usarlo para decidir.", icon="⚠️")
                st.download_button(
                    "⬇️ Modelo empaquetado (.joblib)",
                    deployment.pack_model(rep, st.session_state.get("model_target"),
                                          st.session_state.source),
                    "modelo_crispdm.joblib", "application/octet-stream")
                st.code(
                    "import joblib, pandas as pd\n"
                    "b = joblib.load('modelo_crispdm.joblib')\n"
                    "nuevos = pd.read_csv('nuevos.csv')\n"
                    "nuevos['prediccion'] = b['pipeline'].predict(nuevos[b['variables']])\n",
                    language="python")

        with t2:
            st.subheader("Reporte HTML")
            st.caption("Un archivo autocontenido con KPIs, hallazgos, problemas de calidad, "
                       "bitácora de limpieza y resultados del modelo. Se abre en cualquier navegador.")
            hallazgos = ins_mod.generate_insights(df, profiles, mapping, issues)
            catalogo = kpi_mod.compute_catalog(df, mapping)
            propios, _ = kpi_mod.compute_custom(df, st.session_state.custom)
            html_report = deployment.build_html_report(
                source=st.session_state.source, overview=overview, score=score, counts=counts,
                kpis=catalogo + propios, insights=hallazgos,
                issues_df=quality.issues_table(issues),
                profile_df=profiling.profiles_table(profiles),
                prep_log=st.session_state.prep_log,
                model_report=st.session_state.model,
                summary=ins_mod.executive_summary(hallazgos, overview, score),
                currency=st.session_state.currency,
            )
            st.download_button("⬇️ Descargar reporte (HTML)", html_report.encode("utf-8"),
                               "reporte_crispdm.html", "text/html", type="primary")
            with st.expander("Vista previa"):
                st.components.v1.html(html_report, height=600, scrolling=True)

        with t3:
            st.subheader("Aplicar un modelo a datos nuevos")
            st.caption("Sube el .joblib que descargaste y un archivo con la misma estructura. "
                       "La app agrega la predicción como columna nueva.")
            mfile = st.file_uploader("Modelo (.joblib)", type=["joblib", "pkl"], key="mdl")
            dfile = st.file_uploader("Datos nuevos (CSV/Excel)",
                                     type=["csv", "tsv", "txt", "xlsx", "xlsm"], key="newdata")
            if mfile and dfile:
                try:
                    bundle = deployment.load_model(mfile.getvalue())
                    nuevos = loader.load_bytes(dfile.getvalue(), dfile.name).df
                    st.caption(f"Modelo de **{bundle.get('tarea')}** sobre '{bundle.get('objetivo')}', "
                               f"creado el {bundle.get('creado','?')}.")
                    res, avisos = deployment.score_dataframe(bundle, nuevos)
                    for a in avisos:
                        st.info(a)
                    st.dataframe(res.head(100), use_container_width=True, height=400)
                    st.download_button("⬇️ Descargar resultados",
                                       deployment.to_csv_bytes(res), "datos_calificados.csv", "text/csv",
                                       type="primary")
                except Exception as e:  # noqa: BLE001
                    st.error(f"{type(e).__name__}: {e}")
