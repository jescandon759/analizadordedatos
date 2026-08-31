"""Fase 6 — Deployment: exportación de resultados, empaquetado del modelo y scoring."""
from __future__ import annotations

import html
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from utils import fmt_num, fmt_pct

APP_VERSION = "1.0"


# ---------------------------------------------------------------- exportación de datos


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, d in sheets.items():
            if d is None or len(d) == 0:
                d = pd.DataFrame({"(sin datos)": []})
            safe = str(name)[:31].replace("/", "-").replace("\\", "-").replace("*", "")
            out = d.copy()
            for c in out.columns:
                if isinstance(out[c].dtype, pd.DatetimeTZDtype):
                    out[c] = out[c].dt.tz_localize(None)
            out.to_excel(xw, sheet_name=safe, index=False)
    return buf.getvalue()


# ---------------------------------------------------------------- Excel limpio

MAX_FILAS_EXCEL = 120_000   # arriba de esto el archivo tarda demasiado y come memoria


def _ancho(serie: pd.Series, encabezado: str) -> int:
    muestra = serie.head(300).astype(str)
    largo = int(muestra.str.len().max()) if len(muestra) else 0
    return max(10, min(42, max(largo, len(str(encabezado))) + 3))


def build_excel(
    df: pd.DataFrame, *, source: str, confianza: str, resumen: str,
    prep_log: list[str], problemas: pd.DataFrame, kpis: pd.DataFrame,
    hallazgos: pd.DataFrame, max_filas: int = MAX_FILAS_EXCEL,
) -> tuple[bytes, list[str]]:
    """Excel con los datos ya corregidos y las hojas que explican qué se hizo.

    Devuelve (bytes, avisos). El archivo es el entregable principal para quien
    solo quiere sus datos limpios y seguir trabajando en Excel.
    """
    avisos: list[str] = []
    datos = df
    if len(df) > max_filas:
        datos = df.head(max_filas)
        avisos.append(
            f"El archivo tiene {len(df):,} filas y en Excel se incluyeron las primeras "
            f"{max_filas:,} para que el archivo siga siendo manejable. Usa la descarga "
            "en CSV si necesitas todo.")

    try:
        import xlsxwriter  # noqa: F401
    except ImportError:
        # sin xlsxwriter no hay formato, pero el usuario igual se lleva sus datos
        hojas = {"Datos limpios": datos, "Qué corregimos":
                 pd.DataFrame({"Transformación": [l for l in prep_log
                                                  if not l.startswith("Resultado:")] or ["—"]}),
                 "Qué revisar": problemas, "Tus números": kpis, "Hallazgos": hallazgos}
        avisos.append("El Excel salió sin formato porque falta el paquete `xlsxwriter`; "
                      "los datos están completos.")
        return to_excel_bytes(hojas), avisos

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter",
                        datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as xw:
        wb = xw.book
        f_titulo = wb.add_format({"font_name": "Arial", "bold": True, "font_size": 15})
        f_sub = wb.add_format({"font_name": "Arial", "font_size": 10, "font_color": "#52514E"})
        f_head = wb.add_format({"font_name": "Arial", "bold": True, "font_size": 10,
                                "bg_color": "#2A78D6", "font_color": "white",
                                "border": 1, "border_color": "#1C5CAB",
                                "align": "left", "valign": "vcenter", "text_wrap": True})
        f_txt = wb.add_format({"font_name": "Arial", "font_size": 10, "valign": "top",
                               "text_wrap": True})
        f_num = wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "#,##0.00"})
        f_int = wb.add_format({"font_name": "Arial", "font_size": 10, "num_format": "#,##0"})
        f_fecha = wb.add_format({"font_name": "Arial", "font_size": 10,
                                 "num_format": "yyyy-mm-dd"})
        f_seccion = wb.add_format({"font_name": "Arial", "bold": True, "font_size": 11,
                                   "bottom": 1, "bottom_color": "#D8D7D3"})

        # ---------- Resumen
        ws = wb.add_worksheet("Resumen")
        xw.sheets["Resumen"] = ws
        ws.hide_gridlines(2)
        ws.set_column("A:A", 34)
        ws.set_column("B:B", 92)
        ws.write("A1", "Resumen del análisis", f_titulo)
        ws.write("A2", f"Archivo: {source}", f_sub)
        ws.write("A3", f"Generado: {datetime.now():%d/%m/%Y %H:%M}", f_sub)
        fila = 5
        ws.write(fila, 0, "Confianza en los datos", f_seccion)
        ws.write(fila, 1, confianza, f_txt)
        fila += 2
        ws.write(fila, 0, "En una frase", f_seccion)
        ws.write(fila, 1, resumen.replace("**", ""), f_txt)
        fila += 2
        if len(kpis):
            ws.write(fila, 0, "Tus números", f_seccion)
            fila += 1
            for _, r in kpis.iterrows():
                ws.write(fila, 0, str(r.iloc[0]), f_txt)
                ws.write(fila, 1, str(r.iloc[1]), f_txt)
                fila += 1
            fila += 1
        if len(hallazgos):
            ws.write(fila, 0, "Lo más importante", f_seccion)
            fila += 1
            for _, r in hallazgos.iterrows():
                ws.write(fila, 0, str(r.iloc[0]), f_txt)
                ws.write(fila, 1, str(r.iloc[1]), f_txt)
                fila += 1

        # ---------- Datos limpios
        datos.to_excel(xw, sheet_name="Datos limpios", index=False, startrow=0)
        wsd = xw.sheets["Datos limpios"]
        for j, col in enumerate(datos.columns):
            wsd.write(0, j, str(col), f_head)
            serie = datos[col]
            if pd.api.types.is_datetime64_any_dtype(serie):
                fmt = f_fecha
            elif pd.api.types.is_integer_dtype(serie):
                fmt = f_int
            elif pd.api.types.is_numeric_dtype(serie):
                fmt = f_num
            else:
                fmt = None
            wsd.set_column(j, j, _ancho(serie, col), fmt)
        wsd.freeze_panes(1, 0)
        if len(datos):
            wsd.autofilter(0, 0, len(datos), max(len(datos.columns) - 1, 0))
        wsd.set_row(0, 30)

        # ---------- hojas de explicación
        def hoja_texto(nombre: str, filas: list[tuple[str, str]], anchos=(28, 96)):
            w = wb.add_worksheet(nombre)
            xw.sheets[nombre] = w
            w.hide_gridlines(2)
            w.set_column("A:A", anchos[0])
            w.set_column("B:B", anchos[1])
            w.write(0, 0, nombre, f_titulo)
            for i, (a, b) in enumerate(filas, start=2):
                w.write(i, 0, a, f_txt)
                w.write(i, 1, b, f_txt)
            w.freeze_panes(2, 0)

        acciones = [l for l in prep_log if not l.startswith("Resultado:")]
        hoja_texto("Qué corregimos",
                   [(f"{i}.", l) for i, l in enumerate(acciones, 1)]
                   or [("—", "No hizo falta corregir nada.")])

        if len(problemas):
            hoja_texto("Qué revisar",
                       [(str(r.iloc[0]), str(r.iloc[1])) for _, r in problemas.iterrows()])
        else:
            hoja_texto("Qué revisar", [("—", "No se detectó ningún problema.")])

    return buf.getvalue(), avisos


# ---------------------------------------------------------------- modelo


def pack_model(report, target: str | None, source: str) -> bytes:
    bundle = {
        "app": "Analizador CRISP-DM",
        "version": APP_VERSION,
        "creado": datetime.now().isoformat(timespec="seconds"),
        "origen": source,
        "tarea": report.task,
        "objetivo": target,
        "variables": report.feature_names,
        "metricas": report.metrics,
        "baseline": report.baseline,
        "supera_baseline": report.beats_baseline,
        "pipeline": report.model,
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def load_model(raw: bytes) -> dict:
    return joblib.load(io.BytesIO(raw))


def score_dataframe(bundle: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Aplica un modelo empaquetado a datos nuevos. Devuelve (resultado, avisos)."""
    pipe, feats = bundle.get("pipeline"), bundle.get("variables", [])
    avisos: list[str] = []
    if pipe is None:
        raise ValueError("El archivo no contiene un modelo entrenado.")
    faltantes = [c for c in feats if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Al archivo nuevo le faltan columnas que el modelo necesita: "
            + ", ".join(faltantes[:10]) + ("…" if len(faltantes) > 10 else "")
        )
    sobrantes = [c for c in df.columns if c not in feats]
    if sobrantes:
        avisos.append(f"{len(sobrantes)} columna(s) del archivo no se usan en el modelo y se ignoran.")

    X = df[feats]
    out = df.copy()
    pred = pipe.predict(X)
    out["_prediccion"] = pred
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            out["_confianza"] = proba.max(axis=1)
            clases = list(getattr(pipe, "classes_", getattr(pipe[-1], "classes_", [])))
            if len(clases) == 2:
                out[f"_prob_{clases[1]}"] = proba[:, 1]
        except Exception:
            pass
    avisos.append(f"{len(out):,} registros calificados con el modelo de {bundle.get('tarea')}.")
    return out, avisos


# ---------------------------------------------------------------- reporte HTML

_CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e1;
--blue:#2a78d6;--red:#e34948;--amber:#eda100;--green:#008300;}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232322;--ink:#fff;
--ink2:#c3c2b7;--line:#3a3a37;--blue:#3987e5;--red:#e66767;--amber:#c98500;}}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px;background:var(--bg);color:var(--ink);
font:15px/1.6 Inter,-apple-system,Segoe UI,Roboto,sans-serif;}
.wrap{max-width:940px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px;
padding-bottom:8px;border-bottom:1px solid var(--line)}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--ink2);font-size:13px;margin-bottom:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi .v{font-size:23px;font-weight:600;letter-spacing:-.01em}
.kpi .l{color:var(--ink2);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.ins{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--blue);
border-radius:8px;padding:12px 14px;margin-bottom:10px}
.ins.riesgo{border-left-color:var(--red)}.ins.oportunidad{border-left-color:var(--green)}
.ins.contexto{border-left-color:var(--ink2)}
.ins b{display:block;margin-bottom:4px}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;display:block;
overflow-x:auto}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--ink2);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:11px;font-weight:600}
.t-crit{background:rgba(227,73,72,.14);color:var(--red)}
.t-adv{background:rgba(237,161,0,.16);color:var(--amber)}
.t-info{background:rgba(42,120,214,.14);color:var(--blue)}
.score{font-size:40px;font-weight:700;letter-spacing:-.02em}
footer{margin-top:40px;color:var(--ink2);font-size:12px;border-top:1px solid var(--line);
padding-top:14px}
"""


def _tbl(df: pd.DataFrame, limit: int = 60) -> str:
    if df is None or len(df) == 0:
        return "<p class='sub'>Sin registros.</p>"
    d = df.head(limit)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in d.columns)
    body = ""
    for _, row in d.iterrows():
        cells = "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
        body += f"<tr>{cells}</tr>"
    extra = (f"<p class='sub'>Mostrando {limit} de {len(df):,} filas.</p>"
             if len(df) > limit else "")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{extra}"


def build_html_report(*, source: str, overview: dict, score: int, counts: dict,
                      kpis: list, insights: list, issues_df: pd.DataFrame,
                      profile_df: pd.DataFrame, prep_log: list[str],
                      model_report=None, summary: str = "", currency: str = "$") -> str:
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")
    color = "var(--green)" if score >= 80 else ("var(--amber)" if score >= 55 else "var(--red)")

    kpi_html = "".join(
        f"<div class='card kpi'><div class='l'>{html.escape(k.name)}</div>"
        f"<div class='v'>{html.escape(k.display(currency))}</div></div>"
        for k in kpis
    ) or "<p class='sub'>No se configuraron KPIs.</p>"

    ins_html = "".join(
        f"<div class='ins {i.kind}'><b>{html.escape(i.title)}</b>"
        f"{html.escape(i.text).replace('**','')}</div>"
        for i in insights
    ) or "<p class='sub'>No se generaron hallazgos.</p>"

    prep_html = ("<ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in prep_log) + "</ul>"
                 if prep_log else "<p class='sub'>No se aplicaron transformaciones.</p>")

    model_html = ""
    if model_report is not None and model_report.ok:
        mm = "".join(
            f"<div class='card kpi'><div class='l'>{html.escape(k)}</div>"
            f"<div class='v'>{(fmt_pct(v) if abs(v)<=1 and k not in ('MAE','RMSE','Segmentos','Anomalías') else fmt_num(v))}</div></div>"
            for k, v in model_report.metrics.items()
        )
        warns = ("<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in model_report.warnings)
                 + "</ul>") if model_report.warnings else ""
        model_html = f"""
        <h2>4 · 5 — Modelado y evaluación</h2>
        <p><b>{html.escape(model_report.task.capitalize())}</b> — {html.escape(model_report.headline)}</p>
        <div class='grid'>{mm}</div>
        <div class='ins' style='margin-top:14px'>{html.escape(model_report.verdict).replace('**','')}</div>
        {('<h3>Advertencias</h3>' + warns) if warns else ''}
        """

    issues_view = issues_df.copy() if issues_df is not None and len(issues_df) else pd.DataFrame()
    if len(issues_view):
        issues_view["Severidad"] = issues_view["Severidad"].map(
            lambda s: {"crítico": "CRÍTICO", "advertencia": "ADVERTENCIA"}.get(s, "INFO"))

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reporte CRISP-DM — {html.escape(source)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Reporte de análisis CRISP-DM</h1>
<p class="sub">Fuente: <b>{html.escape(source)}</b> · Generado el {ts}</p>

<h2>Resumen ejecutivo</h2>
<p>{summary.replace('**','')}</p>
<div class="grid">
  <div class="card"><div class="l">Calidad de datos</div>
    <div class="score" style="color:{color}">{score}<span style="font-size:16px">/100</span></div>
    <div class="sub" style="margin:0">{counts.get('crítico',0)} críticos ·
    {counts.get('advertencia',0)} advertencias · {counts.get('info',0)} informativos</div></div>
  <div class="card kpi"><div class="l">Registros</div><div class="v">{overview['filas']:,}</div></div>
  <div class="card kpi"><div class="l">Variables</div><div class="v">{overview['columnas']}</div></div>
  <div class="card kpi"><div class="l">Celdas vacías</div>
    <div class="v">{fmt_pct(overview['pct_vacias'])}</div></div>
</div>

<h2>1 — Comprensión del negocio (KPIs)</h2>
<div class="grid">{kpi_html}</div>

<h2>2 — Comprensión de los datos</h2>
<h3>Hallazgos de negocio</h3>
{ins_html}
<h3>Problemas de calidad detectados</h3>
{_tbl(issues_view)}
<h3>Perfil de las variables</h3>
{_tbl(profile_df)}

<h2>3 — Preparación de los datos</h2>
{prep_html}

{model_html}

<footer>Generado con el Analizador CRISP-DM v{APP_VERSION}. Los hallazgos provienen de reglas
estadísticas deterministas aplicadas a los datos cargados; no sustituyen el criterio de negocio.
</footer>
</div></body></html>"""
