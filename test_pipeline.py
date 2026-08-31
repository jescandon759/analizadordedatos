"""Prueba end-to-end sin Streamlit: recorre las seis fases sobre datos sucios."""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import deployment
import insights as ins_mod
import kpis as kpi_mod
import loader
import prep
import profiling
import quality
from demo import build_demo
from modeling import run_anomalies, run_clustering, run_forecast, run_supervised

FALLOS: list[str] = []


def check(label: str, cond: bool, extra: str = ""):
    print(("  ✓ " if cond else "  ✗ ") + label + (f" — {extra}" if extra else ""))
    if not cond:
        FALLOS.append(label)


def seccion(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------- 0. carga
seccion("0 · Carga")
demo = build_demo()
csv_bytes = demo.to_csv(index=False).encode("utf-8")
res = loader.load_bytes(csv_bytes, "ventas.csv")
check("CSV leído", len(res.df) == len(demo), f"{len(res.df)} filas")
check("Separador detectado", res.separator == ",", repr(res.separator))

# CSV con basura arriba, punto y coma y latin-1
sucio = "Reporte mensual\n\nGenerado por el sistema\nFecha;Región;Monto\n01/03/2024;Bajío;1.234,50\n02/03/2024;Norte;987,00\n"
r2 = loader.load_bytes(sucio.encode("latin-1"), "raro.csv")
check("Encabezado desplazado detectado", list(r2.df.columns) == ["Fecha", "Región", "Monto"],
      str(list(r2.df.columns)))
check("Separador ';' detectado", r2.separator == ";", repr(r2.separator))

# Excel viejo (.xls) — requiere xlrd
try:
    import xlwt
    _wb = xlwt.Workbook(); _ws = _wb.add_sheet("Ventas")
    for _r, _fila in enumerate([["Fecha", "Canal", "Importe"],
                                ["01/03/2024", "Mayoreo", 1500.5],
                                ["02/03/2024", "Menudeo", 320.0]]):
        for _c, _v in enumerate(_fila):
            _ws.write(_r, _c, _v)
    _b = io.BytesIO(); _wb.save(_b)
    _r_xls = loader.load_bytes(_b.getvalue(), "viejo.xls")
    check("Lee Excel viejo (.xls)", _r_xls.df.shape == (2, 3), str(_r_xls.df.shape))
except ImportError:
    check("Lee Excel viejo (.xls)", False, "falta xlrd o xlwt en el entorno")

# mensajes de error en lenguaje llano
_q, _c = loader.mensaje_amigable(ImportError("Import xlrd failed. Install xlrd >= 2.0.1"), "v.xls")
check("Traduce el error de .xls", "Excel 97-2003" in _q and "Guardar como" in _c, _q[:50])
_q2, _ = loader.mensaje_amigable(MemoryError("Unable to allocate"), "g.xlsx")
check("Traduce el error de memoria", "demasiado grande" in _q2)
_q3, _c3 = loader.mensaje_amigable(RuntimeError("algo inesperado"), "x.csv")
check("Error desconocido da una salida útil", "xlsx" in _c3 or "CSV" in _c3)
check("Ningún mensaje expone jerga de Python",
      all("Traceback" not in t and "ImportError:" not in t
          for t in (_q, _c, _q2, _q3, _c3)))

# Excel con hoja
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as xw:
    demo.head(200).to_excel(xw, sheet_name="Ventas", index=False)
r3 = loader.load_bytes(buf.getvalue(), "libro.xlsx", sheet="Ventas")
check("Excel leído", len(r3.df) == 200, f"{len(r3.df)} filas")

df = res.df

# ---------------------------------------------------------------- 1-2. perfil y calidad
seccion("1·2 · Perfilado y calidad")
profiles = profiling.profile_dataframe(df)
issues = quality.detect_issues(df, profiles)
score, counts = quality.quality_score(df, issues)
overview = profiling.dataset_overview(df, profiles)
codes = {i.code for i in issues}
print("  problemas:", sorted(codes))
check("Detecta número como texto", "numero_como_texto" in codes)
check("Detecta fecha como texto", "fecha_como_texto" in codes)
check("Detecta duplicados", "filas_duplicadas" in codes)
check("Detecta columna vacía", "columna_vacia" in codes)
check("Detecta columna constante", "columna_constante" in codes)
check("Detecta categorías inconsistentes", "categorias_inconsistentes" in codes)
check("Detecta mojibake", "mojibake" in codes)
check("Detecta nulos", "nulos" in codes)
check("Detecta fechas imposibles", "fechas_imposibles" in codes)
check("Puntaje en rango", 0 <= score <= 100, str(score))
check("Perfil cubre todas las columnas", len(profiles) == df.shape[1])
check("Tabla de perfil se arma", len(profiling.profiles_table(profiles)) == df.shape[1])
check("Tabla de problemas se arma", len(quality.issues_table(issues)) == len(issues))

# ---------------------------------------------------------------- 3. preparación
seccion("3 · Preparación")
plan = prep.plan_from_issues(issues, aggressive=False)
clean, log = prep.apply_plan(df, plan)
check("Sin duplicados tras limpiar", clean.duplicated().sum() == 0)
check("Importe es numérico", pd.api.types.is_numeric_dtype(clean["Importe"]), str(clean["Importe"].dtype))
check("Fecha es datetime", pd.api.types.is_datetime64_any_dtype(clean["Fecha"]), str(clean["Fecha"].dtype))
check("Canal unificado", clean["Canal"].nunique() < df["Canal"].nunique(),
      f"{df['Canal'].nunique()} → {clean['Canal'].nunique()}")
check("Mojibake reparado", clean["Vendedor"].astype(str).str.contains("María").any())
check("Fechas imposibles vaciadas",
      clean["Fecha"].dropna().min() >= pd.Timestamp("1950-01-01"),
      str(clean["Fecha"].dropna().min()))
check("Bitácora generada", len(log) >= 5, f"{len(log)} líneas")

p2 = profiling.profile_dataframe(clean)
i2 = quality.detect_issues(clean, p2)
s2, _ = quality.quality_score(clean, i2)
check("La calidad mejora tras limpiar", s2 > score, f"{score} → {s2}")

plan_agr = prep.plan_from_issues(issues, aggressive=True)
clean2, log2 = prep.apply_plan(df, plan_agr)
check("Plan agresivo corre sin error", len(clean2) > 0)

# ---------------------------------------------------------------- KPIs
seccion("1 · KPIs")
mapping = kpi_mod.suggest_mapping(p2)
print("  mapeo:", {k: v for k, v in mapping.items() if v})
check("Detecta la fecha", mapping["fecha"] == "Fecha", str(mapping["fecha"]))
check("Detecta el importe", mapping["ingreso"] == "Importe", str(mapping["ingreso"]))
check("Detecta el estatus", mapping["estatus"] == "Estatus", str(mapping["estatus"]))
check("El folio sí califica como transacción", mapping["transaccion"] == "Folio",
      str(mapping["transaccion"]))

# un SKU se repite por diseño: no es folio ni "identificador repetido"
_rng = np.random.default_rng(4)
_sku = pd.DataFrame({
    "Fecha": pd.to_datetime(_rng.choice(pd.date_range("2024-01-01", "2025-08-15"), 4000)),
    "Id_es_SKUIDx": np.minimum(_rng.zipf(1.22, 4000), 2700).astype(float),
    "Sucursal": _rng.choice(list("ABCDE"), 4000),
    "Importe": np.abs(_rng.lognormal(6, 1, 4000)).round(2)})
_p_sku = profiling.profile_dataframe(_sku)
_m_sku = kpi_mod.suggest_mapping(_p_sku)
check("Un SKU repetido no se toma como folio", _m_sku["transaccion"] is None,
      str(_m_sku["transaccion"]))
check("Un SKU repetido no se reporta como clave duplicada",
      "id_duplicado" not in {i.code for i in quality.detect_issues(_sku, _p_sku)})
_h_sku = ins_mod.generate_insights(_sku, _p_sku, _m_sku, [])
_conc = [h for h in _h_sku if "20%" in h.titulo_llano or "Dependes" in h.titulo_llano]
check("Con muchos SKUs no dice que dependes de uno",
      bool(_conc) and "Dependes" not in _conc[0].titulo_llano,
      _conc[0].titulo_llano if _conc else "(sin hallazgo)")
check("Los nombres de categoría salen sin el '.0'",
      all(".0'" not in h.texto_llano for h in _h_sku),
      next((h.texto_llano[:60] for h in _h_sku if ".0'" in h.texto_llano), "ok"))
check("Segmento distinto de producto", mapping["segmento"] != mapping["producto"],
      f"{mapping['segmento']} vs {mapping['producto']}")
check("Detecta el costo", mapping["costo"] == "Costo", str(mapping["costo"]))
check("Detecta el cliente", mapping["cliente"] == "Cliente", str(mapping["cliente"]))

mapping["ingreso"] = "Importe"
cat = kpi_mod.compute_catalog(clean, mapping)
nombres = {k.name for k in cat}
print("  kpis:", sorted(nombres))
check("Calcula ingreso total", "Ingreso total" in nombres)
check("Calcula margen bruto", "Margen bruto" in nombres)
check("Calcula concentración", "Concentración top 10% clientes" in nombres)
total = next(k for k in cat if k.name == "Ingreso total").value
check("Ingreso total cuadra con la suma", abs(total - clean["Importe"].sum()) < 1e-6,
      f"{total:,.2f}")
ticket = next(k for k in cat if k.name == "Ticket promedio").value
check("Ticket promedio consistente", abs(ticket - total / clean["Folio"].nunique()) < 1e-6)

kpi_mod.add_period_deltas(cat, clean, mapping)
check("Delta de periodo calculado",
      any(k.delta is not None for k in cat if k.name == "Ingreso total"))

# fórmulas personalizadas
ev = kpi_mod.FormulaEvaluator(clean)
v = ev.evaluate('(suma("Importe") - suma("Costo")) / suma("Importe")')
esperado = (clean["Importe"].sum() - clean["Costo"].sum()) / clean["Importe"].sum()
check("Fórmula de margen correcta", abs(v - esperado) < 1e-9, f"{v:.4f}")
v2 = ev.evaluate('conteo_si("Estatus", "Cancelado") / conteo()')
check("conteo_si correcto",
      abs(v2 - (clean["Estatus"] == "Cancelado").mean()) < 1e-9, f"{v2:.4f}")
v3 = ev.evaluate('suma_si("Importe", "Canal", "Mayoreo")')
check("suma_si correcto",
      abs(v3 - clean.loc[clean["Canal"].str.lower() == "mayoreo", "Importe"].sum()) < 1e-6)
v4 = ev.evaluate('percentil("Importe", 90)')
check("percentil correcto", abs(v4 - clean["Importe"].quantile(0.9)) < 1e-6)

for mala in ['__import__("os").system("ls")', 'open("/etc/passwd")', 'suma("NoExiste")',
             'suma("Importe") / 0', '1 +', 'exec("x=1")']:
    try:
        ev.evaluate(mala)
        check(f"Rechaza fórmula peligrosa: {mala[:28]}", False)
    except Exception:
        check(f"Rechaza fórmula peligrosa: {mala[:28]}", True)

res_c, errs = kpi_mod.compute_custom(
    clean, [kpi_mod.CustomKPI("Margen", '(suma("Importe")-suma("Costo"))/suma("Importe")',
                              kpi_mod.FMT_PCT),
            kpi_mod.CustomKPI("Rota", 'suma("Inexistente")')])
check("KPI propio válido calcula", len(res_c) == 1)
check("KPI propio inválido reporta error", len(errs) == 1)

# ---------------------------------------------------------------- insights
seccion("2 · Insights")
hall = ins_mod.generate_insights(clean, p2, mapping, i2)
print("  hallazgos:")
for h in hall:
    print(f"    [{h.kind}] {h.title}")
check("Genera hallazgos", len(hall) >= 4, f"{len(hall)}")
check("Ninguna regla falló",
      not any("no pudo ejecutarse" in h.title for h in hall))
check("Detecta estacionalidad o tendencia",
      any("estacional" in h.title.lower() or "tendencia" in h.title.lower() for h in hall))
check("Detecta concentración",
      any("concentración" in h.title.lower() or "cola larga" in h.title.lower() for h in hall))
resumen = ins_mod.executive_summary(hall, profiling.dataset_overview(clean, p2), s2)
check("Resumen ejecutivo generado", len(resumen) > 60)

# ---------------------------------------------------------------- explicaciones
seccion("Explicación de gráficas")
import explicaciones as expl  # noqa: E402
from utils import to_datetime_series, to_numeric_series  # noqa: E402

_f = to_datetime_series(clean["Fecha"])
_d = clean.assign(_f=_f, _v=to_numeric_series(clean["Importe"]))[ins_mod.robust_date_mask(_f)]
_serie = _d.set_index("_f")["_v"].resample("ME").sum()
if _serie.index[-1] > _d["_f"].max():
    _serie = _serie.iloc[:-1]

_lect = expl.leer_serie(_serie, "ME", "mes", "Importe", "$")
check("Lee la serie de tiempo", "Cómo leerla" in _lect and "mejor mes" in _lect, _lect[:70])
check("Nombra el mejor y el peor mes", _lect.count("de 20") >= 2)

_at = expl.atipicos_serie(_serie, clean, "Fecha", "Importe", mapping, "ME", "mes", "$")
print("  atípicos detectados:", len(_at))
for _a in _at:
    print("   ·", _a.replace("<br>", " | ")[:150])
check("Detecta al menos un periodo atípico", len(_at) >= 1)
check("Nombra el periodo en palabras",
      any(m in _at[0].lower() for m in ("enero", "febrero", "marzo", "abril", "mayo", "junio",
                                        "julio", "agosto", "septiembre", "octubre",
                                        "noviembre", "diciembre")) if _at else False)
check("Da al menos una pista del origen", "<br>" in _at[0] if _at else False)

# una serie perfectamente plana no debe inventar atípicos
_plana = pd.Series([100.0] * 24,
                   index=pd.date_range("2024-01-31", periods=24, freq="ME"))
check("No inventa atípicos en una serie plana",
      expl.atipicos_serie(_plana, clean, "Fecha", "Importe", mapping, "ME", "mes") == [])

_agg = (clean.assign(_v=to_numeric_series(clean["Importe"]))
        .groupby("Canal")["_v"].sum().sort_values(ascending=False))
_lr = expl.leer_ranking(_agg, "Canal", "Importe", "$")
check("Lee el ranking", "encabeza" in _lr and "Menudeo" in _lr, _lr[:70])
check("Todas las formas tienen texto de lectura",
      all(expl.como_leer(t) for t in ("hist", "box", "scatter", "bar", "line")))
check("Series muy cortas no truenan",
      expl.leer_serie(pd.Series([1.0], index=pd.date_range("2024-01-31", periods=1, freq="ME")),
                      "ME", "mes", "X") == "")

# ---------------------------------------------------------------- modelado
seccion("4·5 · Modelado y evaluación")
r_clf = run_supervised(clean, p2, "Canal")
check("Clasificación corre", r_clf.ok, r_clf.headline)
check("Clasificación tiene baseline", bool(r_clf.baseline))
check("Veredicto emitido", len(r_clf.verdict) > 20)
print("   ", r_clf.headline, "|", r_clf.verdict[:80])

r_reg = run_supervised(clean, p2, "Importe")
check("Regresión corre", r_reg.ok, r_reg.headline)
print("   ", r_reg.headline)

# fuga de información explícita: una columna que es copia del objetivo
fuga = clean.copy()
fuga["Total facturado"] = clean["Importe"] * 1.16
p_fuga = profiling.profile_dataframe(fuga)
r_leak = run_supervised(fuga, p_fuga, "Importe")
check("Detecta fuga de información en regresión",
      any("correlación" in w.lower() or "misma variable" in w.lower() for w in r_leak.warnings),
      r_leak.warnings[0][:70] if r_leak.warnings else "(sin avisos)")

fuga2 = clean.copy()
fuga2["Resultado operacion"] = clean["Canal"].astype(str) + "-" + clean.index.astype(str).str[-1]
p_fuga2 = profiling.profile_dataframe(fuga2)
r_leak2 = run_supervised(fuga2, p_fuga2, "Canal")
check("Detecta fuga de información en clasificación",
      any("fuga" in w.lower() for w in r_leak2.warnings),
      r_leak2.warnings[0][:70] if r_leak2.warnings else "(sin avisos)")

# objetivo puramente aleatorio: el modelo NO debe ganarle al baseline
rng = np.random.default_rng(0)
ruido = clean.copy()
ruido["Aleatorio"] = rng.normal(size=len(ruido))
p_ruido = profiling.profile_dataframe(ruido)
r_noise = run_supervised(ruido, p_ruido, "Aleatorio")
check("Guardarraíl: no le gana al baseline con ruido puro",
      r_noise.beats_baseline is False, f"R²={r_noise.metrics.get('R²'):.3f}")

r_clu = run_clustering(clean, p2)
check("Segmentación corre", r_clu.ok, r_clu.headline)
r_ano = run_anomalies(clean, p2)
check("Anomalías corre", r_ano.ok, r_ano.headline)
r_fc = run_forecast(clean, "Fecha", "Importe", 6)
check("Pronóstico corre", r_fc.ok, r_fc.headline)

# datos insuficientes
r_small = run_supervised(clean.head(10), p2, "Canal")
check("Rechaza datos insuficientes", not r_small.ok)

# ---------------------------------------------------------------- despliegue
seccion("6 · Despliegue")
blob = deployment.pack_model(r_clf, "Canal", "ventas.csv")
bundle = deployment.load_model(blob)
scored, avisos = deployment.score_dataframe(bundle, clean.head(50))
check("Modelo empaquetado y recargado", "_prediccion" in scored.columns)
check("Predicciones completas", scored["_prediccion"].notna().all())
try:
    deployment.score_dataframe(bundle, clean.drop(columns=["Cantidad"]).head(5))
    check("Rechaza archivo con columnas faltantes", False)
except ValueError:
    check("Rechaza archivo con columnas faltantes", True)

xlsx = deployment.to_excel_bytes({"Datos": clean.head(100),
                                  "Problemas": quality.issues_table(i2)})
check("Excel generado", len(xlsx) > 5000, f"{len(xlsx)/1024:.0f} KB")

# --- Excel limpio para el usuario final
_kpis_df = pd.DataFrame([{"KPI": k.name, "Valor": k.display("$")} for k in cat])
_hall_df = pd.DataFrame([{"Hallazgo": h.titulo_llano, "Detalle": h.texto_llano} for h in hall])
_prob_df = pd.DataFrame([{"Severidad": i.severity, "Qué pasa": i.title} for i in i2])
_xl, _av = deployment.build_excel(
    clean, source="ventas.csv", confianza="Media — revisa los detalles",
    resumen=resumen, prep_log=log, problemas=_prob_df, kpis=_kpis_df, hallazgos=_hall_df)
check("Excel limpio generado", len(_xl) > 8000, f"{len(_xl)/1024:.0f} KB")
_req = Path(__file__).resolve().parent / "requirements.txt"
_reqs = _req.read_text().lower() if _req.exists() else ""
for _pkg in ("xlsxwriter", "xlrd", "openpyxl", "streamlit", "plotly", "scikit-learn",
             "scipy", "statsmodels", "pandas", "numpy", "joblib"):
    check(f"requirements.txt declara {_pkg}", _pkg in _reqs)
Path("/tmp/datos_limpios.xlsx").write_bytes(_xl)
_hojas = pd.ExcelFile(io.BytesIO(_xl)).sheet_names
check("Trae las cuatro hojas",
      _hojas == ["Resumen", "Datos limpios", "Qué corregimos", "Qué revisar"], str(_hojas))
_leido = pd.read_excel(io.BytesIO(_xl), sheet_name="Datos limpios")
check("Los datos del Excel cuadran con los limpios",
      len(_leido) == len(clean) and list(_leido.columns) == list(clean.columns),
      f"{_leido.shape} vs {clean.shape}")
check("El Excel trae los importes ya numéricos",
      pd.api.types.is_numeric_dtype(_leido["Importe"]), str(_leido["Importe"].dtype))
check("El Excel trae las fechas como fecha",
      pd.api.types.is_datetime64_any_dtype(_leido["Fecha"]), str(_leido["Fecha"].dtype))
check("Sin aviso de recorte con este tamaño", _av == [], str(_av))
_xl2, _av2 = deployment.build_excel(
    clean, source="x", confianza="c", resumen="r", prep_log=log,
    problemas=_prob_df, kpis=_kpis_df, hallazgos=_hall_df, max_filas=50)
check("Avisa cuando recorta filas", len(_av2) == 1 and "primeras 50" in _av2[0],
      str(_av2)[:60])

# --- gráficas: el bug del eje numérico
import charts  # noqa: E402
_f = charts.bar_ranked(["4088.0", "3211.0", "5502.0"], [430000, 240000, 215000])
check("Eje de categorías forzado", _f.layout.yaxis.type == "category", str(_f.layout.yaxis.type))
check("Etiquetas de ID sin el '.0'", _f.layout.yaxis.categoryarray == ("5502", "3211", "4088"),
      str(_f.layout.yaxis.categoryarray))
check("Cada barra trae su cifra escrita", len(_f.data[0].text) == 3, str(_f.data[0].text))
_lab, _val, _resto = charts.top_con_otros(
    pd.Series(range(1, 101), index=[f"SKU{i}" for i in range(100)]), 8)
check("Agrupa la cola en 'Otros'", _lab[-1].startswith("Otros (93"), _lab[-1])
check("'Otros' suma la cola completa",
      abs(sum(_val) - sum(range(1, 101))) < 1e-6, f"{sum(_val)}")

html = deployment.build_html_report(
    source="ventas.csv", overview=profiling.dataset_overview(clean, p2), score=s2, counts=counts,
    kpis=cat, insights=hall, issues_df=quality.issues_table(i2),
    profile_df=profiling.profiles_table(p2), prep_log=log, model_report=r_clf, summary=resumen)
check("Reporte HTML generado", len(html) > 8000, f"{len(html)/1024:.0f} KB")
check("Reporte sin marcadores markdown crudos", "**" not in html)
Path("/tmp/reporte_prueba.html").write_text(html, encoding="utf-8")

# ---------------------------------------------------------------- casos límite
seccion("Casos límite")
casos = {
    "una sola columna": pd.DataFrame({"x": [1, 2, 3, 4, 5]}),
    "todo texto": pd.DataFrame({"a": list("abcde"), "b": ["x"] * 5}),
    "una fila": pd.DataFrame({"a": [1], "b": ["x"]}),
    "todo nulo": pd.DataFrame({"a": [None] * 5, "b": [np.nan] * 5}),
    "unicode raro": pd.DataFrame({"columna con espacios": ["ñ", "é", "中文", None, "🙂"],
                                  "n": [1, 2, 3, 4, 5]}),
}
for nombre, d in casos.items():
    try:
        pp = profiling.profile_dataframe(d)
        ii = quality.detect_issues(d, pp)
        sc, _ = quality.quality_score(d, ii)
        ov = profiling.dataset_overview(d, pp)
        mp = kpi_mod.suggest_mapping(pp)
        kk = kpi_mod.compute_catalog(d, mp)
        hh = ins_mod.generate_insights(d, pp, mp, ii)
        pl, lg = prep.apply_plan(d, prep.plan_from_issues(ii))
        check(f"'{nombre}' no truena", True, f"score={sc}, kpis={len(kk)}, insights={len(hh)}")
    except Exception:
        traceback.print_exc()
        check(f"'{nombre}' no truena", False)

seccion("Resultado")
if FALLOS:
    print(f"❌ {len(FALLOS)} verificación(es) fallaron:")
    for f in FALLOS:
        print("   -", f)
    sys.exit(1)
print("✅ Todas las verificaciones pasaron.")
