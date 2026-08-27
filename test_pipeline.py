"""Prueba end-to-end sin Streamlit: recorre las seis fases sobre datos sucios."""
from __future__ import annotations

import io
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import deployment, insights as ins_mod, kpis as kpi_mod, loader, prep, profiling, quality
from core.demo import build_demo
from core.modeling import run_anomalies, run_clustering, run_forecast, run_supervised

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
