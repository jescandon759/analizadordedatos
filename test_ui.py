"""Prueba de la interfaz Streamlit con AppTest: recorre las 8 secciones sin errores."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from core.demo import build_demo  # noqa: E402

FALLOS: list[str] = []
SECCIONES = ["📁 Datos", "1 · Negocio y KPIs", "2 · Comprensión de datos", "3 · Preparación",
             "📊 Dashboard", "4 · Modelado", "5 · Evaluación", "6 · Despliegue"]


def check(label: str, cond: bool, extra: str = ""):
    print(("  ✓ " if cond else "  ✗ ") + label + (f" — {extra}" if extra else ""))
    if not cond:
        FALLOS.append(label)


def nueva_app(timeout: int = 180) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=timeout)
    at.run()
    return at


print("=== Arranque en frío ===")
at = nueva_app()
check("La app arranca sin excepción", not at.exception,
      str(at.exception[0].message) if at.exception else "")
check("Muestra la pantalla de carga", any("Carga de datos" in t.value for t in at.title))

print("\n=== Carga de datos de ejemplo ===")
btn = [b for b in at.button if "ejemplo" in b.label]
check("Existe el botón de datos de ejemplo", len(btn) == 1)
btn[0].click().run()
check("Datos cargados sin error", not at.exception,
      str(at.exception[0].message) if at.exception else "")
check("Sesión tiene el dataframe", at.session_state["raw"] is not None,
      f"{len(at.session_state['raw'])} filas")

print("\n=== Recorrido de las 8 secciones ===")
for sec in SECCIONES:
    at.sidebar.radio[0].set_value(sec).run()
    ok = not at.exception
    check(f"Sección '{sec}'", ok, str(at.exception[0].message) if at.exception else "")

print("\n=== Fase 3: aplicar limpieza ===")
at.sidebar.radio[0].set_value("3 · Preparación").run()
seguras = [b for b in at.button if "seguras" in b.label]
check("Botón de reparaciones seguras presente", len(seguras) == 1)
if seguras:
    seguras[0].click().run()
    check("Selección de reparaciones sin error", not at.exception,
          str(at.exception[0].message) if at.exception else "")
    aplicar = [b for b in at.button if "Aplicar" in b.label]
    check("Botón aplicar presente", len(aplicar) == 1)
    if aplicar:
        aplicar[0].click().run()
        check("Limpieza aplicada", at.session_state["clean"] is not None
              and not at.exception, str(at.exception[0].message) if at.exception else "")
        check("Bitácora registrada", len(at.session_state["prep_log"]) > 3,
              f"{len(at.session_state['prep_log'])} líneas")

print("\n=== Dashboard con datos limpios ===")
at.sidebar.radio[0].set_value("📊 Dashboard").run()
check("Dashboard renderiza", not at.exception,
      str(at.exception[0].message) if at.exception else "")
check("Muestra métricas", len(at.metric) >= 4, f"{len(at.metric)} métricas")

print("\n=== KPI personalizado ===")
at.sidebar.radio[0].set_value("1 · Negocio y KPIs").run()
nombre_in = [i for i in at.text_input if i.label == "Nombre del KPI"]
formula_in = [i for i in at.text_input if i.label == "Fórmula"]
if nombre_in and formula_in:
    nombre_in[0].set_value("Margen bruto propio")
    formula_in[0].set_value('(suma("Importe") - suma("Costo")) / suma("Importe")')
    subs = ([b for b in at.button if "Agregar KPI" in b.label]
            or [b for b in at.get("form_submit_button") if "Agregar KPI" in b.label])
    check("Botón de envío del formulario presente", len(subs) >= 1)
    if subs:
        subs[0].click().run()
    check("KPI propio agregado", len(at.session_state["custom"]) == 1,
          str(at.session_state["custom"]))
    check("Sin error al calcular el KPI propio", not at.exception,
          str(at.exception[0].message) if at.exception else "")
else:
    check("Formulario de KPI propio disponible", False,
          str([i.label for i in at.text_input]))

print("\n=== Modelado y evaluación ===")
at.sidebar.radio[0].set_value("4 · Modelado").run()
sel = [s for s in at.selectbox if "objetivo" in (s.label or "")]
check("Selector de objetivo presente", len(sel) >= 1)
if sel:
    sel[0].set_value("Canal").run()
    entrenar = [b for b in at.button if "Entrenar" in b.label]
    if entrenar:
        entrenar[0].click().run()
        check("Modelo entrenado sin error", not at.exception,
              str(at.exception[0].message) if at.exception else "")
        rep = at.session_state["model"]
        check("Reporte de modelo en sesión", rep is not None and rep.ok,
              rep.headline if rep else "")
        at.sidebar.radio[0].set_value("5 · Evaluación").run()
        check("Evaluación renderiza", not at.exception,
              str(at.exception[0].message) if at.exception else "")

print("\n=== Despliegue ===")
at.sidebar.radio[0].set_value("6 · Despliegue").run()
check("Despliegue renderiza", not at.exception,
      str(at.exception[0].message) if at.exception else "")
check("Ofrece descargas", len(at.get("download_button")) >= 2,
      f"{len(at.get('download_button'))} botones")

print("\n=== Sin archivo cargado ===")
at2 = nueva_app()
for sec in SECCIONES[1:]:
    at2.sidebar.radio[0].set_value(sec).run()
    check(f"'{sec}' sin datos no truena", not at2.exception,
          str(at2.exception[0].message) if at2.exception else "")

print("\n=== Resultado ===")
if FALLOS:
    print(f"❌ {len(FALLOS)} verificación(es) fallaron:")
    for f in FALLOS:
        print("   -", f)
    sys.exit(1)
print("✅ Interfaz verificada de extremo a extremo.")
