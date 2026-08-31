"""Prueba de la interfaz con AppTest: modo sencillo y modo avanzado."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

FALLOS: list[str] = []
FASES = ["📁 Datos", "1 · Negocio y KPIs", "2 · Comprensión de datos", "3 · Preparación",
         "📊 Dashboard", "4 · Modelado", "5 · Evaluación", "6 · Despliegue"]


def check(label: str, cond: bool, extra: str = ""):
    print(("  ✓ " if cond else "  ✗ ") + label + (f" — {extra}" if extra else ""))
    if not cond:
        FALLOS.append(label)


def err(at) -> str:
    return str(at.exception[0].message) if at.exception else ""


def nueva(timeout: int = 240) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=timeout)
    at.run()
    return at


# ------------------------------------------------------------------ sencillo
print("=== Modo sencillo: pantalla de inicio ===")
at = nueva()
check("Arranca sin excepción", not at.exception, err(at))
check("Muestra el título de bienvenida",
      any("Analiza tus datos" in m.value for m in at.markdown))
check("Ofrece subir archivo", len(at.get("file_uploader")) == 1)
ejemplo = [b for b in at.button if "ejemplo" in b.label.lower()]
check("Ofrece el ejemplo", len(ejemplo) == 1)
check("Sin fases visibles al inicio", len(at.sidebar.radio) == 0)

print("\n=== Un solo clic: del archivo al resultado ===")
ejemplo[0].click().run()
check("Carga y analiza sin error", not at.exception, err(at))
check("Limpió automáticamente", at.session_state["clean"] is not None,
      f"{len(at.session_state['prep_log'])} acciones")
check("Detectó las columnas de negocio",
      at.session_state["mapping"].get("ingreso") == "Importe",
      str({k: v for k, v in at.session_state["mapping"].items() if v}))
check("Muestra los KPIs", len(at.metric) >= 4, f"{len(at.metric)} métricas")
check("Muestra el semáforo de confianza",
      any("Confianza en los datos" in m.value for m in at.markdown))
check("Muestra hallazgos", any("Lo más importante" in m.value for m in at.markdown))
check("Avisa qué corrigió", any("Corregimos" in s.value for s in at.success), )
botones = at.get("download_button")
check("Ofrece las tres descargas", len(botones) == 3, f"{len(botones)}")
check("La descarga principal es el Excel limpio",
      any("Excel limpio" in b.label for b in botones),
      str([b.label for b in botones]))
check("Sigue sin fases a la vista", len(at.sidebar.radio) == 0)
textos_md = " ".join(m.value for m in at.markdown)
for jerga in ["ANOVA", "p-valor", "R²", "silueta", "desviaciones estándar", "baseline"]:
    check(f"Sin jerga en pantalla: '{jerga}'", jerga not in textos_md)

check("Explica qué significa cada gráfica",
      textos_md.count("Cómo leerla") >= 2, f"{textos_md.count('Cómo leerla')} lecturas")
check("Señala los valores fuera de lo normal",
      "fuera de lo normal" in textos_md or "se salió de lo normal" in textos_md)

print("\n=== Interacciones del modo sencillo ===")
sin_corregir = [b for b in at.button if "sin corregir" in b.label]
check("Puede ver los datos originales", len(sin_corregir) == 1)
if sin_corregir:
    sin_corregir[0].click().run()
    check("Alterna a datos sin corregir",
          at.session_state["usar_original"] and not at.exception, err(at))
    volver = [b for b in at.button if "corregidos" in b.label]
    if volver:
        volver[0].click().run()
        check("Vuelve a los corregidos", not at.session_state["usar_original"] and not at.exception,
              err(at))

sel = [s for s in at.selectbox if s.label and "Importe" in (s.label or "")]
mapa = [s for s in at.selectbox if s.key == "s_segmento"]
check("Permite corregir el mapeo de columnas", len(mapa) == 1)
if mapa:
    mapa[0].set_value("Producto").run()
    check("Recalcula al cambiar una columna",
          at.session_state["mapping"]["segmento"] == "Producto" and not at.exception, err(at))

print("\n=== Modo avanzado ===")
at.sidebar.toggle[0].set_value(True).run()
check("Abre el modo avanzado", not at.exception, err(at))
check("Aparecen las fases", len(at.sidebar.radio) == 1)
for f in FASES:
    at.sidebar.radio[0].set_value(f).run()
    check(f"Fase '{f}'", not at.exception, err(at))

check("Conserva los datos cargados", at.session_state["raw"] is not None)
at.sidebar.toggle[0].set_value(False).run()
check("Regresa al modo sencillo", not at.exception and len(at.sidebar.radio) == 0, err(at))
check("Los datos siguen ahí", at.session_state["raw"] is not None)

print("\n=== Modo avanzado desde cero ===")
at2 = nueva()
at2.sidebar.toggle[0].set_value(True).run()
check("Avanzado sin datos no truena", not at2.exception, err(at2))
for f in FASES[1:]:
    at2.sidebar.radio[0].set_value(f).run()
    check(f"'{f}' sin datos", not at2.exception, err(at2))

print("\n=== Resultado ===")
if FALLOS:
    print(f"❌ {len(FALLOS)} verificación(es) fallaron:")
    for f in FALLOS:
        print("   -", f)
    sys.exit(1)
print("✅ Interfaz verificada.")
