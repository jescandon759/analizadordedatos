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
check("Ofrece las descargas directas", len(botones) == 2, f"{len(botones)}")
prep_excel = [b for b in at.button if "Preparar Excel" in b.label]
check("El Excel se arma solo cuando se pide", len(prep_excel) == 1,
      str([b.label for b in at.button]))
if prep_excel:
    prep_excel[0].click().run()
    check("Al pedirlo, arma el Excel sin error", not at.exception, err(at))
    check("Aparece la descarga del Excel",
          any("Descargar Excel limpio" in b.label for b in at.get("download_button")),
          str([b.label for b in at.get("download_button")]))
check("Sigue sin fases a la vista", len(at.sidebar.radio) == 0)
# el bloque <style> no es texto que el usuario lea: no cuenta para la jerga
textos_md = " ".join(m.value for m in at.markdown if "<style>" not in m.value)
for jerga in ["ANOVA", "p-valor", "R²", "silueta", "desviaciones estándar", "baseline"]:
    check(f"Sin jerga en pantalla: '{jerga}'", jerga not in textos_md)

print("\n=== Tablero: rejilla y vista a detalle ===")
detalle = [b for b in at.button if "Ver a detalle" in b.label]
check("El tablero muestra varias gráficas juntas", len(detalle) >= 3,
      f"{len(detalle)} gráficas")
if detalle:
    detalle[0].click().run()
    check("Se abre la gráfica a detalle", not at.exception, err(at))
    det = " ".join(m.value for m in at.markdown if "<style>" not in m.value)
    check("A detalle explica cómo leerla", "Cómo leerla" in det)
    check("A detalle señala los valores fuera de lo normal",
          "fuera de lo normal" in det or "se salió de lo normal" in det)
    check("A detalle da cifras de apoyo", len(at.metric) >= 3, f"{len(at.metric)} cifras")
    check("A detalle trae la tabla de datos", len(at.get("dataframe")) >= 1)
    volver = [b for b in at.button if "Volver al tablero" in b.label]
    check("Ofrece volver al tablero", len(volver) == 1)
    if volver:
        volver[0].click().run()
        check("Vuelve a la rejilla sin quedarse atorado",
              not at.exception and len([b for b in at.button
                                        if "Ver a detalle" in b.label]) >= 3, err(at))

print("\n=== Indicadores: recomendados / propios / ninguno ===")
sel_modo = [r for r in at.radio if r.key == "kpi_modo"]
check("Ofrece elegir qué indicadores ver", len(sel_modo) == 1)
if sel_modo:
    check("Por defecto usa los recomendados",
          at.session_state["kpi_modo"] == "recomendados")
    n_metricas = len(at.metric)

    sel_modo[0].set_value("ninguno").run()
    check("Se pueden apagar por completo", len(at.metric) == 0 and not at.exception,
          f"{len(at.metric)} métricas · {err(at)}")

    at.radio(key="kpi_modo").set_value("propios").run()
    check("Modo propio sin error", not at.exception, err(at))
    check("Ofrece el constructor de indicadores",
          any("Crear un indicador" in e.label for e in at.get("expander")),
          str([e.label for e in at.get("expander")]))
    nombre_in = [i for i in at.text_input if "se llama tu indicador" in (i.label or "")]
    check("Pide nombre en lenguaje llano", len(nombre_in) == 1,
          str([i.label for i in at.text_input]))
    op_sel = [s_ for s_ in at.selectbox if "¿Qué quieres calcular?" == (s_.label or "")]
    check("Elige la operación de un menú, sin fórmulas", len(op_sel) == 1)
    if nombre_in and op_sel:
        nombre_in[0].set_value("Margen propio")
        op_sel[0].set_value("margen")
        at.run()
        cols = [s_ for s_ in at.selectbox if (s_.label or "").startswith("¿De qué columna?")]
        segunda = [s_ for s_ in at.selectbox if "Menos esta columna" in (s_.label or "")]
        if cols and segunda:
            cols[0].set_value("Importe")
            segunda[0].set_value("Costo")
        agregar = [b_ for b_ in at.button if "Agregar indicador" in b_.label]
        if agregar:
            agregar[0].click().run()
            check("Crea el indicador sin escribir fórmulas",
                  len(at.session_state["custom"]) == 1 and not at.exception,
                  str(at.session_state["custom"])[:90])
            check("La fórmula generada es la correcta",
                  at.session_state["custom"][0].formula ==
                  '(suma("Importe") - suma("Costo")) / suma("Importe")',
                  at.session_state["custom"][0].formula if at.session_state["custom"] else "")
            check("El indicador propio se muestra", len(at.metric) >= 1, f"{len(at.metric)}")

            # y ahora el segundo: se pueden tener varios propios a la vez
            check("Invita a agregar otro",
                  any("Agregar otro indicador" in e.label for e in at.get("expander")),
                  str([e.label for e in at.get("expander")]))
            n2 = [i for i in at.text_input if "se llama tu indicador" in (i.label or "")]
            o2 = [s_ for s_ in at.selectbox if "¿Qué quieres calcular?" == (s_.label or "")]
            if n2 and o2:
                n2[0].set_value("Ventas totales")
                o2[0].set_value("sumar")
                at.run()
                c2 = [s_ for s_ in at.selectbox if (s_.label or "").startswith("¿De qué columna?")]
                if c2:
                    c2[0].set_value("Importe")
                b2 = [b_ for b_ in at.button if "Agregar indicador" in b_.label]
                if b2:
                    b2[0].click().run()
                    check("Se pueden tener varios indicadores propios",
                          len(at.session_state["custom"]) == 2 and not at.exception,
                          str([k.name for k in at.session_state["custom"]]))
                    check("Los dos se pintan en la rejilla", len(at.metric) >= 2,
                          f"{len(at.metric)} métricas")
                    check("Los lista para poder quitarlos",
                          any("Tus indicadores (2)" in e.label for e in at.get("expander")),
                          str([e.label for e in at.get("expander")]))
                    # nombre repetido: se rechaza para no confundir dos tarjetas iguales
                    n3 = [i for i in at.text_input if "se llama tu indicador" in (i.label or "")]
                    if n3:
                        n3[0].set_value("Ventas totales")
                        at.run()
                        b3 = [b_ for b_ in at.button if "Agregar indicador" in b_.label]
                        if b3:
                            b3[0].click().run()
                            check("No admite dos indicadores con el mismo nombre",
                                  len(at.session_state["custom"]) == 2,
                                  str([k.name for k in at.session_state["custom"]]))

    at.radio(key="kpi_modo").set_value("recomendados").run()
    check("Vuelve a los recomendados", len(at.metric) >= 4 and not at.exception,
          f"{len(at.metric)} métricas")

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
