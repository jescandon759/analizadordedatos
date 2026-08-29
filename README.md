# Analiza tus datos — analizador CRISP-DM

App de Streamlit con **dos modos sobre el mismo motor**:

- **Modo sencillo (el que se abre por defecto).** Subes tu archivo y ya. La app
  limpia lo que puede limpiar sola, te dice qué corrigió, calcula tus KPIs,
  escribe los hallazgos en español claro y te deja descargar el reporte. Cero
  configuración antes de ver resultados.
- **Modo avanzado (un interruptor en el menú lateral).** Las seis fases de
  CRISP-DM con cada decisión en tus manos: mapeo de columnas, limpieza paso a
  paso con bitácora, KPIs por fórmula, modelos predictivos y exportación.

---

## Cómo correrla

```bash
pip install -r requirements.txt
streamlit run app.py
```

Se abre en `http://localhost:8501`. Sin datos a la mano, el botón
**Ver un ejemplo** genera un archivo de ventas con errores típicos.

### Desplegar en Streamlit Community Cloud

1. Sube todos los archivos `.py` y `requirements.txt` a un repositorio de GitHub,
   en la raíz (no dentro de una carpeta).
2. En [share.streamlit.io](https://share.streamlit.io) → *Create app* → *Main file
   path*: `app.py`.

> **Ojo:** en Community Cloud la app es pública. Cualquiera con el enlace puede
> subir archivos. No la uses con datos confidenciales sin ponerle autenticación.

---

## Qué ve el usuario en el modo sencillo

1. **Qué corregimos** — un aviso de cuántas cosas se arreglaron solas, el detalle
   completo en un desplegable, y un botón para ver los datos sin tocar.
2. **Confianza en los datos** — semáforo Alta / Media / Baja, con una frase que
   explica qué significa. Sin puntajes técnicos.
3. **Tus números** — hasta 6 KPIs detectados automáticamente, con su variación
   contra el periodo anterior.
4. **Lo más importante** — los hallazgos ordenados por impacto, cada uno con su
   gráfica y redactado sin jerga.
5. **Tus gráficas** — evolución en el tiempo y ranking por categoría.
6. **Qué revisar en tus datos** — los problemas separados entre «ya lo
   corregimos» y «necesita tu criterio».
7. **Debajo de cada gráfica** — qué significa lo que se ve (mejor periodo, peor,
   promedio, hacia dónde va) y, si hay valores fuera de lo normal, cuál es, qué tan
   lejos está de lo normal y **de dónde salió**.
8. **¿Interpretamos bien tus columnas?** — por si la detección automática falló.
9. **Llévatelo** — reporte HTML y datos corregidos en CSV.

---

## Cómo se explican los valores atípicos

La app **no puede saber la causa** de un pico y no la inventa. Lo que hace es
descomponerlo con hechos verificables:

- **¿Fue volumen o tamaño?** "Hubo 214 operaciones contra las 62 de un mes normal:
  fue volumen, no operaciones más grandes."
- **¿Una sola operación lo explica?** "Una sola operación de $182.1K (CLI-0001,
  Cuishe 750) explica el 29.7% de todo el mes. Vale la pena confirmar que la cifra
  sea correcta."
- **¿Se concentró en una categoría?** "Se concentró en Mayoreo: 71% del mes, cuando
  normalmente representa 24%."
- **¿Es temporada?** "Es diciembre, que año con año es tu mes más fuerte (18% del
  total anual). Probablemente sea temporada, no algo excepcional."

La detección usa mediana y desviación absoluta mediana (MAD), no promedio y
desviación estándar: el promedio se deja arrastrar por el propio pico que se busca.
Debajo de cada bloque queda escrito que son pistas de origen, no causas.

---

## Las seis fases

| Fase | En la app | Qué hace |
|---|---|---|
| 1 · Business Understanding | **Negocio y KPIs** | Mapeas qué columna es la fecha, el importe, el cliente… De ahí salen los KPIs y los insights. Catálogo automático + KPIs propios por fórmula. |
| 2 · Data Understanding | **Comprensión de datos** | Perfila cada columna (tipo real, no dtype) y detecta ~20 clases de error con severidad y alcance. Puntaje de calidad 0–100. |
| 3 · Data Preparation | **Preparación** | Reparaciones seleccionables con **bitácora** de cada transformación. El archivo original nunca se toca. |
| — | **Dashboard** | KPIs, hallazgos priorizados por impacto y exploración interactiva. |
| 4 · Modeling | **Modelado** | Clasificación, regresión, segmentación, anomalías o pronóstico. |
| 5 · Evaluation | **Evaluación** | Métricas **contra un baseline obligatorio**, detección de fuga de información, matriz de confusión, residuos, importancia por permutación. |
| 6 · Deployment | **Despliegue** | Datos limpios (CSV/Excel), reporte HTML, modelo `.joblib` y scoring de archivos nuevos. |

---

## Qué errores detecta

**Estructura** — filas y columnas duplicadas, filas vacías, columnas constantes o
vacías, identificadores repetidos, encabezados desplazados.

**Tipos** — números guardados como texto (`"$ 1,234.50"`), fechas como texto, tipos
mezclados en una misma columna.

**Contenido** — nulos por columna con umbral de severidad, espacios sobrantes,
categorías escritas de varias formas (`Norte` / `norte ` / `NORTE`), mojibake
(`Ã±` por `ñ`), correos y teléfonos mal formados.

**Valores** — outliers por rango intercuartílico, distribuciones muy sesgadas,
negativos en columnas monetarias, exceso de ceros, porcentajes fuera de 0–100.

**Fechas** — futuras, imposibles (1900-01-01), huecos en la serie.

**Redundancia** — columnas con correlación > 0.98, alta cardinalidad.

---

## Insights: reglas, no adivinanzas

Cada hallazgo sale de una prueba estadística o un umbral explícito, y el texto se
arma con los números reales del archivo. **No hay LLM ni generación de texto libre**:
el mismo archivo produce siempre el mismo diagnóstico y todo es verificable.

- **Tendencia** — regresión lineal sobre la serie agregada, con R² y p-valor.
- **Periodo anómalo** — z-score del último periodo *cerrado* contra su historia.
- **Estacionalidad** — por día de semana y por mes, contra la distribución uniforme.
- **Concentración (Pareto)** — cuánto aporta el 20% superior; alerta si un solo
  elemento pasa del 30%.
- **Cola larga** — cuántos elementos aportan menos del 5% del total.
- **Segmentos que discriminan** — ANOVA entre categorías; también reporta cuando
  **no** hay diferencia significativa.
- **Crecimiento/caída por segmento** — primera mitad del periodo contra la segunda.
- **Dispersión** — coeficiente de variación y peso del 1% superior.
- **Correlaciones** — pares fuertes, con la advertencia de que correlación ≠ causalidad.
- **Actualidad y cobertura** — hace cuánto que no hay datos.

---

## KPIs propios

Escribe una fórmula con nombres de columna entre comillas:

```
(suma("Importe") - suma("Costo")) / suma("Importe")
suma("Importe") / unicos("Cliente")
suma_si("Importe", "Canal", "Mayoreo") / suma("Importe")
conteo_si("Estatus", "Cancelado") / conteo()
percentil("Importe", 90)
```

Funciones: `suma`, `promedio`, `mediana`, `minimo`, `maximo`, `conteo`, `unicos`,
`desviacion`, `percentil`, `suma_si`, `conteo_si`, `raiz`, `abs`.

La fórmula se evalúa con un intérprete restringido sobre el AST de Python: solo
aritmética y esas funciones. `__import__`, `open`, `exec` y cualquier otra cosa se
rechazan antes de ejecutarse.

---

## Los guardarraíles del modelado

Sobre datos arbitrarios es fácil producir un modelo con métricas bonitas y
conclusiones falsas. La app hace tres cosas que la mayoría de las herramientas
automáticas omite:

1. **Baseline obligatorio.** Entrena un modelo tonto (clase más frecuente / promedio)
   y compara contra él. Si tu modelo no le gana con holgura, la app lo dice con
   todas sus letras: *"El modelo no le gana a la regla tonta. No lo pongas en
   producción."*
2. **Detección de fuga de información.** Busca variables que contengan la respuesta
   —una columna con correlación 1.0 con el objetivo, o una categórica que separa
   las clases sin traslape— y bloquea la interpretación de las métricas hasta que
   las quites.
3. **Exclusiones explícitas.** Identificadores, texto libre y columnas constantes se
   descartan y se reportan. Las clases con menos de 5 casos se excluyen porque no
   se pueden validar.

Además: validación cruzada además del holdout, importancia por permutación (no la
del árbol, que sobreestima variables de alta cardinalidad), y avisos cuando hay
pocos datos, clases desbalanceadas o desempeño inestable entre particiones.

---

## Estructura

Todo vive en la raíz, sin carpetas (así lo necesita Streamlit Cloud cuando se
suben los archivos por el navegador).

```
app.py               Router: decide modo sencillo o avanzado
sencillo.py          La pantalla única
avanzado.py          Las seis fases de CRISP-DM
estado.py            Estado compartido, carga y limpieza automática
textos.py            Traducción de lo técnico a lenguaje de negocio
loader.py            Lectura robusta (encoding, separador, hoja, encabezado)
profiling.py         Perfilado y tipos semánticos
quality.py           Motor de detección de errores + puntaje
prep.py              Transformaciones con bitácora
kpis.py              Catálogo de KPIs + intérprete de fórmulas
insights.py          Motor de reglas de negocio (texto técnico y texto llano)
modeling.py          AutoML, baselines, detección de fuga, evaluación
charts.py            Gráficas Plotly con paleta validada para daltonismo
deployment.py        Exportación, empaquetado de modelo, reporte HTML
utils.py             Coerción de tipos, formato, normalización de texto
demo.py              Generador de datos sucios de ejemplo
test_pipeline.py     Prueba end-to-end del núcleo (sin Streamlit)
test_ui.py           Prueba de ambos modos con streamlit.testing
```

Correr las pruebas:

```bash
python test_pipeline.py
python test_ui.py
```

---

## Formatos y errores de lectura

Acepta **CSV, TSV, .xlsx, .xlsm y .xls**. El formato viejo `.xls` (Excel 97-2003)
necesita el paquete `xlrd`, que ya viene en `requirements.txt` — si lo quitas, la
app te dirá exactamente eso y te propondrá guardar el archivo como `.xlsx`.

Los errores de lectura nunca se muestran crudos: se traducen a qué pasó y qué hacer
(formato viejo, archivo protegido con contraseña, codificación rara, columnas
desparejas, archivo demasiado grande). El detalle técnico queda en un desplegable
por si hace falta.

---

## Límites conocidos

- **Todo corre en memoria.** Archivos por arriba de ~500 MB no van a funcionar bien.
  Para eso necesitas una base de datos, no esta app.
- **El pronóstico es mensual** y necesita al menos 12 meses de historia; con menos,
  la app se niega en vez de inventar.
- **Los insights no entienden tu negocio.** Detectan patrones estadísticos. Que un
  segmento crezca 40% es un hecho; que sea buena noticia lo decides tú.
- **Correlación no es causalidad.** La app lo repite donde corresponde, pero conviene
  no olvidarlo al leer el dashboard.
