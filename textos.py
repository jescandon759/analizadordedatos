"""Traducción de los hallazgos técnicos a lenguaje de negocio.

Un solo lugar donde vive el vocabulario que ve el usuario final.
"""
from __future__ import annotations

from utils import fmt_pct

# ---------------------------------------------------------------- problemas

# code -> (¿se puede corregir solo?, plantilla en lenguaje llano)
PROBLEMAS = {
    "filas_duplicadas": (True, "Había **{n} filas repetidas** exactamente igual. "
                               "Estaban inflando todos tus totales."),
    "filas_vacias": (True, "**{n} filas** venían completamente en blanco."),
    "sin_filas": (False, "El archivo no tiene ningún registro de datos."),
    "pocos_registros": (False, "Solo hay **{n} registros**. Con tan pocos datos, cualquier "
                               "conclusión es frágil."),
    "id_duplicado": (False, "En **{col}** hay {n} claves repetidas, y deberían ser únicas. "
                            "Casi siempre es doble captura o un cruce mal hecho."),
    "nulos": (False, "A **{col}** le faltan {n} valores ({pct} de los registros)."),
    "columna_vacia": (True, "La columna **{col}** está completamente vacía."),
    "columna_constante": (False, "**{col}** tiene siempre el mismo valor, así que no sirve "
                                 "para comparar nada."),
    "numero_como_texto": (True, "**{col}** traía los números guardados como texto (con signo "
                                "de pesos o comas). Así no se podían sumar."),
    "fecha_como_texto": (True, "**{col}** traía las fechas guardadas como texto. Sin eso no "
                               "hay forma de ver cómo evolucionan las cosas en el tiempo."),
    "tipo_mixto": (False, "**{col}** mezcla números con texto (cosas como «N/D» o "
                          "«pendiente»). Conviene decidir una sola convención."),
    "espacios": (True, "**{col}** tenía {n} valores con espacios de más. Eso parte una misma "
                       "categoría en dos sin que se note."),
    "mojibake": (True, "**{col}** tenía {n} textos con los acentos rotos (se ven como «Ã±»). "
                       "Pasa cuando el archivo se guardó con otra codificación."),
    "categorias_inconsistentes": (True, "En **{col}** el mismo valor está escrito de varias "
                                        "formas (mayúsculas, acentos o espacios). Se contaban "
                                        "como distintos y partían tus totales."),
    "email_invalido": (False, "**{col}** tiene {n} correos con formato inválido."),
    "telefono_invalido": (False, "**{col}** tiene {n} teléfonos con formato irregular."),
    "negativos_sospechosos": (False, "**{col}** tiene {n} valores negativos. Pueden ser "
                                     "devoluciones legítimas o errores de signo — conviene "
                                     "distinguirlos antes de sumar."),
    "celda_rara": (False, "**{col}** tiene {n} celda(s) con una forma distinta al resto de la "
                          "columna (por ejemplo, un dígito donde todas traen cuatro). Suele ser "
                          "un dato incompleto o pegado en el lugar equivocado. Abajo te decimos "
                          "en qué celda están."),
    "valor_raro": (False, "**{col}** tiene {n} registro(s) fuera de escala: se salen tanto del "
                          "resto que casi siempre son un error de captura (un cero de más, "
                          "una carga de prueba o dos campos que se cruzaron). Abajo te decimos "
                          "en qué fila están."),
    "outliers": (False, "**{col}** tiene {n} valores muchísimo más grandes o más chicos que "
                        "el resto. Revísalos: un cero de más cambia todos tus promedios."),
    "asimetria": (False, "**{col}** está muy desbalanceada: el promedio no representa al "
                         "caso típico. Mejor usa la mediana."),
    "porcentaje_fuera_rango": (False, "**{col}** tiene {n} porcentajes fuera del rango 0–100."),
    "muchos_ceros": (False, "**{col}** tiene {n} registros en cero. Muchas veces es un campo "
                            "que se dejó vacío y se guardó como cero, y eso hunde los promedios."),
    "fechas_futuras": (False, "**{col}** tiene {n} fechas posteriores a hoy. Puede ser algo "
                              "programado o un error de captura."),
    "fechas_imposibles": (True, "**{col}** tenía {n} fechas imposibles (año 1900 o antes), "
                                "que es lo que aparece cuando el campo se deja vacío."),
    "hueco_temporal": (False, "Hay un periodo largo sin ningún registro en **{col}**. Puede "
                              "que falte información de esos días."),
    "columnas_redundantes": (False, "Dos de tus columnas son prácticamente la misma cosa."),
    "alta_cardinalidad": (False, "**{col}** tiene demasiados valores distintos como para "
                                 "graficarla o usarla para agrupar."),
}


def explicar(issue) -> str:
    """Frase en lenguaje llano para un problema detectado."""
    entrada = PROBLEMAS.get(issue.code)
    if entrada is None:
        return f"**{issue.column or 'Tus datos'}**: {issue.title.lower()}."
    _, plantilla = entrada
    return plantilla.format(
        col=issue.column or "tus datos",
        n=f"{issue.n_affected:,}" if issue.n_affected else "algunos",
        pct=fmt_pct(issue.pct_affected) if issue.pct_affected else "",
    )


def es_corregible(issue) -> bool:
    entrada = PROBLEMAS.get(issue.code)
    return bool(issue.fix) and bool(entrada and entrada[0])


# ---------------------------------------------------------------- KPIs

# los KPIs que se muestran primero en el modo sencillo, en este orden
PRIORIDAD_KPI = [
    "Ingreso total", "Utilidad bruta", "Margen bruto", "Transacciones",
    "Ticket promedio", "Clientes únicos", "Unidades", "Registros",
    "Ingreso por cliente", "Productos distintos", "Costo total",
    "Precio promedio por unidad", "Ticket mediano", "Concentración top 10% clientes",
]


def ordenar_kpis(kpis: list, limite: int = 6) -> list:
    orden = {n: i for i, n in enumerate(PRIORIDAD_KPI)}
    return sorted(kpis, key=lambda k: orden.get(k.name, 99))[:limite]


# ---------------------------------------------------------------- resumen


def resumen(insights, overview, score) -> str:
    riesgos = [i for i in insights if i.kind == "riesgo"]
    oport = [i for i in insights if i.kind == "oportunidad"]
    partes = [f"Revisamos **{overview['filas']:,} registros** y "
              f"**{overview['columnas']} columnas**."]
    if oport:
        partes.append(f"Lo que más se puede aprovechar: **{oport[0].titulo_llano}**.")
    if riesgos:
        partes.append(f"Lo que más urge atender: **{riesgos[0].titulo_llano}**.")
    if not riesgos and not oport:
        partes.append("No encontramos patrones destacables. Si tu archivo tiene una columna de "
                      "fecha y una de importe, indícalas abajo en «¿Interpretamos bien tus "
                      "columnas?» para sacarle más provecho.")
    return " ".join(partes)
