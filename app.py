"""Analiza tus datos — app de análisis con metodología CRISP-DM.

Dos modos sobre el mismo motor:
  · Sencillo  — subes el archivo y ves el resultado. Nada que configurar.
  · Avanzado  — las seis fases de CRISP-DM, con cada decisión en tus manos.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Analiza tus datos", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

import avanzado  # noqa: E402
import estado  # noqa: E402
import sencillo  # noqa: E402

st.markdown(estado.CSS, unsafe_allow_html=True)
estado.init()
estado.aplicar_goto()

with st.sidebar:
    st.markdown("### 📊 Analiza tus datos")
    if st.session_state.raw is not None:
        st.caption(f"**{st.session_state.source}**")
        st.session_state.currency = st.text_input(
            "Símbolo de moneda", st.session_state.currency, max_chars=4)
    st.divider()
    st.toggle(
        "Modo avanzado", key="avanzado",
        help="Abre las seis fases de CRISP-DM: mapeo de columnas, limpieza paso a paso, "
             "modelos predictivos y exportación. Para quien quiera controlar cada decisión.")
    if not st.session_state.avanzado:
        st.caption("Actívalo solo si necesitas controlar cada paso. Para ver tus números no "
                   "hace falta.")

if st.session_state.avanzado:
    avanzado.render()
else:
    sencillo.render()
