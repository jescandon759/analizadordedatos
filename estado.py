"""Estado compartido, carga y limpieza automática.

Lo usan tanto el modo sencillo como el avanzado, para que ambos vean
exactamente los mismos datos y los mismos hallazgos.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import kpis as kpi_mod
import prep
import profiling
import quality

CSS = """
<style>
  .block-container{padding-top:2rem;max-width:1250px}
  [data-testid="stMetricValue"]{font-size:1.9rem;letter-spacing:-.02em;font-weight:600}
  [data-testid="stMetricLabel"]{font-size:.8rem;text-transform:uppercase;
    letter-spacing:.05em;opacity:.7}
  h1{letter-spacing:-.02em}
  .tarjeta{border:1px solid rgba(128,128,128,.22);border-radius:12px;
    padding:1rem 1.15rem;margin-bottom:.7rem;background:rgba(128,128,128,.04)}
  .ins{border:1px solid rgba(128,128,128,.22);border-left:4px solid #2a78d6;
    border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.5rem}
  .ins-riesgo{border-left-color:#e34948}
  .ins-oportunidad{border-left-color:#008300}
  .ins-contexto{border-left-color:#8a8a85}
  .ins-h{display:block;margin-bottom:.3rem;font-size:1.02rem;line-height:1.35;
    font-weight:700}
  .ins span{font-size:.94rem;opacity:.9;line-height:1.6}
  .ins span b, .lect b, .atip b{font-weight:700}
  .sem{display:flex;align-items:center;gap:.8rem;border-radius:12px;
    padding:.9rem 1.15rem;margin-bottom:.4rem}
  .sem-alta{background:rgba(0,131,0,.10);border:1px solid rgba(0,131,0,.35)}
  .sem-media{background:rgba(237,161,0,.12);border:1px solid rgba(237,161,0,.4)}
  .sem-baja{background:rgba(227,73,72,.10);border:1px solid rgba(227,73,72,.35)}
  .sem .punto{font-size:1.6rem;line-height:1}
  .sem .txt b{display:block;font-size:1rem;margin-bottom:.1rem}
  .sem .txt span{font-size:.9rem;opacity:.85}
  .pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:.7rem;
    font-weight:600;letter-spacing:.03em}
  .p-crit{background:rgba(227,73,72,.16);color:#e34948}
  .p-adv{background:rgba(237,161,0,.18);color:#c98500}
  .p-info{background:rgba(42,120,214,.16);color:#2a78d6}
  .fix{font-size:.93rem;line-height:1.65;margin:.15rem 0}
  .lect{font-size:.92rem;line-height:1.65;opacity:.88;margin:-.5rem 0 .6rem 0}
  .atip{border-left:3px solid #eda100;background:rgba(237,161,0,.09);
    border-radius:8px;padding:.6rem .85rem;font-size:.92rem;line-height:1.6;
    margin-bottom:.45rem}
  .atip-t{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;
    opacity:.65;margin:.2rem 0 .35rem 0}
</style>
"""

DEFAULTS = {
    "raw": None,             # datos tal como vinieron
    "clean": None,           # datos después de la limpieza automática
    "usar_original": False,  # el usuario pidió ver los datos sin corregir
    "prep_log": [],
    "source": "",
    "meta": None,
    "mapping": {},
    "custom": [],
    "model": None,
    "selected_kpis": None,
    "currency": "$",
    "avanzado": False,
    "nav": None,
    "plan": None,
}


PHASES = [
    ("datos", "📁 Datos", "Carga"),
    ("negocio", "1 · Negocio y KPIs", "Business Understanding"),
    ("comprension", "2 · Comprensión de datos", "Data Understanding"),
    ("preparacion", "3 · Preparación", "Data Preparation"),
    ("dashboard", "📊 Dashboard", "Resultados"),
    ("modelado", "4 · Modelado", "Modeling"),
    ("evaluacion", "5 · Evaluación", "Evaluation"),
    ("despliegue", "6 · Despliegue", "Deployment"),
]
_ETIQUETA = {k: e for k, e, _ in PHASES}


def goto(phase_key: str):
    """Navegación diferida: no se puede tocar la clave de un widget ya pintado."""
    st.session_state["_goto"] = _ETIQUETA[phase_key]


def aplicar_goto():
    if st.session_state.get("_goto"):
        st.session_state.nav = st.session_state.pop("_goto")


def reset_downstream():
    st.session_state.clean = None
    st.session_state.prep_log = []
    st.session_state.plan = prep.PrepPlan()
    st.session_state.model = None
    st.session_state.mapping = {}
    st.session_state.custom = []
    st.session_state.selected_kpis = None
    st.session_state.usar_original = False


def init():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v() if callable(v) else v
    if st.session_state.plan is None:
        st.session_state.plan = prep.PrepPlan()
    if st.session_state.nav is None:
        st.session_state.nav = PHASES[0][1]


@st.cache_data(show_spinner=False, max_entries=8)
def analyze(df: pd.DataFrame):
    profiles = profiling.profile_dataframe(df)
    issues = quality.detect_issues(df, profiles)
    score, counts = quality.quality_score(df, issues)
    overview = profiling.dataset_overview(df, profiles)
    return profiles, issues, score, counts, overview


def active_df() -> pd.DataFrame | None:
    if st.session_state.usar_original or st.session_state.clean is None:
        return st.session_state.raw
    return st.session_state.clean


def cargar(df: pd.DataFrame, source: str) -> None:
    """Guarda el archivo y aplica de una vez las correcciones seguras."""
    st.session_state.raw = df
    st.session_state.source = source
    st.session_state.usar_original = False
    st.session_state.model = None
    st.session_state.custom = []
    st.session_state.selected_kpis = None

    profiles, issues, _, _, _ = analyze(df)
    plan = prep.plan_from_issues(issues, aggressive=False)
    st.session_state.plan = plan
    if plan.is_empty():
        st.session_state.clean = None
        st.session_state.prep_log = []
    else:
        limpio, log = prep.apply_plan(df, plan)
        st.session_state.clean = limpio
        st.session_state.prep_log = log

    base = active_df()
    prof2, _, _, _, _ = analyze(base)
    st.session_state.mapping = kpi_mod.suggest_mapping(prof2)


def mapping_actual(profiles) -> dict:
    if not st.session_state.mapping:
        st.session_state.mapping = kpi_mod.suggest_mapping(profiles)
    return st.session_state.mapping


# ---------------------------------------------------------------- confianza


def confianza(score: int, counts: dict) -> tuple[str, str, str, str]:
    """(nivel, clase css, emoji, frase) — el puntaje técnico traducido."""
    if score >= 78:
        return ("Alta", "sem-alta", "🟢",
                "Tus datos están en buena forma. Puedes confiar en los números de abajo.")
    if score >= 50:
        return ("Media", "sem-media", "🟡",
                "Los números sirven para orientarte, pero hay detalles que conviene revisar "
                "antes de tomar una decisión importante.")
    return ("Baja", "sem-baja", "🔴",
            f"Encontramos {counts.get('crítico', 0)} problema(s) serio(s) que pueden estar "
            "alterando los totales. Revísalos antes de decidir nada con estas cifras.")
