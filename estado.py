"""Estado compartido, carga y limpieza automática.

Lo usan tanto el modo sencillo como el avanzado, para que ambos vean
exactamente los mismos datos y los mismos hallazgos.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

import kpis as kpi_mod
import prep
import profiling
import quality

CSS = """
<style>
  :root{--acc:#2a78d6;--acc-2:#215fa8;--line:#e6e4de;--ink:#0b0b0b;--mut:#6e6c66;}

  /* el encabezado flotante de Streamlit tapa lo primero: hay que dejarle aire */
  .block-container{padding-top:3.1rem;padding-bottom:3rem;max-width:1300px}
  h1{letter-spacing:-.02em}

  /* ---------------------------------------------------- barra superior */
  .topbar{display:flex;align-items:baseline;gap:.75rem;flex-wrap:wrap;
    border-bottom:1px solid var(--line);padding:.1rem 0 .75rem 0}
  .topbar .marca{font-size:1.02rem;font-weight:800;letter-spacing:-.01em}
  .topbar .sep{color:var(--line)}
  .topbar .arch{font-size:.93rem;color:var(--mut);font-weight:600}
  .topbar .meta{margin-left:auto;font-size:.78rem;color:var(--mut);
    letter-spacing:.02em}

  /* ------------------------------------------------ pestañas = navegación */
  .stTabs [data-baseweb="tab-list"]{gap:.15rem;border-bottom:1px solid var(--line);
    margin-bottom:.4rem}
  .stTabs [data-baseweb="tab"]{height:44px;padding:0 .95rem;font-size:.92rem;
    font-weight:600}
  .stTabs [data-baseweb="tab-highlight"]{background:var(--acc)}
  .stTabs [data-baseweb="tab-border"]{background:transparent}

  /* --------------------------------------------- encabezado de sección */
  .sec{display:flex;align-items:center;gap:.6rem;margin:1.15rem 0 .5rem 0}
  .sec .chip{background:rgba(42,120,214,.11);color:var(--acc);font-size:.64rem;
    font-weight:800;letter-spacing:.11em;padding:4px 9px;border-radius:5px;
    text-transform:uppercase;white-space:nowrap;line-height:1}
  .sec .tit{font-size:1.1rem;font-weight:700;letter-spacing:-.01em}
  .sec .sub{margin-left:auto;font-size:.8rem;color:var(--mut);text-align:right}

  /* -------------------------------------------------------- tarjetas */
  [data-testid="stVerticalBlockBorderWrapper"]{background:#fff;border-radius:12px;
    box-shadow:0 1px 2px rgba(16,16,16,.045)}
  [data-testid="stMetricValue"]{font-size:1.75rem;letter-spacing:-.025em;
    font-weight:700;line-height:1.15}
  [data-testid="stMetricLabel"]{font-size:.68rem;text-transform:uppercase;
    letter-spacing:.09em;font-weight:700;opacity:.62}

  /* ------------------------------------------------------ panel destacado */
  /* el panel ocupa todo el alto de su columna: los datos al pie quedan alineados
     con la última tarjeta de la rejilla de al lado */
  [data-testid="stHorizontalBlock"]:has(.hero){align-items:stretch}
  [data-testid="stColumn"]:has(.hero) > [data-testid="stVerticalBlock"]
    > [data-testid="stElementContainer"]{flex:1 1 auto}
  [data-testid="stColumn"]:has(.hero) [data-testid="stMarkdown"],
  [data-testid="stColumn"]:has(.hero) [data-testid="stMarkdownContainer"]{height:100%}
  [data-testid="stColumn"]:has(.hero) [data-testid="stMarkdown"] > div{
    height:100%;align-items:stretch}
  .hero{background:linear-gradient(158deg,#2a78d6 0%,#215fa8 100%);color:#fff;
    border-radius:14px;padding:1.15rem 1.25rem 1.05rem 1.25rem;height:100%;
    display:flex;flex-direction:column;min-height:200px}
  .hero .tiles{margin-top:auto}
  .hero .et{font-size:.63rem;letter-spacing:.12em;text-transform:uppercase;
    font-weight:800;opacity:.82}
  .hero .val{font-size:2.25rem;font-weight:800;letter-spacing:-.035em;
    line-height:1.08;margin:.4rem 0 .1rem 0}
  .hero .nom{font-size:.93rem;opacity:.9;margin-bottom:.95rem;line-height:1.4}
  .hero .tiles{display:flex;gap:.4rem}
  .hero .tile{flex:1;background:rgba(255,255,255,.15);border-radius:9px;
    padding:.5rem .6rem;min-width:0}
  .hero .tile .t{display:block;font-size:.58rem;letter-spacing:.09em;
    text-transform:uppercase;font-weight:800;opacity:.85}
  .hero .tile .v{display:block;font-size:1rem;font-weight:700;margin-top:.2rem;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

  /* -------------------------------------------------------- hallazgos */
  .ins{border:1px solid var(--line);border-left:4px solid var(--acc);
    background:#fff;border-radius:10px;padding:.9rem 1.05rem;margin-bottom:.55rem;
    box-shadow:0 1px 2px rgba(16,16,16,.045)}
  /* el hallazgo que va dentro de una tarjeta no necesita su propio marco */
  .ins-plano{border:none;box-shadow:none;border-left-width:4px;border-left-style:solid;
    border-radius:0;padding:.1rem 0 .2rem .8rem;margin-bottom:.6rem}
  .ins-riesgo{border-left-color:#e34948}
  .ins-oportunidad{border-left-color:#008300}
  .ins-contexto{border-left-color:#8a8a85}
  .ins-h{display:block;margin-bottom:.3rem;font-size:1rem;line-height:1.35;
    font-weight:700}
  .ins span{font-size:.93rem;opacity:.9;line-height:1.6}
  .ins span b, .lect b, .atip b{font-weight:700}

  /* -------------------------------------------------------- semáforo */
  .sem{display:flex;align-items:center;gap:.8rem;border-radius:12px;
    padding:.85rem 1.1rem;margin-bottom:.4rem}
  .sem-alta{background:rgba(0,131,0,.09);border:1px solid rgba(0,131,0,.32)}
  .sem-media{background:rgba(237,161,0,.11);border:1px solid rgba(237,161,0,.38)}
  .sem-baja{background:rgba(227,73,72,.09);border:1px solid rgba(227,73,72,.32)}
  .sem .punto{font-size:1.5rem;line-height:1}
  .sem .txt b{display:block;font-size:.98rem;margin-bottom:.1rem}
  .sem .txt span{font-size:.89rem;opacity:.85}

  .pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:.7rem;
    font-weight:600;letter-spacing:.03em}
  .p-crit{background:rgba(227,73,72,.16);color:#e34948}
  .p-adv{background:rgba(237,161,0,.18);color:#c98500}
  .p-info{background:rgba(42,120,214,.16);color:#2a78d6}
  .fix{font-size:.92rem;line-height:1.65;margin:.15rem 0}
  .lect{font-size:.9rem;line-height:1.6;opacity:.86;margin:-.35rem 0 .5rem 0}
  .atip{border-left:3px solid #eda100;background:rgba(237,161,0,.09);
    border-radius:8px;padding:.6rem .85rem;font-size:.9rem;line-height:1.55;
    margin-bottom:.4rem}
  .atip-t{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;
    opacity:.6;font-weight:700;margin:.25rem 0 .35rem 0}
</style>
"""

def esc(t) -> str:
    return html.escape(str(t))


def negritas(texto: str) -> str:
    """Convierte **x** en <b>x</b>: dentro de HTML crudo Streamlit no lee markdown."""
    partes = texto.split("**")
    return "".join(p if i % 2 == 0 else f"<b>{p}</b>" for i, p in enumerate(partes))


def sec(chip: str, titulo: str, sub: str = ""):
    """Encabezado de sección: etiqueta de color + título, como en un tablero."""
    st.markdown(
        f"<div class='sec'><span class='chip'>{esc(chip)}</span>"
        f"<span class='tit'>{esc(titulo)}</span>"
        + (f"<span class='sub'>{esc(sub)}</span>" if sub else "")
        + "</div>", unsafe_allow_html=True)


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
    "kpi_modo": "recomendados",   # recomendados | propios | ninguno
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


def olvidar_excel():
    """El Excel armado deja de valer en cuanto cambian los datos activos."""
    st.session_state.pop("_excel_listo", None)
    st.session_state.pop("_excel_avisos", None)
    # el tablero vuelve a la rejilla: la gráfica abierta era de los datos viejos
    st.session_state["_vista"] = None
    st.session_state["_gen"] = st.session_state.get("_gen", 0) + 1


def cargar(df: pd.DataFrame, source: str) -> None:
    """Guarda el archivo y aplica de una vez las correcciones seguras."""
    olvidar_excel()
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
