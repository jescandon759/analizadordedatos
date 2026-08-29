"""Fases 4 y 5 — Modeling y Evaluation.

Principio de diseño: sobre datos arbitrarios, un modelo automático produce con
facilidad métricas buenas y conclusiones falsas. Por eso aquí SIEMPRE se entrena
un baseline tonto, se compara contra él, y se buscan fugas de información antes
de reportar nada. Si el modelo no le gana al baseline, la app lo dice.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import fmt_num, fmt_pct, safe_div, to_datetime_series, to_numeric_series

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CLASIFICACION, REGRESION, CLUSTERING, ANOMALIAS, FORECAST = (
    "clasificación", "regresión", "segmentación", "anomalías", "pronóstico"
)


@dataclass
class ModelReport:
    task: str
    ok: bool
    headline: str
    verdict: str                          # veredicto honesto sobre utilidad
    metrics: dict = field(default_factory=dict)
    baseline: dict = field(default_factory=dict)
    beats_baseline: bool | None = None
    warnings: list[str] = field(default_factory=list)
    importance: pd.DataFrame | None = None
    extra: dict = field(default_factory=dict)
    model: object | None = None
    feature_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- preparación


def usable_features(df: pd.DataFrame, profiles: dict, target: str | None) -> tuple[list[str], list[str], list[str]]:
    """Devuelve (numéricas, categóricas, descartadas_con_motivo)."""
    num, cat, drop = [], [], []
    for c in df.columns:
        if c == target:
            continue
        p = profiles.get(c)
        if p is None:
            continue
        if p.semantic in ("vacia", "constante"):
            drop.append(f"{c} (sin variación)")
        elif p.semantic == "identificador" or p.role == "identificador":
            drop.append(f"{c} (identificador)")
        elif p.semantic == "fecha":
            drop.append(f"{c} (fecha: se usan sus componentes)")
        elif p.semantic == "texto":
            drop.append(f"{c} (texto libre)")
        elif p.semantic == "numerico":
            num.append(c)
        elif p.semantic in ("categorico", "booleano"):
            if p.n_unique > 50:
                drop.append(f"{c} ({p.n_unique} categorías)")
            else:
                cat.append(c)
    return num, cat, drop


class FrameCoercer(BaseEstimator, TransformerMixin):
    """Normaliza el DataFrame antes del preprocesador.

    Convierte a número las columnas que traen moneda o separadores de miles como
    texto, y expande cada fecha en año / mes / día de semana. Va DENTRO del
    pipeline para que el modelo empaquetado funcione con archivos nuevos tal como
    salen del origen, sin pasar por la fase de preparación.
    """

    def __init__(self, date_cols: list[str] | None = None,
                 numeric_cols: list[str] | None = None):
        self.date_cols = date_cols or []
        self.numeric_cols = numeric_cols or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = pd.DataFrame(X).copy()
        for c in self.numeric_cols:
            if c in out.columns:
                out[c] = to_numeric_series(out[c])
        for c in self.date_cols:
            if c not in out.columns:
                continue
            s = to_datetime_series(out[c])
            out[f"{c}__anio"] = s.dt.year.astype("float64")
            out[f"{c}__mes"] = s.dt.month.astype("float64")
            out[f"{c}__dia_semana"] = s.dt.dayofweek.astype("float64")
            out = out.drop(columns=[c])
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else [])


def date_feature_names(date_cols: list[str]) -> list[str]:
    return [f"{c}__{suf}" for c in date_cols for suf in ("anio", "mes", "dia_semana")]


def build_preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    try:
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01,
                            sparse_output=False)
    except TypeError:  # sklearn antiguo
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", ohe)]), cat),
        ],
        remainder="drop", verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------- fuga de información


def detect_leakage(X: pd.DataFrame, y: pd.Series, num: list[str], cat: list[str],
                   is_classification: bool) -> list[str]:
    alerts = []
    if is_classification:
        yy = y.astype(str)
        for c in cat:
            tab = pd.crosstab(X[c].astype(str), yy)
            if len(tab) < 2:
                continue
            puros = (tab.gt(0).sum(axis=1) == 1).mean()
            if puros > 0.98 and X[c].nunique() > 2:
                alerts.append(
                    f"'{c}' determina casi por completo el objetivo (cada categoría cae en una sola "
                    "clase). Es fuga de información: probablemente se genera DESPUÉS de conocer el "
                    "resultado."
                )
        for c in num:
            try:
                grupos = [g.values for _, g in to_numeric_series(X[c]).groupby(yy) if len(g) > 2]
                if len(grupos) > 1:
                    rangos = [(np.nanmin(g), np.nanmax(g)) for g in grupos]
                    rangos.sort()
                    if all(rangos[i][1] < rangos[i + 1][0] for i in range(len(rangos) - 1)):
                        alerts.append(
                            f"'{c}' separa las clases sin traslape alguno. Casi siempre significa "
                            "que esa columna se deriva del objetivo."
                        )
            except Exception:
                continue
    else:
        yn = to_numeric_series(y)
        for c in num:
            try:
                r = to_numeric_series(X[c]).corr(yn)
                if pd.notna(r) and abs(r) > 0.98:
                    alerts.append(
                        f"'{c}' tiene correlación {r:.3f} con el objetivo: es la misma variable "
                        "con otro nombre o un componente directo de ella."
                    )
            except Exception:
                continue
    return alerts


# ---------------------------------------------------------------- supervisado


def infer_task(y: pd.Series) -> str:
    from pandas.api import types as pt
    if pt.is_numeric_dtype(y) and not pt.is_bool_dtype(y):
        nun = y.nunique()
        if nun > 20 or (nun > 10 and y.dtype.kind == "f"):
            return REGRESION
    return CLASIFICACION


def run_supervised(df: pd.DataFrame, profiles: dict, target: str,
                   test_size: float = 0.25, random_state: int = 42) -> ModelReport:
    data = df.dropna(subset=[target]).copy()
    if len(data) < 30:
        return ModelReport(CLASIFICACION, False, "Datos insuficientes",
                           "Se necesitan al menos 30 registros con el objetivo definido; "
                           f"solo hay {len(data)}.")

    y_raw = data[target]
    task = infer_task(y_raw)
    is_clf = task == CLASIFICACION

    num, cat, dropped = usable_features(data, profiles, target)
    date_cols = [c for c in data.columns
                 if c != target and profiles.get(c) is not None
                 and profiles[c].semantic == "fecha"
                 and to_datetime_series(data[c]).notna().mean() >= 0.5]
    dropped = [d for d in dropped if not d.startswith(tuple(f"{c} (fecha" for c in date_cols))]
    derived = date_feature_names(date_cols)
    num_model = list(dict.fromkeys([c for c in num if c != target] + derived))
    if not num_model and not cat:
        return ModelReport(task, False, "No hay variables predictoras utilizables",
                           "Todas las columnas quedaron descartadas por ser identificadores, "
                           "texto libre o constantes. Descartadas: " + ", ".join(dropped[:8]) + ".")

    X = data[[c for c in num if c != target] + cat + date_cols]
    warns: list[str] = []
    if dropped:
        warns.append("Columnas excluidas del modelo: " + ", ".join(dropped[:10]) +
                     ("…" if len(dropped) > 10 else "") + ".")

    if is_clf:
        y = y_raw.astype(str)
        vc = y.value_counts()
        raras = vc[vc < 5]
        if len(raras):
            X = X[~y.isin(raras.index)]
            y = y[~y.isin(raras.index)]
            warns.append(f"Se excluyeron {len(raras)} clase(s) con menos de 5 casos "
                         f"({', '.join(map(str, raras.index[:5]))}): no se pueden validar.")
        if y.nunique() < 2:
            return ModelReport(task, False, "El objetivo tiene una sola clase utilizable",
                               "No hay nada que predecir.")
        if vc.max() / len(y) > 0.9:
            warns.append(f"Clases muy desbalanceadas: '{vc.index[0]}' es el "
                         f"{vc.max()/len(y):.0%} de los casos. Mira la exactitud balanceada y el "
                         "F1, no la exactitud simple.")
    else:
        y = to_numeric_series(y_raw)
        mask = y.notna()
        X, y = X[mask], y[mask]
        if y.nunique() < 5:
            warns.append("El objetivo numérico tiene muy pocos valores distintos; quizá deberías "
                         "tratarlo como clasificación.")

    leak = detect_leakage(X, y, [c for c in num if c != target], cat, is_clf)

    strat = y if (is_clf and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    pre = build_preprocessor(num_model, cat)
    expander = FrameCoercer(date_cols, [c for c in num if c != target])

    if is_clf:
        candidates = {
            "Regresión logística": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=250, random_state=random_state, class_weight="balanced_subsample",
                min_samples_leaf=2, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        }
        dummy = DummyClassifier(strategy="most_frequent")
        scoring = "balanced_accuracy" if y.nunique() > 1 else "accuracy"
        cv = StratifiedKFold(n_splits=min(5, int(y_tr.value_counts().min())) or 2,
                             shuffle=True, random_state=random_state)
        cv = cv if cv.get_n_splits() >= 2 else StratifiedKFold(n_splits=2)
    else:
        candidates = {
            "Regresión Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=250, random_state=random_state, min_samples_leaf=2, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        }
        dummy = DummyRegressor(strategy="mean")
        scoring = "r2"
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)

    resultados = {}
    for name, est in candidates.items():
        pipe = Pipeline([("fechas", expander), ("pre", pre), ("est", est)])
        try:
            scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
            resultados[name] = (float(np.mean(scores)), float(np.std(scores)), pipe)
        except Exception as e:  # noqa: BLE001
            warns.append(f"{name} no pudo entrenarse ({type(e).__name__}).")

    if not resultados:
        return ModelReport(task, False, "Ningún modelo pudo entrenarse",
                           "Revisa los tipos de las columnas seleccionadas.", warnings=warns)

    best_name = max(resultados, key=lambda k: resultados[k][0])
    cv_mean, cv_std, best_pipe = resultados[best_name]
    best_pipe.fit(X_tr, y_tr)
    y_pred = best_pipe.predict(X_te)

    base_pipe = Pipeline([("fechas", clone(expander)), ("pre", clone(pre)), ("est", dummy)])
    base_pipe.fit(X_tr, y_tr)
    y_base = base_pipe.predict(X_te)

    if is_clf:
        metrics = {
            "Exactitud": accuracy_score(y_te, y_pred),
            "Exactitud balanceada": balanced_accuracy_score(y_te, y_pred),
            "F1 (macro)": f1_score(y_te, y_pred, average="macro", zero_division=0),
        }
        if y.nunique() == 2:
            try:
                proba = best_pipe.predict_proba(X_te)[:, 1]
                metrics["ROC AUC"] = roc_auc_score((y_te == best_pipe.classes_[1]).astype(int), proba)
            except Exception:
                pass
        baseline = {
            "Exactitud": accuracy_score(y_te, y_base),
            "Exactitud balanceada": balanced_accuracy_score(y_te, y_base),
            "F1 (macro)": f1_score(y_te, y_base, average="macro", zero_division=0),
        }
        key = "Exactitud balanceada"
        beats = metrics[key] > baseline[key] + 0.02
        labels = sorted(pd.unique(pd.concat([pd.Series(y_te), pd.Series(y_pred)]).astype(str)))
        cm = confusion_matrix(y_te.astype(str), pd.Series(y_pred).astype(str), labels=labels)
        extra = {"confusion": cm, "labels": labels,
                 "cv_mean": cv_mean, "cv_std": cv_std, "modelo": best_name,
                 "n_train": len(X_tr), "n_test": len(X_te)}
        headline = (f"{best_name}: exactitud balanceada {fmt_pct(metrics[key])} "
                    f"(baseline {fmt_pct(baseline[key])})")
    else:
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        metrics = {
            "R²": r2_score(y_te, y_pred),
            "MAE": mean_absolute_error(y_te, y_pred),
            "RMSE": rmse,
        }
        denom = np.where(np.abs(y_te) < 1e-9, np.nan, np.abs(y_te))
        mape = float(np.nanmean(np.abs((y_te - y_pred) / denom)))
        if np.isfinite(mape):
            metrics["MAPE"] = mape
        baseline = {
            "R²": r2_score(y_te, y_base),
            "MAE": mean_absolute_error(y_te, y_base),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, y_base))),
        }
        key = "R²"
        beats = metrics[key] > max(baseline[key], 0) + 0.05
        extra = {"y_true": np.asarray(y_te, dtype=float), "y_pred": np.asarray(y_pred, dtype=float),
                 "cv_mean": cv_mean, "cv_std": cv_std, "modelo": best_name,
                 "n_train": len(X_tr), "n_test": len(X_te)}
        headline = f"{best_name}: R² {metrics['R²']:.3f}, MAE {fmt_num(metrics['MAE'])}"

    # importancia por permutación (mide utilidad real, no el sesgo del árbol)
    importance = None
    try:
        pi = permutation_importance(best_pipe, X_te, y_te, n_repeats=8,
                                    random_state=random_state, n_jobs=1,
                                    scoring=scoring)
        importance = (pd.DataFrame({"Variable": X.columns, "Importancia": pi.importances_mean})
                      .sort_values("Importancia", ascending=False).head(15))
    except Exception:
        pass

    if cv_std > abs(cv_mean) * 0.5 and cv_mean > 0:
        warns.append(f"El desempeño varía mucho entre particiones (±{cv_std:.3f} sobre "
                     f"{cv_mean:.3f}). Con tan pocos datos el resultado no es estable.")
    if len(data) < 200:
        warns.append(f"Solo {len(data):,} registros. Cualquier métrica aquí tiene un margen de "
                     "error amplio; trátala como indicio, no como medición.")
    warns = leak + warns

    if leak:
        verdict = ("⚠️ **No confíes en estas métricas todavía.** Se detectó posible fuga de "
                   "información: alguna variable predictora contiene, de forma directa, la "
                   "respuesta. Quita esas columnas y vuelve a entrenar.")
    elif not beats:
        verdict = ("❌ **El modelo no le gana a la regla tonta.** Predecir siempre "
                   + ("la clase más común" if is_clf else "el promedio") +
                   " da prácticamente el mismo resultado. Con estas variables el objetivo no es "
                   "predecible: no lo pongas en producción.")
    elif is_clf and metrics["Exactitud balanceada"] < 0.6:
        verdict = ("🟡 **Le gana al baseline, pero apenas.** Sirve para priorizar (ordenar casos "
                   "por probabilidad), no para decidir automáticamente.")
    elif not is_clf and metrics["R²"] < 0.4:
        verdict = (f"🟡 **Explica solo el {metrics['R²']:.0%} de la variación.** Útil como señal "
                   "direccional; el error típico sigue siendo grande.")
    else:
        verdict = ("✅ **Supera claramente al baseline.** Valida los resultados contra tu "
                   "conocimiento del negocio antes de operarlo.")

    return ModelReport(task, True, headline, verdict, metrics, baseline, beats,
                       warns, importance, extra, best_pipe, list(X.columns))


# ---------------------------------------------------------------- no supervisado


def run_clustering(df: pd.DataFrame, profiles: dict, k: int | None = None,
                   random_state: int = 42) -> ModelReport:
    num, cat, dropped = usable_features(df, profiles, None)
    if len(num) + len(cat) < 2:
        return ModelReport(CLUSTERING, False, "Variables insuficientes",
                           "Se necesitan al menos 2 columnas numéricas o categóricas utilizables.")
    X = df[num + cat].copy()
    X = X.dropna(how="all")
    if len(X) < 20:
        return ModelReport(CLUSTERING, False, "Datos insuficientes",
                           f"Solo hay {len(X)} registros utilizables; se necesitan al menos 20.")
    pre = Pipeline([("coerce", FrameCoercer([], num)),
                    ("ct", build_preprocessor(num, cat))])
    Z = pre.fit_transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()

    scores = {}
    rango = [k] if k else range(2, min(9, max(3, len(X) // 10)))
    for kk in rango:
        if kk >= len(X):
            continue
        km = KMeans(n_clusters=kk, n_init=10, random_state=random_state)
        lab = km.fit_predict(Z)
        if len(set(lab)) < 2:
            continue
        scores[kk] = (float(silhouette_score(Z, lab)), km, lab)
    if not scores:
        return ModelReport(CLUSTERING, False, "No se pudo segmentar",
                           "Los datos no permitieron formar grupos distinguibles.")
    best_k = max(scores, key=lambda kk: scores[kk][0])
    sil, km, labels = scores[best_k]

    perfil = df.loc[X.index].copy()
    perfil["_segmento"] = [f"Segmento {i+1}" for i in labels]
    resumen = perfil.groupby("_segmento").agg(
        {**{c: "mean" for c in num}, **({cat[0]: (lambda s: s.mode().iloc[0] if len(s.mode()) else "—")} if cat else {})}
    )
    resumen.insert(0, "Registros", perfil["_segmento"].value_counts().reindex(resumen.index))

    if sil >= 0.5:
        verdict = ("✅ **Grupos bien separados** (silueta ≥ 0.5). La segmentación refleja "
                   "estructura real en los datos.")
    elif sil >= 0.25:
        verdict = ("🟡 **Separación moderada.** Los grupos existen pero se traslapan; úsalos como "
                   "guía, no como categorías estrictas.")
    else:
        verdict = ("❌ **Los grupos no están realmente separados** (silueta < 0.25). Estos datos "
                   "no tienen segmentos naturales: forzarlos produciría conclusiones inventadas.")

    return ModelReport(
        CLUSTERING, True, f"{best_k} segmentos, silueta {sil:.2f}", verdict,
        {"Silueta": sil, "Segmentos": best_k}, {}, sil >= 0.25,
        [f"Columnas excluidas: {', '.join(dropped[:8])}."] if dropped else [],
        None, {"labels": perfil["_segmento"], "resumen": resumen, "sil_por_k":
               {kk: v[0] for kk, v in scores.items()}, "index": X.index},
        Pipeline([("pre", pre), ("est", km)]), num + cat,
    )


def run_anomalies(df: pd.DataFrame, profiles: dict, contamination: float = 0.02,
                  random_state: int = 42) -> ModelReport:
    num, cat, dropped = usable_features(df, profiles, None)
    if len(num) < 2:
        return ModelReport(ANOMALIAS, False, "Variables numéricas insuficientes",
                           "La detección de anomalías necesita al menos 2 columnas numéricas.")
    X = df[num + cat]
    if len(X) < 50:
        return ModelReport(ANOMALIAS, False, "Datos insuficientes",
                           f"Se necesitan al menos 50 registros; hay {len(X)}.")
    pre = Pipeline([("coerce", FrameCoercer([], num)),
                    ("ct", build_preprocessor(num, cat))])
    Z = pre.fit_transform(X)
    if hasattr(Z, "toarray"):
        Z = Z.toarray()
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=200)
    pred = iso.fit_predict(Z)
    score = iso.score_samples(Z)
    n_anom = int((pred == -1).sum())
    idx = df.index[pred == -1]
    return ModelReport(
        ANOMALIAS, True, f"{n_anom:,} registros atípicos ({n_anom/len(X):.1%})",
        ("Estos registros son raros **en conjunto**, considerando todas las variables a la vez "
         "— no solo por tener un valor alto en una columna. Revísalos manualmente: suelen ser "
         "errores de captura, fraude o casos legítimos excepcionales."),
        {"Anomalías": n_anom, "% del total": n_anom / len(X)}, {}, None,
        [f"Columnas excluidas: {', '.join(dropped[:8])}."] if dropped else [],
        None, {"index": idx, "score": pd.Series(score, index=df.index)},
        Pipeline([("pre", pre), ("est", iso)]), num + cat,
    )


def run_forecast(df: pd.DataFrame, date_col: str, value_col: str | None,
                 horizon: int = 6) -> ModelReport:
    """Pronóstico con Holt-Winters, validado contra un baseline estacional ingenuo."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    f = to_datetime_series(df[date_col])
    d = df.assign(_f=f).dropna(subset=["_f"]).set_index("_f").sort_index()
    if value_col and value_col in df.columns:
        s = to_numeric_series(d[value_col]).resample("ME").sum()
    else:
        s = d.resample("ME").size().astype(float)
    s = s.dropna()
    if len(s) < 12:
        return ModelReport(FORECAST, False, "Serie demasiado corta",
                           f"Se necesitan al menos 12 meses completos; hay {len(s)}. "
                           "Con menos historia, cualquier pronóstico es adivinanza.")

    n_test = max(3, min(6, len(s) // 4))
    train, test = s.iloc[:-n_test], s.iloc[-n_test:]
    warns: list[str] = []
    seasonal = "add" if len(train) >= 24 else None
    try:
        fit = ExponentialSmoothing(train, trend="add", seasonal=seasonal,
                                   seasonal_periods=12 if seasonal else None,
                                   initialization_method="estimated").fit()
        pred = fit.forecast(n_test)
    except Exception as e:  # noqa: BLE001
        return ModelReport(FORECAST, False, "El modelo de series de tiempo no convergió",
                           f"Detalle: {type(e).__name__}. Prueba con una serie más larga o regular.")
    if seasonal is None:
        warns.append("Menos de 24 meses de historia: se modeló tendencia sin componente estacional.")

    naive = pd.Series([float(train.iloc[-1])] * n_test, index=test.index)
    def _mape(a, b):
        den = np.where(np.abs(a) < 1e-9, np.nan, np.abs(a))
        return float(np.nanmean(np.abs((a - b) / den)))
    mape_m, mape_n = _mape(test.values, pred.values), _mape(test.values, naive.values)
    mae_m = float(mean_absolute_error(test, pred))
    beats = np.isfinite(mape_m) and np.isfinite(mape_n) and mape_m < mape_n

    try:
        full = ExponentialSmoothing(s, trend="add", seasonal=seasonal,
                                    seasonal_periods=12 if seasonal else None,
                                    initialization_method="estimated").fit()
        fut = full.forecast(horizon)
    except Exception:
        fut = pred

    verdict = ("✅ **Le gana al pronóstico ingenuo** (repetir el último valor): "
               f"error de {fmt_pct(mape_m)} contra {fmt_pct(mape_n)}."
               if beats else
               "❌ **No le gana a repetir el último valor conocido.** La serie no tiene patrón "
               f"aprovechable (error del modelo {fmt_pct(mape_m)} vs. {fmt_pct(mape_n)} del "
               "método ingenuo). Usa el promedio histórico y concéntrate en entender el negocio.")
    if mape_m > 0.3:
        warns.append(f"Error promedio del {fmt_pct(mape_m)}: el pronóstico es demasiado impreciso "
                     "para planeación fina.")

    return ModelReport(
        FORECAST, True, f"Pronóstico a {horizon} meses (MAPE {fmt_pct(mape_m)})", verdict,
        {"MAPE": mape_m, "MAE": mae_m}, {"MAPE ingenuo": mape_n}, beats, warns, None,
        {"serie": s, "test": test, "pred_test": pred, "futuro": fut, "horizonte": horizon},
        None, [date_col],
    )
