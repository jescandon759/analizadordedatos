"""Genera un dataset de demostración deliberadamente sucio, para probar la app."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_demo(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2024-01-01", "2025-12-20", freq="D")
    f = rng.choice(fechas, size=n)

    canales = rng.choice(
        ["Mayoreo", "mayoreo ", "Menudeo", "MENUDEO", "Online", "Distribuidor"],
        size=n, p=[0.24, 0.06, 0.28, 0.05, 0.22, 0.15],
    )
    productos = rng.choice(
        ["Espadín 750", "Espadín 375", "Tobalá 750", "Cuishe 750", "Barril 750", "Kit Regalo"],
        size=n, p=[0.34, 0.18, 0.14, 0.12, 0.12, 0.10],
    )
    estados = rng.choice(
        ["San Luis Potosí", "CDMX", "Nuevo León", "Jalisco", "Querétaro", "Guanajuato"],
        size=n, p=[0.30, 0.22, 0.16, 0.14, 0.10, 0.08],
    )

    base = {"Espadín 750": 480, "Espadín 375": 260, "Tobalá 750": 950,
            "Cuishe 750": 820, "Barril 750": 890, "Kit Regalo": 1450}
    precio = np.array([base[p] for p in productos], dtype=float)
    factor_canal = np.where(np.char.strip(np.char.lower(canales.astype(str))) == "mayoreo", 0.72, 1.0)
    cantidad = rng.integers(1, 13, size=n)

    dias = (pd.to_datetime(f) - pd.Timestamp("2024-01-01")).days.values
    tendencia = 1 + dias / 1400 * 0.35
    mes = pd.to_datetime(f).month.values
    estacional = np.where(np.isin(mes, [11, 12]), 1.45, np.where(np.isin(mes, [1, 2]), 0.8, 1.0))

    importe = precio * factor_canal * cantidad * tendencia * estacional * rng.normal(1, 0.12, n)
    importe = np.round(np.abs(importe), 2)
    costo = np.round(importe * rng.normal(0.58, 0.06, n), 2)

    df = pd.DataFrame({
        "Folio": [f"V{100000+i}" for i in range(n)],
        "Fecha": pd.to_datetime(f).strftime("%d/%m/%Y"),
        # los clientes siguen una ley de potencias: unos pocos concentran la mayoría
        "Cliente": [f"CLI-{c:04d}" for c in
                    np.minimum(rng.zipf(1.45, size=n), 190)],
        "Producto": productos,
        "Canal": canales,
        "Estado": estados,
        "Cantidad": cantidad,
        "Precio unitario": [f"$ {p:,.2f}" for p in precio * factor_canal],
        "Importe": [f"$ {v:,.2f}" for v in importe],
        "Costo": costo,
        "Vendedor": rng.choice(["Ana Ramírez", "Luis Ortega", "MarÃ­a Solís", "Jorge Pineda"],
                               size=n, p=[0.3, 0.28, 0.22, 0.20]),
        "Estatus": rng.choice(["Pagado", "Pendiente", "Cancelado"], size=n, p=[0.82, 0.12, 0.06]),
        "Notas": rng.choice([None, "", "Entrega en sucursal", "Cliente frecuente"],
                            size=n, p=[0.75, 0.10, 0.09, 0.06]),
        "Region": "Bajío",                                  # columna constante
        "Comentario interno": None,                          # columna vacía
    })

    # --- suciedad deliberada ---
    faltan = rng.choice(n, size=int(n * 0.07), replace=False)
    df.loc[faltan, "Costo"] = np.nan
    faltan2 = rng.choice(n, size=int(n * 0.04), replace=False)
    df.loc[faltan2, "Estado"] = np.nan

    extremos = rng.choice(n, size=6, replace=False)
    df.loc[extremos, "Importe"] = [f"$ {v:,.2f}" for v in rng.uniform(90000, 260000, 6)]

    dup = df.sample(28, random_state=3)
    df = pd.concat([df, dup], ignore_index=True)

    err = rng.choice(len(df), size=5, replace=False)
    df.loc[err, "Cantidad"] = -df.loc[err, "Cantidad"].abs()
    df.loc[rng.choice(len(df), size=4, replace=False), "Fecha"] = "31/12/1899"

    return df.sample(frac=1, random_state=11).reset_index(drop=True)
